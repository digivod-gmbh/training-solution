import os
import sys
import time
import warnings
import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from mxnet.gluon import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform, SSDDefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.data.transforms import image as timage

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.networks import Network, NetworkMonitor


class NetworkSSD512(Network):

    def __init__(self, architecture='resnet50'):
        super().__init__()
        self.architecture_name = architecture
        if architecture == 'resnet50':
            self.net_name = 'ssd_512_resnet50_v1_coco'
            self.model_file_name = 'ssd_512_resnet50_v1_coco-c4835162.params'
        elif architecture == 'mobilenet1.0':
            self.net_name = 'ssd_512_mobilenet1.0_coco'
            self.model_file_name = 'ssd_512_mobilenet1.0_coco-da9756fa.params'
        elif architecture == 'vgg16_atrous':
            self.net_name = 'ssd_512_vgg16_atrous_coco'
            self.model_file_name = 'ssd_512_vgg16_atrous_coco-5c860642.params'
        else:
            raise Exception('Unknown architecture {}'.format(architecture))
        self.network = 'ssd512'
        self.files = {
            'architecture': '{}-symbol.json'.format(self.network),
            'weights': '{}-0000.params'.format(self.network), 
        }

    def getGpuSizes(self):
        # (base size, additional size per batch item)
        if self.architecture_name == 'resnet50':
            return (1500, 1000)
        elif self.architecture_name == 'mobilenet1.0':
            return (1100, 600)
        elif self.architecture_name == 'vgg16_atrous':
            return (1300, 1000)
        raise Exception('Unknown architecture {}'.format(self.architecture_name))

    def getDefaultLearningRate(self):
        return 0.001

    def getDefaultArgs(self):
        default_args = super().getDefaultArgs()
        default_args['data_shape'] = 512 # 300, 512
        default_args['lr_decay_epoch'] = '160,200'
        return default_args
    
    def getDataloader(self, train_dataset, val_dataset):
        width, height = self.args.data_shape, self.args.data_shape
        # use fake data to generate fixed anchors for target generation
        with autograd.train_mode():
            foo, bar, anchors = self.net(mx.nd.zeros((1, 3, height, width), self.ctx[0]))
        anchors = anchors.as_in_context(mx.cpu())
        batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
            self.args.batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=self.args.num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = None
        if val_dataset is not None:
            val_loader = gluon.data.DataLoader(
                val_dataset.transform(SSDDefaultValTransform(width, height)),
                self.args.batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=self.args.num_workers)
        return train_loader, val_loader

    def train(self):
        self.net.collect_params().reset_ctx(self.ctx)
        num_batches = self.args.num_samples // self.args.batch_size

        trainer = gluon.Trainer(
            self.net.collect_params(), 'sgd',
            {'learning_rate': self.args.learning_rate, 'wd': self.args.wd, 'momentum': self.args.momentum},
            update_on_kvstore=(None)
        )

        # Learning rate decay policy
        lr_decay = float(self.args.lr_decay)
        lr_steps = sorted([float(ls) for ls in self.args.lr_decay_epoch.split(',') if ls.strip()])

        # Losses
        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        best_map = [0.]

        # Epoch loop
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # Batch size can vary from epoch to epoch +/-1 
            num_batches = len(self.train_data)

            self.beforeEpoch(epoch, num_batches=num_batches)

            while lr_steps and epoch >= lr_steps[0]:
                new_lr = trainer.learning_rate * lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                logger.info('[Epoch {}] Set learning rate to {}'.format(epoch, new_lr))
            ce_metric.reset()
            smoothl1_metric.reset()

            tic = time.time()
            btic = time.time()
            self.net.hybridize(static_alloc=True, static_shape=True)

            # Batch loop
            for i, batch in enumerate(self.train_data):
                self.beforeBatch(i, epoch, num_batches)

                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)                                                       
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx, batch_axis=0)        
                            
                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, foo = self.net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    autograd.backward(sum_loss)

                trainer.step(1)
                ce_metric.update(0, [l * batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * batch_size for l in box_loss])

                speed = batch_size/(time.time()-btic)
                self.afterBatch(i, epoch, num_batches, trainer.learning_rate, speed,
                    metrics=[ce_metric, smoothl1_metric])
                btic = time.time()

            current_mAP = self.validateEpoch(epoch, epoch_time=(time.time()-tic), validate_params={'static_shape': True})
            self.saveParams(best_map, current_mAP, epoch)

            self.afterEpoch(epoch=epoch+1)

        return epoch
