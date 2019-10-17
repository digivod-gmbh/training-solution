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
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.data.transforms import image as timage

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.networks import Network, NetworkMonitor


class NetworkYoloV3(Network):

    def __init__(self, architecture='darknet53'):
        super().__init__()
        self.architecture_name = architecture
        if architecture == 'darknet53':
            self.net_name = 'yolo3_darknet53_coco'
            self.model_file_name = 'yolo3_darknet53_coco-09767802.params'
        elif architecture == 'mobilenet1.0':
            self.net_name = 'yolo3_mobilenet1.0_coco'
            self.model_file_name = 'yolo3_mobilenet1.0_coco-66dbbae6.params'
        else:
            raise Exception('Unknown architecture {}'.format(architecture))
        self.network = 'yolo3'
        self.files = {
            'architecture': '{}-symbol.json'.format(self.network),
            'weights': '{}-0000.params'.format(self.network), 
        }

    def getGpuSizes(self):
        # (base size, additional size per batch item)
        if self.architecture_name == 'darknet53':
            return (2500, 1000)
        elif self.architecture_name == 'mobilenet1.0':
            return (1300, 600)
        raise Exception('Unknown architecture {}'.format(self.architecture_name))

    def getDefaultLearningRate(self):
        return 0.0002

    def getDefaultArgs(self):
        default_args = super().getDefaultArgs()
        default_args['data_shape'] = 608 # 320, 416, 608
        default_args['lr_mode'] = 'step'
        default_args['lr_decay_period'] = 0
        default_args['warmup_lr'] = 0.0
        default_args['warmup_epochs'] = 0
        default_args['no_random_shape'] = True
        default_args['no_wd'] = False
        default_args['label_smooth'] = False
        return default_args
    
    def getDataloader(self, train_dataset, val_dataset):
        width, height = self.args.data_shape, self.args.data_shape
        batchify_fn = Tuple(*([Stack() for foo in range(6)] + [Pad(axis=0, pad_val=-1) for bar in range(1)]))  # stack image, all targets generated
        if self.args.no_random_shape:
            logger.debug('no random shape')
            train_loader = gluon.data.DataLoader(
                train_dataset.transform(YOLO3DefaultTrainTransform(width, height, self.net, mixup=self.args.mixup)),
                self.args.batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=self.args.num_workers)
        else:
            logger.debug('with random shape')
            transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, self.net, mixup=self.args.mixup) for x in range(10, 20)]
            train_loader = RandomTransformDataLoader(
                transform_fns, train_dataset, batch_size=self.args.batch_size, interval=10, last_batch='rollover',
                shuffle=True, batchify_fn=batchify_fn, num_workers=self.args.num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = None
        if val_dataset is not None:
            val_loader = gluon.data.DataLoader(val_dataset.transform(YOLO3DefaultValTransform(width, height)),
                self.args.batch_size, True, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=self.args.num_workers)
        return train_loader, val_loader
    
    def train(self):
        self.net.collect_params().reset_ctx(self.ctx)
        num_batches = self.args.num_samples // self.args.batch_size

        if self.args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if self.args.label_smooth:
            self.net._target_generator._label_smooth = True

        if self.args.lr_decay_period > 0:
            lr_decay_epoch = list(range(self.args.lr_decay_period, self.args.epochs, self.args.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self.args.lr_decay_epoch.split(',')]
        lr_decay_epoch = [e - self.args.warmup_epochs for e in lr_decay_epoch]
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self.args.learning_rate,
                        nepochs=self.args.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self.args.lr_mode, base_lr=self.args.learning_rate,
                        nepochs=self.args.epochs - self.args.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=self.args.lr_decay, power=2),
        ])

        trainer = gluon.Trainer(
            self.net.collect_params(), 'sgd',
            {'wd': self.args.wd, 'momentum': self.args.momentum, 'lr_scheduler': lr_scheduler},
            kvstore='local')

        # targets
        sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        l1_loss = gluon.loss.L1Loss()

        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')

        best_map = [0.]

        # Epoch loop
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # Batch size can vary from epoch to epoch +/-1 
            num_batches = len(self.train_data)

            self.beforeEpoch(epoch, num_batches=num_batches)

            if self.args.mixup:
                try:
                    self.train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    self.train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= self.args.epochs - self.args.no_mixup_epochs:
                    try:
                        self.train_data._dataset.set_mixup(None)
                    except AttributeError:
                        self.train_data._dataset._data.set_mixup(None)

            tic = time.time()
            btic = time.time()
            mx.nd.waitall()
            self.net.hybridize()

            # Batch loop
            for i, batch in enumerate(self.train_data):
                self.beforeBatch(i, epoch, num_batches)

                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.ctx, batch_axis=0)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []

                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = self.net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    autograd.backward(sum_losses)

                trainer.step(batch_size)
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)

                speed = batch_size/(time.time()-btic)
                self.afterBatch(i, epoch, num_batches, trainer.learning_rate, speed,
                    metrics=[obj_metrics, center_metrics, scale_metrics, cls_metrics])
                btic = time.time()

            current_mAP = self.validateEpoch(epoch, epoch_time=(time.time()-tic), validate_params={'waitall': True})
            self.saveParams(best_map, current_mAP, epoch)

            self.afterEpoch(epoch)

        return epoch
