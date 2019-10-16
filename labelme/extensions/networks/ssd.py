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

    _network = 'ssd512'
    _files = {
        'architecture': '{}-symbol.json'.format(_network),
        'weights': '{}-0000.params'.format(_network), 
    }

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

    def training(self):
        self.prepare()

        # save config & architecture before training
        from labelme.config import Training
        config_file = os.path.join(self.output_folder, Training.config('config_file'))
        files = list(NetworkSSD512._files.values())
        self.saveConfig(config_file, files)
        self.saveTraining(NetworkSSD512._network, 0)

        self.thread.update.emit(_('Start training ...'), -1, -1)
        last_epoch = self.train()

        # export trained weights
        self.saveTraining(NetworkSSD512._network, last_epoch)

        self.thread.update.emit(_('Finished training'), -1, -1)

    def setArgs(self, args):
        default_args = {
            'training_name': 'unknown',
            'train_dataset': '',
            'validate_dataset': '',
            'data_shape': 512,
            'batch_size': 16,
            'gpus': '0',
            'epochs': 10,
            'resume': '',
            'start_epoch': 0,
            'num_workers': 0,
            'learning_rate': 0.0001,
            'mixup': False,
            'no_mixup_epochs': 20,
            'wd': 0.0005,
            'momentum': 0.9,
            'log_interval': 1,
            'save_prefix': '',
            'save_interval': 1,
            'val_interval': 1,
            'seed': 42,
            'num_samples': -1,
            'early_stop_epochs': 0,
        }
        self.args = default_args.copy()
        self.args.update(args)
        self.args = Map(self.args)
        logger.debug(self.args)

    def prepare(self):
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self.args.seed)

        if not self.args.validate_dataset:
            self.args.val_interval = sys.maxsize

        self.thread.update.emit(_('Loading model ...'), -1, -1)

        self.ctx = self.getContext(self.args.gpus)

        # network
        self.args.save_prefix += self.net_name

        classes = self.train_dataset.getLabels()
        self.labels = classes

        self.net = get_model(self.net_name, pretrained=False, ctx=self.ctx)

        self.thread.update.emit(_('Loading weights ...'), -1, -1)

        if self.args.resume.strip():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.net.initialize(ctx=self.ctx)
            self.net.reset_class(classes)
            self.net.load_parameters(self.args.resume.strip(), ctx=self.ctx)
        else:
            model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../networks/models'))
            weights_file = os.path.join(model_path, self.model_file_name)
            self.net.load_parameters(weights_file, ctx=self.ctx)
            self.net.reset_class(classes)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.net.initialize()
    
        self.thread.update.emit(_('Loading dataset ...'), -1, -1)

        # training data
        train_dataset, val_dataset, self.eval_metric = self.get_dataset()
        self.train_data, self.val_data = self.get_dataloader(self.net, train_dataset, val_dataset, self.args.data_shape, self.args.batch_size, self.args.num_workers)
    
    def get_dataset(self):
        train_dataset = self.train_dataset.getDatasetForTraining()
        val_dataset = None
        if self.args.validate_dataset:
            val_dataset = self.val_dataset.getDatasetForTraining()
        classes = self.train_dataset.getLabels()

        val_metric = self.getValidationMetric(classes)
        
        if self.args.num_samples < 0:
            self.args.num_samples = len(train_dataset)
        if self.args.mixup:
            from gluoncv.data import MixupDetection
            train_dataset = MixupDetection(train_dataset)

        return train_dataset, val_dataset, val_metric
    
    def get_dataloader(self, net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
        width, height = data_shape, data_shape
        # use fake data to generate fixed anchors for target generation
        with autograd.train_mode():
            foo, bar, anchors = net(mx.nd.zeros((1, 3, height, width), self.ctx[0]))
        anchors = anchors.as_in_context(mx.cpu())
        batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = None
        if val_dataset is not None:
            val_loader = gluon.data.DataLoader(
                val_dataset.transform(SSDDefaultValTransform(width, height)),
                batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
        return train_loader, val_loader
    
    def validate(self, nms_thresh=0.45, nms_topk=400):
        self.eval_metric.reset()
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=nms_thresh, nms_topk=nms_topk)
        mx.nd.waitall()
        self.net.hybridize()
        for batch in self.val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else [None])
            # update metric
            self.eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return self.eval_metric.get()
    
    def save_params(self, best_map, current_map, epoch, save_interval, prefix):
        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
            with open(prefix+'_best_map.log', 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
        if save_interval and epoch % save_interval == 0:
            self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

    def train(self):
        if self.args.early_stop_epochs > 0:
            self.monitor = NetworkMonitor(self.args.early_stop_epochs)

        self.thread.data.emit({
            'validation': {
                _('Waiting for first validation ...'): '',
            },
        })

        self.net.collect_params().reset_ctx(self.ctx)
        num_batches = self.args.num_samples // self.args.batch_size

        trainer = gluon.Trainer(
            self.net.collect_params(), 'sgd',
            {'learning_rate': self.args.learning_rate, 'wd': self.args.wd, 'momentum': self.args.momentum})

        # metrics
        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        logger.info('Start training from [Epoch {}]'.format(self.args.start_epoch))
        best_map = [0.]

        self.checkAborted()

        start_time = time.time()

        for epoch in range(self.args.start_epoch, self.args.epochs):

            self.thread.update.emit(_('Start training on epoch {} ...').format(epoch + 1), None, -1)
            if self.isAborted():
                self.saveTraining(NetworkSSD512._network, epoch-1)
                self.checkAborted()

            self.thread.data.emit({
                'progress': {
                    'epoch': epoch + 1,
                    'epoch_max': self.args.epochs,
                    'batch': 1,
                    'batch_max': num_batches,
                    'speed': 0,
                },
            })

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

            ce_metric.reset()
            smoothl1_metric.reset()
            tic = time.time()
            btic = time.time()
            self.net.hybridize(static_alloc=True, static_shape=True)
            for i, batch in enumerate(self.train_data):
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

                if self.args.log_interval and not (i + 1) % self.args.log_interval:
                    name1, loss1 = ce_metric.get()
                    name2, loss2 = smoothl1_metric.get()
                    logger.info('[Epoch {}/{}][Batch {}/{}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                        epoch + 1, self.args.epochs, i + 1, num_batches, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2))

                    speed = batch_size / (time.time() - btic)
                    self.thread.data.emit({
                        'progress': {
                            'epoch': epoch + 1,
                            'epoch_max': self.args.epochs,
                            'batch': i + 1,
                            'batch_max': num_batches,
                            'speed': batch_size / (time.time() - btic),
                            'metric': {
                                name1: loss1,
                                name2: loss2,
                            }
                        },
                    })
                    self.thread.update.emit(_('Training ...\nEpoch {}, Batch {}/{}, Speed: {:.3f} samples/sec\n{}={:.3f}, {}={:.3f}')
                        .format(epoch + 1, i + 1, num_batches, batch_size/(time.time()-btic), name1, loss1, name2, loss2), None, -1)

                self.thread.update.emit(None, -1, -1)
                if self.isAborted():
                    self.saveTraining(NetworkSSD512._network, epoch)
                    self.checkAborted()

                btic = time.time()

            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch + 1, (time.time()-tic), name1, loss1, name2, loss2))
            if self.val_data and not (epoch + 1) % self.args.val_interval:
                logger.debug('validate: {}'.format(epoch + 1))

                self.thread.data.emit({
                    'validation': {
                        _('Validating...'): '',
                    },
                    'progress': {
                        'speed': 0,
                    }
                })

                # consider reduce the frequency of validation to save time
                map_name, mean_ap = self.validate()
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch + 1, val_msg))
                current_map = float(mean_ap[-1])
                val_data = {
                    'validation': {}
                }
                for i, name in enumerate(map_name[:]):
                    val_data['validation'][name] = mean_ap[i]
                self.thread.data.emit(val_data)

                # Early Stopping
                self.monitor.update(epoch, mean_ap[-1])
                if self.monitor.shouldStopEarly():
                    return epoch
            else:
                current_map = 0.

            self.save_params(best_map, current_map, epoch, self.args.save_interval, os.path.join(self.output_folder, self.args.save_prefix))

            from labelme.config import Training
            config_file = os.path.join(self.output_folder, Training.config('config_file'))
            self.updateConfig(config_file, last_epoch=epoch+1)

        return epoch
