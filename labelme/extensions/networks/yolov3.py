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
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential, export_block
from gluoncv.data.transforms import image as timage
from gluoncv.utils import download, viz

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.networks import Network


class NetworkYoloV3(Network):

    _network = 'yolo3'
    _files = {
        'architecture': '{}-symbol.json'.format(_network),
        'weights': '{}-0000.params'.format(_network), 
    }

    def __init__(self, architecture='darknet53'):
        super().__init__()
        if architecture == 'darknet53':
            self.net_name = 'yolo3_darknet53_coco'
            self.model_file_name = 'yolo3_darknet53_coco-09767802.params'
        elif architecture == 'mobilenet1.0':
            self.net_name = 'yolo3_mobilenet1.0_coco'
            self.model_file_name = 'yolo3_mobilenet1.0_coco-66dbbae6.params'
        else:
            raise Exception('Unknown architecture {}'.format(architecture))

    def training(self):
        self.prepare()
        self.thread.update.emit(_('Start training ...'), -1, -1)
        self.train()

        # export
        training_name = '{}_{}'.format(self.args.training_name, self.net_name)
        export_block(os.path.join(self.output_folder, NetworkYoloV3._network), self.net, preprocess=True, layout='HWC')

        # save
        from labelme.config import Training
        config_file = os.path.join(self.output_folder, Training.config('config_file'))
        files = list(NetworkYoloV3._files.values())
        #self.args.dataset_folder
        self.saveConfig(config_file, NetworkYoloV3._network, files, '', self.labels, self.args)

        self.thread.update.emit(_('Finished training'), -1, -1)

    def setArgs(self, args):
        default_args = {
            'training_name': 'unknown',
            'train_dataset': '',
            'validate_dataset': '',
            'data_shape': 608, # 320, 416, 608
            'batch_size': 8,
            'gpus': '0',
            'epochs': 10,
            'resume': '',
            'start_epoch': 0,
            'num_workers': 0,
            'lr': 0.0001,
            'lr_mode': 'step',
            'lr_decay': 0.1,
            'lr_decay_period': 0,
            'lr_decay_epoch': '160,180',
            'warmup_lr': 0.0,
            'warmup_epochs': 0,
            'momentum': 0.9,
            'wd': 0.0005,
            'log_interval': 1,
            'save_prefix': '',
            'save_interval': 1,
            'val_interval': 1,
            'seed': 42,
            'num_samples': -1,
            'syncbn': False,
            'no_random_shape': True,
            'no_wd': False,
            'mixup': False,
            'no_mixup_epochs': 20,
            'pretrained': 0,
            'label_smooth': False,
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

        # use sync bn if specified
        num_sync_bn_devices = len(self.ctx) if self.args.syncbn else -1
        
        classes = self.train_dataset.getLabels()
        #classes = self.readLabelFile(self.label_file)
        self.labels = classes

        self.net = get_model(self.net_name, pretrained=False, ctx=self.ctx)

        self.thread.update.emit(_('Loading weights ...'), -1, -1)

        if self.args.resume.strip():
            self.net.load_parameters(self.args.resume.strip())
            async_net.load_parameters(self.args.resume.strip())
        else:
            model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../networks/models'))
            weights_file = os.path.join(model_path, self.model_file_name)
            self.net.load_parameters(weights_file, ctx=self.ctx)
            self.net.reset_class(classes)
            async_net = self.net
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize()
                async_net.initialize()
    
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

        # Metrics:
        # - VOC07MApMetric
        # - VOCMApMetric
        # - COCODetectionMetric
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
        #val_metric = VOCMApMetric(iou_thresh=0.5, class_names=classes)
        #val_metric = COCODetectionMetric(iou_thresh=0.5, class_names=classes)
        
        if self.args.num_samples < 0:
            self.args.num_samples = len(train_dataset)
        if self.args.mixup:
            from gluoncv.data import MixupDetection
            train_dataset = MixupDetection(train_dataset)

        return train_dataset, val_dataset, val_metric
    
    def get_dataloader(self, net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
        """Get dataloader."""
        width, height = data_shape, data_shape
        batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
        if self.args.no_random_shape:
            logger.debug("no random shape")
            train_loader = gluon.data.DataLoader(
                train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=self.args.mixup)),
                batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
        else:
            logger.debug("with random shape")
            transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=self.args.mixup) for x in range(10, 20)]
            train_loader = RandomTransformDataLoader(
                transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
                shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = None
        if val_dataset is not None:
            val_loader = gluon.data.DataLoader(val_dataset.transform(YOLO3DefaultValTransform(width, height)),
                batch_size, True, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
        return train_loader, val_loader
    
    def save_params(self, best_map, current_map, epoch, save_interval, prefix):
        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
            with open(prefix+'_best_map.log', 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
        if save_interval and epoch % save_interval == 0:
            self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
    
    def validate(self):
        """Test on validation dataset."""
        self.eval_metric.reset()
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
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
    
    def train(self):
        """Training pipeline"""
        self.net.collect_params().reset_ctx(self.ctx)
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
        num_batches = self.args.num_samples // self.args.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self.args.lr,
                        nepochs=self.args.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self.args.lr_mode, base_lr=self.args.lr,
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
    
        logger.info('Start training from [Epoch {}]'.format(self.args.start_epoch))
        best_map = [0.]
        epoch_count = 0
        progress_start = 4

        self.checkAborted()

        for epoch in range(self.args.start_epoch, self.args.epochs):

            self.thread.update.emit(_('Start training on epoch {} ...').format(epoch + 1), None, -1)
            self.checkAborted()
            epoch_count += 1

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
            for i, batch in enumerate(self.train_data):
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
                if self.args.log_interval and not (i + 1) % self.args.log_interval:
                    name1, loss1 = obj_metrics.get()
                    name2, loss2 = center_metrics.get()
                    name3, loss3 = scale_metrics.get()
                    name4, loss4 = cls_metrics.get()
                    logger.info('[Epoch {}][Batch {}/{}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i + 1, num_batches, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))

                    self.thread.update.emit(_('Training ...\nEpoch {}, Batch {}/{}, Speed: {:.3f} samples/sec\n{}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}')
                        .format(epoch + 1, i + 1, num_batches, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4), None, -1)

                
                self.thread.update.emit(None, -1, -1)
                self.checkAborted()

                btic = time.time()
    
            name1, loss1 = obj_metrics.get()
            name2, loss2 = center_metrics.get()
            name3, loss3 = scale_metrics.get()
            name4, loss4 = cls_metrics.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            if self.val_data and not (epoch + 1) % self.args.val_interval:
                logger.debug('validate: {}'.format(epoch + 1))
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = self.validate()
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            self.save_params(best_map, current_map, epoch, self.args.save_interval, os.path.join(self.output_folder, self.args.save_prefix))
            if current_map > best_map[0]:
                best_map[0] = current_map

        param_file = '{}.params'.format(self.args.training_name)
        self.net.save_parameters(os.path.join(self.output_folder, param_file))
