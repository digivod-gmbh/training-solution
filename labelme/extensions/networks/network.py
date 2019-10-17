import os
import sys
import json
import warnings
import numpy as np
import traceback

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import utils as gutils
from gluoncv.data.transforms import image as timage
from gluoncv.utils import viz, export_block
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

from labelme.logger import logger
from labelme.extensions.thread import WorkerExecutor
from labelme.utils.map import Map


class AbortTrainingException(Exception):
    def __init__(self, epoch):
        self.epoch = epoch


class Network(WorkerExecutor):

    def __init__(self):
        super().__init__()
        self.monitor = NetworkMonitor()
        self.dataset_format = None

        self.net_name = None
        self.model_file_name = None
        self.network = 'network'
        self.files = {}

    def training(self):
        gutils.random.seed(self.args.seed)

        # Prepare network and data
        self.args.save_prefix += self.net_name
        if not self.args.validate_dataset:
            self.args.val_interval = sys.maxsize
        self.labels = self.train_dataset.getLabels()

        self.thread.update.emit(_('Loading model ...'), -1, -1)
        self.loadModel()

        self.thread.update.emit(_('Loading dataset ...'), -1, -1)
        self.loadDataset()

        self.thread.update.emit(_('Start training ...'), -1, -1)
        self.beforeTrain()
        last_full_epoch = self.trainBase()
        self.afterTrain(last_full_epoch)
        self.thread.update.emit(_('Finished training'), -1, -1)

    def getDefaultArgs(self):
        default_args = {
            'training_name': 'unknown',
            'train_dataset': '',
            'validate_dataset': '',
            'data_shape': 0,
            'batch_size': 8,
            'gpus': '0',
            'epochs': 10,
            'resume': '',
            'start_epoch': 0,
            'num_workers': 0,
            'learning_rate': self.getDefaultLearningRate(),
            'lr_decay': 0.1,
            'lr_decay_epoch': '160,180',
            'momentum': 0.9,
            'wd': 0.0005,
            'log_interval': 1,
            'save_prefix': '',
            'save_interval': 1,
            'val_interval': 1,
            'seed': 42,
            'num_samples': -1,
            'syncbn': False,
            'mixup': False,
            'no_mixup_epochs': 20,
            'early_stop_epochs': 0,
        }
        return default_args

    def setArgs(self, args):
        default_args = self.getDefaultArgs()
        self.args = default_args.copy()
        self.args.update(args)
        self.args = Map(self.args)
        logger.debug(self.args)

    # def setArgs(self, args):
    #     self.args = args

    def loadDataset(self):
        train_dataset = self.train_dataset.getDatasetForTraining()
        val_dataset = None
        if self.args.validate_dataset:
            val_dataset = self.val_dataset.getDatasetForTraining()
        
        self.eval_metric = self.getValidationMetric()
        
        if self.args.num_samples < 0:
            self.args.num_samples = len(train_dataset)
        if self.args.mixup:
            from gluoncv.data import MixupDetection
            train_dataset = MixupDetection(train_dataset)

        self.train_data, self.val_data = self.getDataloader(train_dataset, val_dataset)

    def getDataloader(self, train_dataset, val_dataset):
        raise NotImplementedError('Method getDataloader() needs to be implemented in subclasses')
        # return train_loader, val_loader (both of type gluon.data.DataLoader)

    def loadModel(self):
        self.ctx = self.getContext(self.args.gpus)
        self.net = gcv.model_zoo.get_model(self.net_name, pretrained=False, ctx=self.ctx)
        if self.args.resume.strip():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.net.initialize(ctx=self.ctx)
            self.net.reset_class(self.labels)
            self.net.load_parameters(self.args.resume.strip(), ctx=self.ctx)
        else:
            model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../networks/models'))
            weights_file = os.path.join(model_path, self.model_file_name)
            self.net.load_parameters(weights_file, ctx=self.ctx)
            self.net.reset_class(self.labels)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.net.initialize()

    def trainBase(self):
        try:
            last_full_epoch = self.train()
            return last_full_epoch
        except AbortTrainingException as e:
            return e.epoch

    def train(self):
        raise NotImplementedError('Method train() needs to be implemented in subclasses')
        # return last_full_epoch

    def checkTrainingAborted(self, epoch):
        if self.isAborted():
            self.saveTraining(self.network, epoch)
            self.checkAborted()

    def beforeTrain(self):
        # Save config & architecture before training
        from labelme.config import Training
        config_file = os.path.join(self.output_folder, Training.config('config_file'))
        files = list(self.files.values())
        self.saveConfig(config_file, files)
        self.saveTraining(self.network, 0)

        if self.args.early_stop_epochs > 0:
            self.monitor = NetworkMonitor(self.args.early_stop_epochs)

        self.thread.data.emit({
            'validation': {
                _('Waiting for first validation ...'): '',
            },
        })
        self.thread.update.emit(_('Start training ...'), 1, self.args.epochs + 2)

        self.checkAborted()
        logger.info('Start training from [Epoch {}]'.format(self.args.start_epoch))

    def afterTrain(self, last_full_epoch):
        self.saveTraining(self.network, last_full_epoch)

    def beforeEpoch(self, epoch, num_batches):
        self.thread.update.emit(_('Start training on epoch {} ...').format(epoch + 1), None, -1)
        self.thread.data.emit({
            'progress': {
                'epoch': epoch + 1,
                'epoch_max': self.args.epochs,
                'batch': 1,
                'batch_max': num_batches,
                'speed': 0,
            },
        })
        self.checkTrainingAborted(epoch)

    def afterEpoch(self, epoch):
        from labelme.config import Training
        config_file = os.path.join(self.output_folder, Training.config('config_file'))
        self.updateConfig(config_file, last_epoch=epoch+1)
        self.thread.update.emit(_('Finished training on epoch {}').format(epoch + 1), epoch + 2, -1)
        self.checkTrainingAborted(epoch)

    def beforeBatch(self, batch_idx, epoch, num_batches):
        self.checkTrainingAborted(epoch)

    def afterBatch(self, batch_idx, epoch, num_batches, learning_rate, speed, metrics):
        i = batch_idx
        if self.args.log_interval and not (i + 1) % self.args.log_interval:
            log_msg = '[Epoch {}/{}][Batch {}/{}], LR: {:.2E}, Speed: {:.3f} samples/sec'.format(
                epoch + 1, self.args.epochs, i + 1, num_batches, learning_rate, speed
            )
            update_msg = '{}\n{} {}, {} {}/{}, {}: {:.3f} {}\n'.format(
                _('Training ...'), _('Epoch'), epoch + 1, _('Batch'), i + 1, num_batches, _('Speed'), speed, _('samples/sec')
            )
            progress_metrics = {}
            for metric in metrics:
                name, loss = metric.get()
                msg = ', {}={:.3f}'.format(name, loss)
                log_msg += msg
                update_msg += msg
                progress_metrics[name] = loss
            logger.info(log_msg)

            self.thread.data.emit({
                'progress': {
                    'epoch': epoch + 1,
                    'epoch_max': self.args.epochs,
                    'batch': i + 1,
                    'batch_max': num_batches,
                    'speed': speed,
                    'metric': progress_metrics
                },
            })
            self.thread.update.emit(update_msg, None, -1)
        self.checkTrainingAborted(epoch)

    def validateEpoch(self, epoch, epoch_time, validate_params):
        self.checkTrainingAborted(epoch)
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

            map_name, mean_ap = self.validate(**validate_params)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation [{:.3f}sec]: \n{}'.format(epoch, epoch_time, val_msg))
            current_mAP = float(mean_ap[-1])

            val_data = {
                'validation': {}
            }
            for i, name in enumerate(map_name[:]):
                val_data['validation'][name] = mean_ap[i]
            self.thread.data.emit(val_data)

            # Early Stopping
            self.monitor.update(epoch, mean_ap[-1])
            if self.monitor.shouldStopEarly():
                raise AbortTrainingException(epoch)
        else:
            current_mAP = 0.
        return current_mAP

    def saveParams(self, best_map, current_map, epoch):
        current_map = float(current_map)
        prefix = os.path.join(self.output_folder, self.args.save_prefix)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
            with open(prefix + '_best_map.log', 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
        if self.args.save_interval and epoch % self.args.save_interval == 0:
            self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

    def validate(self, waitall=False, static_shape=False):
        self.eval_metric.reset()
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        if waitall:
            mx.nd.waitall()
        self.net.hybridize(static_alloc=static_shape, static_shape=static_shape)
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
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # update metric
            self.eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return self.eval_metric.get()

    def getValidationMetric(self, iou_thresh=0.5):
        val_metric = VOC07MApMetric(iou_thresh=iou_thresh, class_names=self.labels)
        #val_metric = VOCMApMetric(iou_thresh=iou_thresh, class_names=self.labels)
        #val_metric = COCODetectionMetric(iou_thresh=iou_thresh, class_names=self.labels)
        return val_metric

    def getGpuSizes(self):
        raise NotImplementedError('Method getGpuSizes() needs to be implemented in subclasses')

    def getDefaultLearningRate(self):
        raise NotImplementedError('Method getDefaultLearningRate() needs to be implemented in subclasses')

    def updateConfig(self, config_file, **kwargs):
        logger.debug('Update training config with data: {}'.format(kwargs))
        data = {}
        with open(config_file, 'r') as f:
            data = json.load(f)
        for key in kwargs.keys():
            data[key] = kwargs[key]
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
        logger.debug('Updated training config in file: {}'.format(config_file))

    def saveConfig(self, config_file, files):
        args = self.args.copy()
        del args['network']
        del args['resume']
        data = {
            'network': self.args.network,
            'files': files,
            'dataset': self.dataset_format,
            'labels': self.labels,
            'args': args,
        }
        logger.debug('Create training config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved training config in file: {}'.format(config_file))

    def loadConfig(self, config_file):
        logger.debug('Load training config from file: {}'.format(config_file))
        with open(config_file, 'r') as f:
            data = json.load(f)
            logger.debug('Loaded training config: {}'.format(data))
            return Map(data)
        raise Exception('Could not load training config from file {}'.format(config_file))

    def setOutputFolder(self, output_folder):
        self.output_folder = output_folder

    def setLabels(self, labels):
        self.labels = labels

    def setTrainDataset(self, dataset, dataset_format):
        self.train_dataset = dataset
        self.dataset_format = dataset_format

    def setValDataset(self, dataset):
        self.val_dataset = dataset

    def getContext(self, gpus=None):
        if gpus is None or gpus == '':
            return [mx.cpu()]
        ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
        try:
            tmp = mx.nd.array([1, 2, 3], ctx=ctx[0])
        except mx.MXNetError as e:
            ctx = [mx.cpu()]
            logger.error(traceback.format_exc())
            logger.warning('Unable to use GPU. Using CPU instead')
        logger.debug('Use context: {}'.format(ctx))
        return ctx

    def saveTraining(self, network_name, epoch=0):
        from labelme.config import Training
        config_file = os.path.join(self.output_folder, Training.config('config_file'))
        # Export weights to .params file
        export_block(os.path.join(self.output_folder, network_name), self.net, epoch=epoch, preprocess=True, layout='HWC')
        # Update config with last epoch
        modified_weights_file = self.files['weights'].replace('0000', '{:04d}'.format(epoch))
        files = [self.files['architecture'], modified_weights_file]
        self.updateConfig(config_file, files=files)

    def inference(self, input_image_file, labels, architecture_file, weights_file, args = None):
        default_args = {
            'threshold': 0.5,
            'print_top_n': 10,
        }
        tmp_args = default_args.copy()
        if args:
            tmp_args.update(args)
        args = Map(tmp_args)
        logger.debug('Try loading network from files "{}" and "{}"'.format(architecture_file, weights_file))

        self.checkAborted()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ctx = self.getContext()
            net = gluon.nn.SymbolBlock.imports(architecture_file, ['data'], weights_file, ctx=ctx)
            class_names = labels
            net.collect_params().reset_ctx(ctx)
            img = mx.image.imread(input_image_file)
            img = timage.resize_short_within(img, 608, max_size=1024, mult_base=1)

            self.checkAborted()
            self.thread.update.emit(None, -1, -1)

            def make_tensor(img):
                np_array = np.expand_dims(np.transpose(img, (0,1,2)),axis=0).astype(np.float32)
                return mx.nd.array(np_array)

            image = img.asnumpy().astype('uint8')
            x = make_tensor(image)
            cid, score, bbox = net(x)

            self.thread.data.emit({
                'files': {
                    'input_image_file': input_image_file,
                    'architecture_file': architecture_file,
                    'weights_file': weights_file,
                },
                'imgsize': [image.shape[0], image.shape[1]],
                'classid': cid.asnumpy().tolist(),
                'score': score.asnumpy().tolist(),
                'bbox': bbox.asnumpy().tolist(),
                'labels': labels,
            })
            self.thread.update.emit(None, -1, -1)

            n_top = args.print_top_n
            classes = cid[0][:n_top].asnumpy().astype('int32').flatten().tolist()
            scores = score[0][:n_top].asnumpy().astype('float32').flatten().tolist()
            result_str = '\n'.join(['class: {}, score: {}'.format(classes[i], scores[i]) for i in range(n_top)])
            logger.debug('Top {} inference results:\n {}'.format(n_top, result_str))

            #ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=class_names, thresh=args.threshold)
            #plt.show()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if self.ctx:
            for context in self.ctx:
                context.empty_cache()


class NetworkMonitor:

    def __init__(self, stop_epochs = 3, stop_threshold = 5):
        self.curent_epoch = 0
        self.best_epoch = -1
        self.best_validation_value = 0
        self.stop_epochs = stop_epochs
        self.stop_threshold = stop_threshold
        self.stop_epoch_count = 0

    def shouldStopEarly(self):
        return self.stop_epoch_count >= self.stop_epochs

    def update(self, epoch, validation_value):
        self.current_epoch = epoch
        if validation_value > self.best_validation_value:
            self.best_validation_value = validation_value
            self.best_epoch = epoch
            self.stop_epoch_count = 0
        else:
            # Loss of validation in percent (0-100)
            validation_loss = (self.best_validation_value - validation_value) / self.best_validation_value * 100
            if validation_loss >= self.stop_threshold:
                self.stop_epoch_count += 1
