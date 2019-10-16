import os
import json
import warnings
import numpy as np
import traceback

import mxnet as mx
from mxnet import gluon
from gluoncv.data.transforms import image as timage
from gluoncv.utils import viz, export_block
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

from labelme.logger import logger
from labelme.extensions.thread import WorkerExecutor
from labelme.utils.map import Map


class Network(WorkerExecutor):

    def __init__(self):
        super().__init__()
        self.monitor = NetworkMonitor()
        self.dataset_format = None

    def training(self):
        raise NotImplementedError('Method training() needs to be implemented in subclasses')

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

    def setArgs(self, args):
        self.args = args

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

    def getValidationMetric(self, classes):
        # Metrics:
        # - VOC07MApMetric
        # - VOCMApMetric
        # - COCODetectionMetric
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
        #val_metric = VOCMApMetric(iou_thresh=0.5, class_names=classes)
        #val_metric = COCODetectionMetric(iou_thresh=0.5, class_names=classes)
        return val_metric

    def saveTraining(self, network_name, epoch=0):
        export_block(os.path.join(self.output_folder, network_name), self.net, epoch=epoch, preprocess=True, layout='HWC')

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
