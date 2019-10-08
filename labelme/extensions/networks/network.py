import os
import json
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from gluoncv.data.transforms import image as timage
from gluoncv.utils import viz, export_block

from labelme.logger import logger
from labelme.extensions.thread import WorkerExecutor
from labelme.utils.map import Map


class Network(WorkerExecutor):

    def __init__(self):
        super().__init__()

    def training(self):
        raise NotImplementedError('Method training() needs to be implemented in subclasses')

    def saveConfig(self, config_file, network, files, dataset, labels, args):
        data = {
            'network': network,
            'files': files,
            'dataset': dataset,
            'labels': labels,
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

    # def setLabelFile(self, label_file):
    #     self.label_file = label_file

    # def readLabelFile(self, label_file):
    #     with open(label_file) as f:
    #         return f.read().split('\n')

    def setLabels(self, labels):
        self.labels = labels

    def setTrainDataset(self, dataset):
        self.train_dataset = dataset

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
            logger.error(e)
            logger.warning('Unable to use GPU. Using CPU instead')
        logger.debug('Use context: {}'.format(ctx))
        return ctx

    def saveTraining(self, network_name):
        export_block(os.path.join(self.output_folder, network_name), self.net, preprocess=True, layout='HWC')

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
