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
from gluoncv.model_zoo import get_model

from gluoncv.utils import download, viz

from labelme.utils.map import Map
from labelme.logger import logger
from labelme.extensions.networks import Network


class NetworkSSD(Network):

    def __init__(self, architecture='resnet50'):
        super().__init__()
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
        self.architecture_filename = self.net_name
        self.weights_filename = self.net_name

    def init_training(self, thread, training_name, output_dir, classes_list, train_dataset, 
        validate_dataset='', 
        data_shape=512,
        batch_size=8, 
        gpus='0', 
        epochs=10, 
        resume='',
        start_epoch=0,
        num_workers=0,  
        lr=0.0001, 
        lr_decay=0.1, 
        lr_decay_epoch='160,200', 
        momentum=0.9, 
        wd=0.0005,
        log_interval=1, 
        save_prefix='', 
        save_interval=1, 
        val_interval=1, 
        seed=42, 
    ):
        self.thread = thread
        self.args = Map({
            'training_name': training_name,
            'output_dir': output_dir,
            'classes_list': classes_list,
            'train_dataset': train_dataset,
            'validate_dataset': validate_dataset,
            'data_shape': data_shape,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'gpus': gpus,
            'epochs': epochs,
            'resume': resume,
            'start_epoch': start_epoch,
            'lr': lr,
            'lr_decay': lr_decay,
            'lr_decay_epoch': lr_decay_epoch,
            'momentum': momentum,
            'wd': wd,
            'log_interval': log_interval,
            'save_prefix': save_prefix,
            'save_interval': save_interval,
            'val_interval': val_interval,
            'seed': seed,
        })
        logger.debug(self.args)
        self.architecture_filename = '{}_{}'.format(self.args.training_name, self.architecture_filename)
        self.weights_filename = '{}_{}'.format(self.args.training_name, self.weights_filename)

    def training(self):
        self.prepare()
        self.thread.update.emit(_('Start training ...'), 4, -1)
        self.train()
        training_name = '{}_{}'.format(self.args.training_name, self.net_name)
        self.net.export(os.path.join(self.args.output_dir, self.architecture_filename))
        self.thread.update.emit(_('Finished training'), self.args.epochs + 4, -1)

    def inference(self, input_image_file, classes_list, architecture_file, weights_file, args):
        logger.debug('Try loading network from files "{}" and "{}"'.format(architecture_file, weights_file))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ctx = self.get_context()
            net = gluon.nn.SymbolBlock.imports(architecture_file, ['data'], weights_file, ctx=ctx)
            classes = self.read_classes(classes_list)
            net.collect_params().reset_ctx(ctx)
            x, image = gcv.data.transforms.presets.ssd.load_test(input_image_file, args.data_shape)
            cid, score, bbox = net(x)
            ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes, thresh=0.5)
            plt.show()

    def prepare(self):
        pass