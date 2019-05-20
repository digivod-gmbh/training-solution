import json

from labelme.logger import logger
from labelme.utils.map import Map
from labelme.extensions.networks import NetworkYoloV3, NetworkSSD

class Training():

    @staticmethod
    def config(key = None):
        config = {
            'config_file_extension': '.train',
            'networks': {
                'yolov3_darknet': _('YoloV3 (Darknet53)'),
                'yolov3_mobilenet': _('YoloV3 (Mobilenet1.0)'),
                'ssd512': _('SSD512'),
            },
            'objects': {
                'yolov3_darknet': lambda: NetworkYoloV3(architecture='darknet53'),
                'yolov3_mobilenet': lambda: NetworkYoloV3(architecture='mobilenet1.0'),
                'ssd512': lambda: NetworkSSD(architecture='resnet50'),
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config

    @staticmethod
    def create_training_config(config_file, network, dataset_format, label_list, datasets, args):
        data = {
            'network': network,
            'dataset_format': dataset_format,
            'label_list': label_list,
            'datasets': datasets,
            'args': args,
            'architecture': None,
            'weights': None,
        }
        logger.debug('Create training config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved training config in file: {}'.format(config_file))
    
    @staticmethod
    def update_training_config(config_file, new_data):
        old_data = {}
        with open(config_file, 'r') as f:
            old_data = json.loads(f.read())
            logger.debug('Loaded training config: {}'.format(old_data))
        data = old_data.copy()
        data.update(new_data)
        logger.debug('Update training config: {}'.format(new_data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved training config in file: {}'.format(config_file))

    @staticmethod
    def read_training_config(config_file):
        data = {}
        with open(config_file, 'r') as f:
            data = json.loads(f.read())
            logger.debug('Read training config: {}'.format(data))
        return Map(data)