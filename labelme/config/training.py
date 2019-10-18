import json

from labelme.logger import logger
from labelme.utils.map import Map
from labelme.extensions.networks import NetworkYoloV3, NetworkSSD512

class Training():

    @staticmethod
    def config(key = None):
        config = {
            'config_file': 'config.json',
            'networks': {
                'yolov3_darknet53': _('YoloV3 (Darknet53)'),
                'yolov3_mobilenet10': _('YoloV3 (Mobilenet1.0)'),
                'ssd512_resnet50': _('SSD512 (Resnet50)'),
                'ssd512_vgg16': _('SSD512 (VGG16)'),
                'ssd512_mobilenet10': _('SSD512 (Mobilenet1.0)'),
            },
            'objects': {
                'yolov3_darknet53': lambda: NetworkYoloV3(architecture='darknet53'),
                'yolov3_mobilenet10': lambda: NetworkYoloV3(architecture='mobilenet1.0'),
                'ssd512_resnet50': lambda: NetworkSSD512(architecture='resnet50'),
                'ssd512_vgg16': lambda: NetworkSSD512(architecture='vgg16_atrous'),
                'ssd512_mobilenet10': lambda: NetworkSSD512(architecture='mobilenet1.0'),
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config
