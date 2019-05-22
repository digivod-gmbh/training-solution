import json

from labelme.logger import logger
from labelme.utils.map import Map
from labelme.extensions.networks import NetworkYoloV3

class Training():

    @staticmethod
    def config(key = None):
        config = {
            'config_file': 'config.json',
            'networks': {
                'yolov3_darknet': _('YoloV3 (Darknet53)'),
                'yolov3_mobilenet': _('YoloV3 (Mobilenet1.0)'),
                #'ssd512': _('SSD512'),
            },
            'objects': {
                'yolov3_darknet': lambda: NetworkYoloV3(architecture='darknet53'),
                'yolov3_mobilenet': lambda: NetworkYoloV3(architecture='mobilenet1.0'),
                #'ssd512': lambda: NetworkSSD(architecture='resnet50'),
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config
