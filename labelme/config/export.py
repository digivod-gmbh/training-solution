import json

from labelme.logger import logger
from labelme.utils.map import Map
import labelme.extensions.formats as formats

class Export():

    _filters = None
    _filter2format = None
    _extension2format = None

    @staticmethod
    def config(key = None):
        config = {
            'labels_file': 'labels.txt',
            'formats': {
                'imagerecord': _('ImageRecord'),
                'coco': _('COCO'),
                'voc': _('VOC'),
            },
            'extensions': {
                'imagerecord': '*.rec',
                'coco': '*.json',
                'voc': False,
            },
            'objects': {
                'imagerecord': lambda: formats.FormatImageRecord(),
                'coco': lambda: formats.FormatCoco(),
                'voc': lambda: formats.FormatVoc(),
            },
            'limits': {
                'max_num_labels': 20
            }
        }
        if key is not None:
            if key in config:
                return config[key]
            return None
        return config

    @staticmethod
    def invertDict(in_dict):
        inverted_dict = {}
        for key in in_dict:
            val = in_dict[key]
            if val in inverted_dict:
                logger.warning('Overwriting key {} with value: {}, previous value: {}'.format(val, key, inverted_dict[val]))
            inverted_dict[val] = key
        return inverted_dict

    @staticmethod
    def detectDatasetFormat(dataset_folder):
        objects = Export.config('objects')
        for key in objects:
            candidate = objects[key]()
            if candidate.isValidFormat(dataset_folder):
                return key
        return None
