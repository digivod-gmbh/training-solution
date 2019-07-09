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
            'config_file': 'config.json',
            'labels_file': 'labels.txt',
            'formats': {
                'imagerecord': _('ImageRecord'),
                'coco': _('COCO'),
                'voc': _('VOC'),
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
    def detectDatasetFormat(dataset_folder):
        objects = Export.config('objects')
        for key in objects:
            candidate = objects[key]()
            if candidate.isValidFormat(dataset_folder):
                return key
        return None

    @staticmethod
    def filters():
        Export.init_filters()
        return Export._filters

    @staticmethod
    def filter2format(key):
        Export.init_filters()
        if key in Export._filter2format:
            return Export._filter2format[key]
        return None

    @staticmethod
    def extension2format(key):
        Export.init_filters()
        if key in Export._extension2format:
            return Export._extension2format[key]
        return None

    @staticmethod
    def init_filters():
        if Export._filters is None or Export._filter2format is None:
            formats = Export.config('formats')
            extensions = Export.config('extensions')
            filters = []
            Export._filter2format = {}
            Export._extension2format = {}
            for key in formats:
                f = '{} (*{})'.format(formats[key], extensions[key])
                filters.append(f)
                Export._filter2format[f] = formats[key]
                ext = extensions[key]
                Export._extension2format[ext] = formats[key]
            Export._filters = ';;'.join(filters)