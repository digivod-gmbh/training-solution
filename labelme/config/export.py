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
            'config_file_extension': '.dataset',
            'formats': {
                'imagerecord': _('ImageRecord'),
            },
            'extensions': {
                'imagerecord': formats.FormatImageRecord.getExtension(),
            },
            'objects': {
                'imagerecord': lambda: formats.FormatImageRecord(),
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
    def create_dataset_config(config_file, dataset_format, label_list, args):
        data = {
            'format': dataset_format,
            'label_list': label_list,
            'args': args
        }
        logger.debug('Create dataset config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved dataset config in file: {}'.format(config_file))

    @staticmethod
    def update_dataset_config(config_file, new_data):
        old_data = {}
        with open(config_file, 'r') as f:
            old_data = json.loads(f.read())
            logger.debug('Loaded dataset config: {}'.format(old_data))
        data = old_data.copy()
        data.update(new_data)
        logger.debug('Update dataset config: {}'.format(new_data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved dataset config in file: {}'.format(config_file))

    @staticmethod
    def read_dataset_config(config_file):
        data = {}
        with open(config_file, 'r') as f:
            data = json.loads(f.read())
            logger.debug('Read dataset config: {}'.format(data))
        return Map(data)


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