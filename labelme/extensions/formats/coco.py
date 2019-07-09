from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config import Export


class FormatCoco(DatasetFormat):

    _files = {
        'annotations': 'annotations.json',
    }
    _format = 'coco'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatCoco._files['annotations'],
        ]
        FormatCoco._files['labels'] = Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        pass

    def getValFile(self, dataset_path):
        pass

    def import_folder(self, folder):
        pass

    def export(self):
        pass


