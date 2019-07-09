from .format import DatasetFormat
from .intermediate import IntermediateFormat
from labelme.config import Export


class FormatVoc(DatasetFormat):

    _files = {
        'labels': 'class_names.txt',
    }
    _format = 'voc'

    def __init__(self):
        super().__init__()
        self.intermediate = None
        self.needed_files = [
            FormatVoc._files['labels'],
        ]
        FormatVoc._files['labels'] = 'class_names.txt' # Export.config('labels_file')

    def getTrainFile(self, dataset_path):
        pass

    def getValFile(self, dataset_path):
        pass

    def import_folder(self, folder):
        pass

    def export(self):
        pass

    
