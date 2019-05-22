import json
import os

from labelme.logger import logger
from labelme.utils.map import Map
from labelme.extensions import ThreadExtension


class DatasetFormat(ThreadExtension):

    def export(self):
        raise NotImplementedError('Method export() must be implemented in subclass')

    def isValidFormat(self, dataset_folder):
        needed_files = self.needed_files.copy()
        for root, dirs, files in os.walk(dataset_folder):  
            for filename in files:
                try:
                    idx = needed_files.index(filename)
                    if idx > -1:
                        del needed_files[idx]
                except:
                    pass
        return len(needed_files) == 0

    def saveConfig(self, config_file, dataset_format, files, samples, args):
        data = {
            'format': dataset_format,
            'files': files,
            'samples': samples,
            'args': args,
        }
        logger.debug('Create dataset config: {}'.format(data))
        with open(config_file, 'w+') as f:
            json.dump(data, f, indent=2)
            logger.debug('Saved dataset config in file: {}'.format(config_file))

    def loadConfig(self, config_file):
        logger.debug('Load dataset config from file: {}'.format(config_file))
        with open(config_file, 'r') as f:
            data = json.load(f)
            logger.debug('Loaded dataset config: {}'.format(data))
            return Map(data)
        raise Exception('Could not load dataset config from file {}'.format(config_file))

    def getLabelFile(self):
        raise NotImplementedError('Method getLabelFile() must be implemented in subclass')

    def getTrainFile(self):
        raise NotImplementedError('Method getTrainFile() must be implemented in subclass')

    def getValFile(self):
        raise NotImplementedError('Method getValFile() must be implemented in subclass')

    def setArgs(self, args):
        self.args = args

    def setOutputFolder(self, output_folder):
        self.output_folder = output_folder

    def setIntermediateFormat(self, intermediate):
        self.intermediate = intermediate

    # def fromIntermediateFormat(self):
    #     raise NotImplementedError('Method fromIntermediateFormat() must be implemented in subclass')

    # def toIntermediateFormat(self):
    #     raise NotImplementedError('Method toIntermediateFormat() must be implemented in subclass')


# class DatasetFormat(ThreadExtension):

#     @staticmethod
#     def getExtension():
#         raise NotImplementedError('Method getExtension() needs to be implemented in subclasses')

#     @staticmethod
#     def getTrainingFilename():
#         raise NotImplementedError('Method getTrainingFilename() needs to be implemented in subclasses')

#     @staticmethod
#     def getValidateFilename():
#         raise NotImplementedError('Method getValidateFilename() needs to be implemented in subclasses')

#     @staticmethod
#     def getTrainingFilesNumber():
#         raise NotImplementedError('Method getTrainingFilesNumber() needs to be implemented in subclasses')

#     @staticmethod
#     def getValidateFilesNumber():
#         raise NotImplementedError('Method getValidateFilesNumber() needs to be implemented in subclasses')

#     def export(self):
#         raise NotImplementedError('Method export() needs to be implemented in subclasses')

    