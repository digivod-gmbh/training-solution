import json
import os

from labelme.logger import logger
from labelme.utils.map import Map
from labelme.extensions import ThreadExtension


class DatasetFormat(ThreadExtension):

    def importFolder(self):
        raise NotImplementedError('Method importFolder() must be implemented in subclass')

    def export(self):
        raise NotImplementedError('Method export() must be implemented in subclass')

    def isValidFormat(self, dataset_folder_or_file):
        raise NotImplementedError('Method isValidFormat() must be implemented in subclass')

    def getLabels(self):
        raise NotImplementedError('Method getLabels() must be implemented in subclass')

    def getNumSamples(self):
        raise NotImplementedError('Method getNumSamples() must be implemented in subclass')

    def getDatasetForTraining(self):
        raise NotImplementedError('Method getDatasetForTraining() must be implemented in subclass')

    def setArgs(self, args):
        self.args = args

    def setInputFolderOrFile(self, input_folder_or_file):
        self.input_folder_or_file = input_folder_or_file

    def setOutputFolder(self, output_folder):
        self.output_folder = output_folder

    def setIntermediateFormat(self, intermediate):
        self.intermediate = intermediate
