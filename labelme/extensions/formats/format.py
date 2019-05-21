from labelme.extensions import ThreadExtension

class DatasetFormat(ThreadExtension):

    @staticmethod
    def getExtension():
        raise NotImplementedError('Method getExtension() needs to be implemented in subclasses')

    @staticmethod
    def getTrainingFilename():
        raise NotImplementedError('Method getTrainingFilename() needs to be implemented in subclasses')

    @staticmethod
    def getValidateFilename():
        raise NotImplementedError('Method getValidateFilename() needs to be implemented in subclasses')

    @staticmethod
    def getTrainingFilesNumber():
        raise NotImplementedError('Method getTrainingFilesNumber() needs to be implemented in subclasses')

    @staticmethod
    def getValidateFilesNumber():
        raise NotImplementedError('Method getValidateFilesNumber() needs to be implemented in subclasses')

    def export(self):
        raise NotImplementedError('Method export() needs to be implemented in subclasses')

    