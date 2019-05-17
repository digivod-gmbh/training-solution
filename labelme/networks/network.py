import sys
from labelme.utils.threads import AbortWorkerException

class Network():

    def __init__(self):
        self.isAborted = False

    def training(self):
        raise NotImplementedError('Method training() needs to be implemented in subclasses')

    def inference(self):
        raise NotImplementedError('Method inference() needs to be implemented in subclasses')

    def getArgs(self):
        return self.args

    def getArchitectureFilename(self):
        return self.architecture_filename + '-symbol.json'

    def getWeightsFilename(self):
        return self.weights_filename + '-0000.params'
    
    def checkAborted(self):
        if self.isAborted:
            raise AbortWorkerException()

    def abort(self):
        self.isAborted = True
