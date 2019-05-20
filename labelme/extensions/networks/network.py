import mxnet as mx

from labelme.logger import logger
from labelme.extensions import ThreadExtension

class Network(ThreadExtension):

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

    def get_context(self, gpus=None):
        if gpus is None:
            return [mx.cpu()]
        ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
        try:
            tmp = mx.nd.array([1, 2, 3], ctx=ctx[0])
        except mx.MXNetError as e:
            ctx = [mx.cpu()]
            logger.error(e)
            logger.warning('Unable to use GPU. Using CPU instead')
        logger.debug('Use context: {}'.format(ctx))
        return ctx

    def read_classes(self, classes_list):
        with open(classes_list) as f:
            return f.read().split('\n')
    