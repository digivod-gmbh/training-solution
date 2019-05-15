from qtpy import QtCore
from labelme.logger import logger

class Worker(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def addObject(self, obj):
        self.started.connect(obj.start)

    def success(self, args=None):
        logger.debug('Worker thread finished successfully: {}'.format(args))
        super().quit()
    
    def abort(self, args=None):
        logger.debug('Worker thread was aborted: {}'.format(str(args)))
        super().quit()

    def error(self, args=None):
        logger.error('Error occured in worker thread: {}'.format(str(args)))
        super().quit()

class WorkerObject(QtCore.QObject):

    success = QtCore.Signal()
    abort = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, worker):
        super().__init__()
        self.moveToThread(worker)
        self.success.connect(worker.success)
        self.abort.connect(worker.abort)
        self.error.connect(worker.error)

    def start(self):
        print('OBJECT STARTED')
        self.success.emit()


class TrainingObject(WorkerObject):

    update = QtCore.Signal(str, int)

    def __init__(self, worker, start_func, update_func):
        super().__init__(worker)
        self.start = start_func
        self.update.connect(update_func)

    # def start(self):
    #     print('CHILD')
    #     self.update.emit('PROGRESS', 42)


# class TrainingProgressSignal():
#     def __init__():
#         pass

# class Communicate(QtCore.QObject):
#     thread_signal = QtCore.Signal(str) # TODO: Use TrainingProgressSignal
