from qtpy import QtCore
from labelme.logger import logger

class Worker(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.finished.connect(self.finish)

    def addObject(self, obj):
        self.started.connect(obj.start)

    def finish(self, args=None):
        logger.debug('Worker thread finished')

    def success(self, args=None):
        logger.debug('Worker thread requested success ...')
        self.quit()
    
    def abort(self, args=None):
        logger.debug('Worker thread requested abort ...')
        self.quit()

    def error(self, args=None):
        logger.error('Error occured in worker thread: {}'.format(str(args)))
        self.quit()


class WorkerObject(QtCore.QObject):

    success = QtCore.Signal()
    aborted = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, worker):
        super().__init__()
        self.start_func = None
        self.abort_func = None
        self.moveToThread(worker)
        self.success.connect(worker.success)
        self.aborted.connect(worker.abort)
        self.error.connect(worker.error)

    def start(self):
        try:
            self.start_func()
        except Exception as e:
            self.error.emit(str(e))
        self.success.emit()

    def abort(self):
        try:
            self.abort_func()
            self.aborted.emit()
        except Exception as e:
            self.error.emit(str(e))

    def setAbortFunc(self, abort_func):
        self.abort_func = abort_func


class TrainingObject(WorkerObject):

    update = QtCore.Signal(str, int)

    def __init__(self, worker, start_func, update_func):
        super().__init__(worker)
        self.start_func = start_func
        self.update_func = update_func
        self.update.connect(update_func)

