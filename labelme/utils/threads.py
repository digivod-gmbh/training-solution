import traceback

from qtpy import QtCore
from labelme.logger import logger

class AbortWorkerException(Exception):
    pass


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
    error = QtCore.Signal(str)

    def __init__(self, worker):
        super().__init__()
        self.moveToThread(worker)
        self.success.connect(worker.success)
        self.error.connect(worker.error)


class ProgressObject(WorkerObject):

    handleError = QtCore.Signal(str)
    finished = QtCore.Signal()
    update = QtCore.Signal(str, int)
    aborted = QtCore.Signal()

    def __init__(self, worker, start_func, error_func, abort_func=None, update_func=None, finish_func=None):
        super().__init__(worker)

        self.start_func = start_func
        self.error_func = error_func
        self.abort_func = abort_func
        self.update_func = update_func
        self.finish_func = finish_func

        self.update.connect(self.update_func)
        self.finished.connect(self.finish_func)
        self.handleError.connect(self.error_func)
        self.aborted.connect(worker.abort)

    def start(self):
        try:
            self.start_func()
        except AbortWorkerException as e:
            return
        except Exception as e:
            self.handleError.emit(str(e))
            self.error.emit(traceback.format_exc())
            return
        self.finish()

    def abort(self):
        try:
            self.abort_func()
            self.aborted.emit()
        except Exception as e:
            self.error.emit(traceback.format_exc())

    def finish(self):
        try:
            self.finished.emit()
            self.success.emit()
        except Exception as e:
            self.error.emit(traceback.format_exc())
