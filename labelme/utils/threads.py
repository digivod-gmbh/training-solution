import time
import traceback

from qtpy import QtCore
from qtpy.QtCore import Qt

from labelme.logger import logger


class AbortWorkerException(Exception):
    pass


class UserException(Exception):
    pass


class Abortable:

    def __init__(self):
        self.isAborted = False

    def checkAborted(self):
        if self.isAborted:
            raise AbortWorkerException()

    def abort(self):
        self.isAborted = True


class WorkerExecutor():

    def __init__(self):
        self.abortable = Abortable()
        self.wait_for_confirm = False
        self.confirm_value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import gc
        gc.collect()

    def run(self):
        raise NotImplementedError('Method run() must be implemented in subclass')

    def isAborted(self):
        return self.abortable.isAborted

    def abort(self):
        self.abortable.abort()

    def checkAborted(self):
        self.abortable.checkAborted()

    def throwUserException(self, msg):
        raise UserException(msg)

    def setAbortable(self, abortable):
        self.abortable = abortable

    def setThread(self, thread):
        self.thread = thread

    def doConfirm(self, title, message, kind):
        self.wait_for_confirm = True
        self.confirm_value = None
        self.thread.confirm.emit(title, message, kind)
        while self.wait_for_confirm:
            time.sleep(0.5)
        return self.confirm_value

    def confirmResult(self, result):
        self.wait_for_confirm = False
        self.confirm_value = result


class Worker(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.finished.connect(self.finish)

    def addObject(self, obj):
        self.started.connect(obj.start)

    def finish(self, args=None):
        logger.debug('Worker thread finished')
        self.deleteLater()

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

    handleError = QtCore.Signal(str, bool)
    finished = QtCore.Signal()
    update = QtCore.Signal(str, int, int)
    aborted = QtCore.Signal()
    data = QtCore.Signal(dict)
    message = QtCore.Signal(str, str, int)
    confirm = QtCore.Signal(str, str, int)

    def __init__(self, worker, start_func, error_func, abort_func=None, update_func=None, finish_func=None, data_func=None, message_func=None, confirm_func=None):
        super().__init__(worker)

        self.start_func = start_func
        self.error_func = error_func
        self.abort_func = abort_func
        self.update_func = update_func
        self.finish_func = finish_func
        self.data_func = data_func
        self.message_func = message_func
        self.confirm_func = confirm_func

        if self.update_func is not None:
            self.update.connect(self.update_func)
        if self.finish_func is not None:
            self.finished.connect(self.finish_func)
        if self.data_func is not None:
            self.data.connect(self.data_func)
        if self.message_func is not None:
            self.message.connect(self.message_func)
        if self.confirm_func is not None:
            self.confirm.connect(self.confirm_func)
        self.handleError.connect(self.error_func)
        self.aborted.connect(worker.abort)

    def start(self):
        try:
            self.start_func()
        except UserException as e:
            self.handleError.emit(str(e), True)
            self.error.emit(traceback.format_exc())
            return
        except AbortWorkerException:
            return
        except Exception as e:
            self.handleError.emit(str(e), False)
            self.error.emit(traceback.format_exc())
            return
        self.finish()

    def abort(self):
        try:
            if self.abort_func is not None:
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
