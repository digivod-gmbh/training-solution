from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from labelme.logger import logger
from labelme.utils import Application, Worker, ProgressObject
from labelme.config import MessageType
from labelme.extensions.thread import WorkerExecutor


class WorkerDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.progress = None
        self.current_worker_idx = None
        self.current_worker_object = None
        self.worker_executor = None
        self.finish_func = None
        self.data = {}
        self.custom_progress = None

    def run_thread(self, worker_executor, finish_func=None, custom_progress=None):
        if not isinstance(worker_executor, WorkerExecutor):
            raise Exception('Object {} must be of type WorkerExecutor'.format(worker_executor))

        if self.current_worker_object is not None or self.current_worker_idx is not None:
            raise Exception('An other thread is already running with index {}'.format(self.current_worker_idx))

        self.worker_executor = worker_executor
        self.finish_func = finish_func
        self.custom_progress = custom_progress

        self.current_worker_idx, worker = Application.createWorker(self)
        self.current_worker_object = ProgressObject(worker, worker_executor.run, self.on_error, worker_executor.abort, 
            self.on_progress, self.on_finish, self.on_data, self.on_message, self.on_confirm)
        worker_executor.setThread(self.current_worker_object)

        self.init_progress()

        worker.addObject(self.current_worker_object)
        worker.start()

        return self.current_worker_idx
    
    def reset_thread(self):
        Application.destroyWorker(self.current_worker_idx)
        self.current_worker_idx = None
        self.current_worker_object = None

    def on_abort(self):
        logger.debug('on_abort')
        self.progress.setLabelText(_('Cancelling ...'))
        self.progress.setMaximum(0)
        self.current_worker_object.abort()
        worker = Application.getWorker(self.current_worker_idx)
        worker.wait()
        self.cancel_progress()
        self.reset_thread()

    def on_error(self, e):
        logger.debug('on_error')
        self.cancel_progress()
        mb = QtWidgets.QMessageBox()
        mb.warning(self, _('Export'), _('An error occured during export of dataset'))

    def on_finish(self):
        logger.debug('on_finish')
        self.reset_thread()
        self.cancel_progress()
        if not self.worker_executor.isAborted():
            if self.finish_func is not None:
                self.finish_func()
            else:
                self.close()

    def on_message(self, title, message, kind=None):
        log_msg = 'Show message: {}'.format(message)
        mb = QtWidgets.QMessageBox
        if kind == MessageType.Warning:
            logger.warn(log_msg)
            mb.warning(self, title, message)
        elif kind == MessageType.Error:
            logger.error(log_msg)
            mb.critical(self, title, message)
        elif kind == MessageType.Question:
            logger.info(log_msg)
            mb.question(self, title, message)
        else:
            logger.info(log_msg)
            mb.information(self, title, message)

    def on_confirm(self, title, message, kind=None):
        clicked_btn = False
        mb = QtWidgets.QMessageBox
        if kind == MessageType.Warning:
            clicked_btn = mb.warning(self, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        elif kind == MessageType.Error:
            clicked_btn = mb.critical(self, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        elif kind == MessageType.Question:
            clicked_btn = mb.question(self, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        else:
            clicked_btn = mb.information(self, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        result = clicked_btn == QtWidgets.QMessageBox.Yes
        self.worker_executor.confirmResult(result)

    def on_progress(self, message=None, value=None, maximum=None):
        if isinstance(self.progress, QtWidgets.QProgressDialog):
            if self.progress.wasCanceled():
                return
        if message:
            self.progress.setLabelText(message)
        if value is not None:
            self.progress.setValue(value)
        if value == -1:
            val = self.progress.value() + 1
            self.progress.setValue(val)
        if maximum is not None and maximum > -1:
            self.progress.setMaximum(maximum)

    def cancel_progress(self):
        if isinstance(self.progress, QtWidgets.QProgressDialog):
            self.progress.cancel()

    def on_data(self, data):
        logger.debug('on_data')
        self.data = data

    def init_progress(self, maximum=100):
        if not self.custom_progress:
            self.progress = QtWidgets.QProgressDialog(_('Initializing ...'), _('Cancel'), 0, maximum, self)
            self.set_default_window_flags(self.progress)
            self.progress.setWindowModality(Qt.ApplicationModal)
            self.progress.show()
            self.progress.canceled.disconnect()
            self.progress.canceled.connect(self.on_abort)
        else:
            self.progress = self.custom_progress

    def set_default_window_flags(self, obj):
        obj.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)