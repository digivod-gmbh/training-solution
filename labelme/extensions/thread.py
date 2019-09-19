import time


class AbortWorkerException(Exception):
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

    def run(self):
        raise NotImplementedError('Method run() must be implemented in subclass')

    def isAborted(self):
        return self.abortable.isAborted

    def abort(self):
        self.abortable.abort()

    def checkAborted(self):
        self.abortable.checkAborted()

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
