from labelme.utils.threads import AbortWorkerException

class ThreadExtension():

    def __init__(self):
        self.isAborted = False

    def checkAborted(self):
        if self.isAborted:
            raise AbortWorkerException()

    def abort(self):
        self.isAborted = True
