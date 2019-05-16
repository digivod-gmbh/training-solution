import sys

class AbortException(Exception):
    pass

class Network():

    def __init__(self, thread):
        self.thread = thread
        self.isAborted = False

    def start(self):
        raise NotImplementedError('Method start() needs to be implemented in subclasses')
    
    def checkAborted(self):
        if self.isAborted:
            raise AbortException()

    def abort(self):
        self.isAborted = True
