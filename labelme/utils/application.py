from labelme.utils import Worker

class Application():

    workerPool = {}

    def __init__(self):
        pass

    def createWorker(parent):
        worker = Worker(parent)
        idx = str(len(Application.workerPool))
        Application.workerPool[idx] = worker
        return idx, worker

    def getWorker(idx):
        if idx in Application.workerPool:
            return Application.workerPool[idx]
        return None

    def destroyWorker(idx):
        if idx in Application.workerPool:
            del Application.workerPool[idx]
            return True
        return False
        
