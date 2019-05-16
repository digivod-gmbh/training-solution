from labelme.utils import Worker

class Application():

    workerPool = {}
    #trainingWorker = None

    def __init__(self):
        pass

    def createWorker():
        worker = Worker()
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
        
        # for i, w in enumerate(Application.workerPool):
        #     if w is worker:
        #         print('Worker found: {}, {}'.format(i, w))
        #         Application.workerPool.remove(i)
        #         #del Application.workerPool[i]
        #         return True
        # return False
        
