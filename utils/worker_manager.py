import queue
from threading import Thread, Lock
from queue import Empty, Queue
import traceback
from abc import ABC

class MultithreadParams(ABC):
    def get_params(self):
        raise NotImplemented


class WorkerManager:
    '''
    Class that handles multithreading
    '''
    
    class WorkerMangerException(Exception):
        pass

    def __init__(self, target_func, worker_num=20) -> None:
        self._threads = []
        self._worker_num = worker_num
        self._target_function = target_func
        self._stop = True
        self._queue = Queue(worker_num * 5)
        self._lock = Lock()
    
    def start(self):
        '''
        Initialize worker manager
        '''
        with self._lock:
            if not self._stop:
                raise self.WorkerMangerException("Worker Manager has already started")

            self._stop = False
            for idx in range(self._worker_num):
                thread = Thread(target=self.assign_thread_to_task, name="%d thread" % (idx))
                thread.setDaemon(True)
                self._threads.append(thread)
                thread.start()
            
        return
    
    def put(self, record, idx):
        '''
        Add record to queue
        '''
        if self._stop:
            raise self.WorkerMangerException("Worker Manger has not been started")
        self._queue.put((record, idx))
    
    def stop(self):
        '''
        Stops the worker manager, joins remaining threads
        '''
        with self._lock:
            if self._stop:
                return
            self._stop = True
            
        for thread in self._threads:
            thread.join()
    

    def assign_thread_to_task(self):
        '''
        Pull from Queue and assign each thread to a task if workers are available
        '''
        def handle_stop():
            if self._stop:
                pass
                #print("queue is empty")
        while not self._stop:
            self._lock.acquire()
            while not self._queue.empty():
                self._lock.release()
                try:
                    self._lock.acquire()
                    record, idx = self._queue.get()
                    self._lock.release()
                    
                    self._target_function(record)
                except Empty:
                    print('empty queue at this time')
                    handle_stop()
                except Exception as e:
                    traceback.print_exc()
                    print("%s exception occured for idx: %s, record: %s" % (e, idx, record))
                self._lock.acquire()

            self._lock.release()
        handle_stop()
        


