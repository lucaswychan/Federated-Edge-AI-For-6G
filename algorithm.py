from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def local_train(self):
        pass
    
    @abstractmethod
    def local_eval(self):
        pass

class FedDyn(Algorithm):
    def __init__(self):
        pass
    
    def local_train(self, params: dict):
        pass
    
    def local_eval(, params: dict):
        pass 