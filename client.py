from algorithm import Algorithm
import torch

class Client(object):
    def __init__(self, algorithm):
        self.algorithm: Algorithm = algorithm
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def local_train(self):
        params = {}
        self.algorithm.local_train(params)