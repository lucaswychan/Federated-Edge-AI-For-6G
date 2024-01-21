from algorithm import Algorithm

class Server(object):
    def __init__(self, algorithm):
        self.algorithm: Algorithm = algorithm
    
    def update(self, inputs: dict):
        self.algorithm.update(inputs)