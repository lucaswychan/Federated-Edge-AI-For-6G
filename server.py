class Server(object):
    def __init__(self, algorithm):
        self.algorithm = algorithm
    
    def aggregate(self, inputs: dict):
        self.algorithm.aggregate(inputs)