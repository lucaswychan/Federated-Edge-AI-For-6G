class Server(object):
    def __init__(self, avg_model, all_model, device, algorithm):
        self.avg_model = avg_model
        self.all_model = all_model
        self.device = device
        self.algorithm = algorithm

    def aggregate(self, inputs):
        self.algorithm.aggregate(self, inputs)
        
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
