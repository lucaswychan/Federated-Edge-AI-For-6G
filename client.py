class Client(object):
    def __init__(
        self, algorithm, device, weight, train_data_X, train_data_Y, model, client_param
    ):
        self.algorithm = algorithm
        self.device = device
        self.weight = weight
        self.train_data_X = train_data_X
        self.train_data_Y = train_data_Y
        self.model = model
        self.client_param = client_param

    def local_train(self, inputs: dict):
        self.algorithm.local_train(self, inputs)

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
