from abc import abstractmethod


class Algorithm:
    def __init__(
        self,
        name,
        lr,
        lr_decay_per_round,
        batch_size,
        epoch,
        weight_decay,
        model_func,
        n_param,
        max_norm,
        noiseless,
        dataset_name,
        save_period,
        print_per,
    ):
        self.name = name
        self.lr = lr
        self.lr_decay_per_round = lr_decay_per_round
        self.batch_size = batch_size
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.model_func = model_func
        self.n_param = n_param
        self.max_norm = max_norm
        self.noiseless = noiseless
        self.dataset_name = dataset_name
        self.save_period = save_period
        self.print_per = print_per

    @abstractmethod
    def local_train(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass
