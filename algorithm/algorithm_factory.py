import numpy as np

from algorithm.algorithm_base import Algorithm
from algorithm.fedavg import FedAvg
from algorithm.feddyn import FedDyn
from algorithm.fedprox import FedProx
from algorithm.scaffold import SCAFFOLD


class AlgorithmFactory:
    def __init__(self, args):
        self.args = args

    def create_algorithm(self, algorithm_name) -> Algorithm:
        algorithm = None

        if algorithm_name == "FedDyn":
            alpha_coef = 1e-2

            algorithm = FedDyn(
                self.args.lr,
                self.args.lr_decay_per_round,
                self.args.batch_size,
                self.args.epoch,
                self.args.weight_decay,
                self.args.model_func,
                self.args.n_param,
                self.args.max_norm,
                self.args.noiseless,
                self.args.data_obj.dataset,
                self.args.save_period,
                self.args.print_per,
                alpha_coef,
            )

        elif algorithm_name == "FedProx":
            mu = 1e-4

            algorithm = FedProx(
                self.args.lr,
                self.args.lr_decay_per_round,
                self.args.batch_size,
                self.args.epoch,
                self.args.weight_decay,
                self.args.model_func,
                self.args.n_param,
                self.args.max_norm,
                self.args.noiseless,
                self.args.data_obj.dataset,
                self.args.save_period,
                self.args.print_per,
                mu,
            )

        elif algorithm_name == "SCAFFOLD":
            n_data_per_client = (
                np.concatenate(self.args.data_obj.clnt_x, axis=0).shape[0]
                / self.args.n_clients
            )
            n_iter_per_epoch = np.ceil(n_data_per_client / self.args.batch_size)
            n_minibatch = (self.args.epoch * n_iter_per_epoch).astype(np.int64)
            self.args.print_per = self.args.print_per * n_iter_per_epoch
            global_learning_rate = 1

            algorithm = SCAFFOLD(
                self.args.lr,
                self.args.lr_decay_per_round,
                self.args.batch_size,
                self.args.epoch,
                self.args.weight_decay,
                self.args.model_func,
                self.args.n_param,
                self.args.max_norm,
                self.args.noiseless,
                self.args.data_obj.dataset,
                self.args.save_period,
                self.args.print_per,
                n_minibatch,
                global_learning_rate,
            )

        elif algorithm_name == "FedAvg":
            algorithm = FedAvg(
                self.args.lr,
                self.args.lr_decay_per_round,
                self.args.batch_size,
                self.args.epoch,
                self.args.weight_decay,
                self.args.model_func,
                self.args.n_param,
                self.args.max_norm,
                self.args.noiseless,
                self.args.data_obj.dataset,
                self.args.save_period,
                self.args.print_per,
            )

        else:
            raise ValueError(f"Unknown algorithm name: {self.algorithm_name}")

        return algorithm
