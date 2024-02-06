import copy

import torch
from torch.utils import data

from algorithm.algorithm_base import Algorithm
from client import Client
from dataset import Dataset
from utils import *


class SCAFFOLD(Algorithm):
    def __init__(self, act_prob, lr, lr_decay_per_round, batch_size, epoch, weight_decay, model_func, init_model, data_obj, n_param, max_norm, air_comp, noiseless, save_period, print_per):
        super().__init__("SCAFFOLD", act_prob, lr, lr_decay_per_round, batch_size, epoch, weight_decay, model_func, init_model, data_obj, n_param, max_norm, air_comp, noiseless, save_period, print_per)
        
    # override
    def local_train(self, client: Client, inputs: dict):
        pass
    
    def _train_model(self, model, trn_x, trn_y, curr_round):
        pass
    
    # override
    def aggregate(self, inputs: dict):
        pass