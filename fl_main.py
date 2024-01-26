import copy
import os

import numpy as np
import torch

from algorithm import FedDyn
from client import Client
from dataset import DatasetObject
from model import Model
from server import Server
from utils import *
from optimize import *

n_clients = 10 # number of clients

# data for train and test
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_clients, rule='iid', unbalanced_sgm=0)
client_x_all = data_obj.clnt_x
client_y_all = data_obj.clnt_y
cent_x = np.concatenate(client_x_all, axis=0)
cent_y = np.concatenate(client_y_all, axis=0)
dataset      = data_obj.dataset

###

# the weight corresponds to the number of data that the client i has
# the more data the client has, the larger the weight is
weight_list = np.asarray([len(client_y_all[i]) for i in range(n_clients)])
weight_list = weight_list / np.sum(weight_list) * n_clients

# global parameters
model_name           = 'cifar10' # Model type
communication_rounds = 10
rand_seed            = 0
device               = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# algorithm parameters
act_prob             = 0.6
learning_rate        = 0.1
lr_decay_per_round   = 1
batch_size           = 50
epoch                = 5
weight_decay         = 1e-3
model_func           = lambda : Model(model_name)
init_model           = model_func()
save_period          = 1
print_per            = 5

init_par_list        = get_model_params([init_model])[0] # parameters of the initial model
n_param              = len(init_par_list)

# FedDyn parameters
alpha_coef           = 1e-2
max_norm             = 10

algorithm            = FedDyn(act_prob, learning_rate, lr_decay_per_round, batch_size, epoch, weight_decay, model_func, init_model, data_obj, n_param, save_period, print_per, alpha_coef, max_norm)

###
# Channel setup

alpha_direct = 3.76  # PL = path loss component
# User-BS Path loss exponent
fc = 915 * 10**6  # carrier frequency, wavelength lambda=3.0*10**8/fc
BS_Gain = 10 ** (5.0 / 10)  # BS antenna gain
RIS_Gain = 10 ** (5.0 / 10)  # RIS antenna gain
User_Gain = 10 ** (0.0 / 10)  # User antenna gain
d_RIS = 1.0 / 10  # dimension length of RIS element/wavelength
BS = np.array([-50, 0, 10])
RIS = np.array([0, 0, 10])

range = 20
x0 = np.ones([n_clients], dtype=int)
Jmax = 50

SCA_Gibbs = np.ones([Jmax + 1, communication_rounds]) * np.nan

###
# FL system components

clients_list = np.array([Client(algorithm=algorithm,
                                device=device, 
                                weight=weight_list[i], 
                                train_data_X=client_x_all[i], 
                                train_data_Y=client_y_all[i], 
                                model=init_model, 
                                client_param=np.copy(init_par_list)
                            ) for i in range(n_clients)])

server = Server(algorithm=algorithm)

###

local_param_list = np.zeros((n_clients, n_param)).astype('float32')

trn_perf_sel = np.zeros((communication_rounds, 2)); trn_perf_all = np.zeros((communication_rounds, 2))
tst_perf_sel = np.zeros((communication_rounds, 2)); tst_perf_all = np.zeros((communication_rounds, 2))

# average model
avg_model = model_func().to(device)
avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

# all clients model
all_model = model_func().to(device)
all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

# cloud (server) model
cloud_model = model_func().to(device)
cloud_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
cloud_model_param = get_model_params([cloud_model], n_param)[0]

###

if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))   

###

print('\nDevice: %s' %device)
print("Training starts with algorithm: %s\n" %algorithm.name)

for t in range(communication_rounds):

    # random client selection
    control_seed = 0
    selected_clnts = []
    while len(selected_clnts) == 0:
        # Fix randomness in client selection
        np.random.seed(t + rand_seed + control_seed)
        act_list    = np.random.uniform(size=n_clients)
        act_clients = act_list <= act_prob
        selected_clnts_idx = np.sort(np.where(act_clients)[0])
        selected_clnts = clients_list[selected_clnts_idx]
        control_seed += 1
    
    print('Selected Clients: %s' %(', '.join(['%2d' %clnt for clnt in selected_clnts_idx])))
    
    cloud_model_param_tensor = torch.tensor(cloud_model_param, dtype=torch.float32, device=device)
    
    ###
    # partial training
    for i, client in enumerate(selected_clnts):
        # Train locally 
        print('---- Training client %d' %selected_clnts_idx[i])
        
        feddyn_inputs = {
            "curr_round": t,
            "cloud_model": cloud_model,
            "cloud_model_param_tensor": cloud_model_param_tensor,
            "cloud_model_param": cloud_model_param,
            "local_param": local_param_list[selected_clnts_idx[i]]
        }
        
        client.local_train(feddyn_inputs)
        
        local_param_list[selected_clnts_idx[i]] = feddyn_inputs["local_param"]
    
    inputs = {
        "clients_list": clients_list,
        "selected_clnts_idx": selected_clnts_idx,
        "local_param_list": local_param_list,
        "avg_model": avg_model,
        "all_model": all_model,
        "cloud_model": cloud_model,
        "cloud_model_param": cloud_model_param
    }
    
    server.aggregate(inputs)
    
    avg_model = inputs["avg_model"]
    all_model = inputs["all_model"]
    cloud_model = inputs["cloud_model"]
    cloud_model_param = inputs["cloud_model_param"]

    # get the test accuracy
    algorithm.evaluate(data_obj, cent_x, cent_y, avg_model, all_model, device, tst_perf_sel, trn_perf_sel, tst_perf_all, trn_perf_all, t)

plot_performance(communication_rounds, tst_perf_sel, algorithm.name, data_obj.name)