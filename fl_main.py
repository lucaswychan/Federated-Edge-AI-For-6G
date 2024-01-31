import copy
import os
import time

import numpy as np
import torch

from air_comp import AirComp
from algorithm import FedDyn, FedAvg, FedProx
from args import args_parser
from client import Client
from dataset import DatasetObject
from model import Model
from optimize import *
from server import Server
from utils import *

args = args_parser()

# data for train and test
storage_path = 'LEAF/shakespeare/data/'
# data_obj =  DatasetObject(dataset='CIFAR10', n_client=args.n_clients, unbalanced_sgm=0, rule='Dirichlet', rule_arg=0.6)
data_obj = DatasetObject(dataset='CIFAR10', n_client=args.n_clients, rule='iid', unbalanced_sgm=0)
client_x_all = data_obj.clnt_x
client_y_all = data_obj.clnt_y
cent_x = np.concatenate(client_x_all, axis=0)
cent_y = np.concatenate(client_y_all, axis=0)
dataset      = data_obj.dataset

###

# the weight corresponds to the number of data that the client i has
# the more data the client has, the larger the weight is
weight_list = np.asarray([len(client_y_all[i]) for i in range(args.n_clients)])
weight_list = weight_list / np.sum(weight_list) * args.n_clients

# global parameters
device               = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# algorithm parameters
model_func           = lambda : Model(args.model_name)
init_model           = model_func()

init_par_list        = get_model_params([init_model])[0] # parameters of the iargs.nitial model
n_param              = len(init_par_list)

# FedDyn parameters
alpha_coef           = 1e-2
max_norm             = 10

###
# Channel setup

# User-BS Path loss exponent
fc                   = 915 * 10**6  # carrier frequency, wavelength lambda=3.0*10**8/fc
BS_Gain              = 10 ** (5.0 / 10)  # BS antenna gain
RIS_Gain             = 10 ** (5.0 / 10)  # RIS antenna gain
User_Gain            = 10 ** (0.0 / 10)  # User antenna gain
dimen_RIS            = 1.0 / 10  # dimension length of RIS element/wavelength
BS                   = np.array([-50, 0, 10])  # Cartesian coordinate of BS
RIS                  = np.array([0, 0, 10])    # Cartesian coordinate of RIS
#
x0                   = np.ones([args.n_clients], dtype=int)

# parameters passed to Gibbs
K                    = np.asarray([len(client_y_all[i]) for i in range(args.n_clients)])
#
gibbs                = Gibbs(n_clients=args.n_clients, n_receive_antennas=args.n_receive_antennas, n_RIS_ele=args.n_RIS_ele, Jmax=args.Jmax, K=weight_list, RISON=args.RISON, tau=args.tau, nit=args.nit, threshold=args.threshold)

#
air_comp             = AirComp(n_receive_antennas=args.n_receive_antennas, K=weight_list, transmit_power=args.transmit_power)

np.random.seed(args.rand_seed)

###
# FL system components

algorithm            = FedDyn(args.act_prob, args.learning_rate, args.lr_decay_per_round, args.batch_size, args.epoch, args.weight_decay, model_func, init_model, data_obj, n_param, air_comp, args.save_period, args.print_per, alpha_coef, max_norm)


clients_list = np.array([Client(algorithm=algorithm,
                                device=device, 
                                weight=weight_list[i], 
                                train_data_X=client_x_all[i], 
                                train_data_Y=client_y_all[i], 
                                model=init_model, 
                                client_param=np.copy(init_par_list)
                            ) for i in range(args.n_clients)])

server = Server(algorithm=algorithm)

###

local_param_list = np.zeros((args.n_clients, n_param)).astype('float32')

trn_perf_sel = np.zeros((args.communication_rounds, 2)); trn_perf_all = np.zeros((args.communication_rounds, 2))
tst_perf_sel = np.zeros((args.communication_rounds, 2)); tst_perf_all = np.zeros((args.communication_rounds, 2))

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

if not os.path.exists('Output/%s/%s_iargs.nit_mdl.pt' %(data_obj.name, args.model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_iargs.nit_mdl.pt' %(data_obj.name, args.model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_iargs.nit_mdl.pt' %(data_obj.name, args.model_name)))   

###
# Start Training

print('\nDevice: %s' %device)
print("Training starts with algorithm: %s\n" %algorithm.name)
print("The system is {}".format("args.noiseless" if args.noiseless else "noisy"))

for t in range(args.communication_rounds):
    
    print("This is the {0}-th trial".format(t+1))

    ref = (1e-10) ** 0.5
    sigma_n = np.power(10, -args.SNR / 10)
    sigma = sigma_n / ref**2 # [100,100+range]

    # set 2
    dx2 = (
        np.random.rand(int(args.n_clients - np.round(args.n_clients / 2))) * args.location_range
        + 200
    )

    dx1 = (
        np.random.rand(int(np.round(args.n_clients / 2))) * args.location_range - args.location_range
    )  # [-location_range , 0]
    
    dx = np.concatenate((dx1, dx2))
    np.random.shuffle(dx)
    
    dy = np.random.rand(args.n_clients) * 20 - 10
    d_UR = (
        (dx - RIS[0]) ** 2
        + (dy - RIS[1]) ** 2
        + RIS[2] ** 2
    ) ** 0.5
    d_RB = np.linalg.norm(BS - RIS)
    d_RIS_total = d_UR + d_RB
    d_direct = (
        (dx - BS[0]) ** 2
        + (dy - BS[1]) ** 2
        + BS[2] ** 2
    ) ** 0.5
    PL_direct = (
        BS_Gain
        * User_Gain
        * (3 * 10**8 / fc / 4 / np.pi / d_direct) ** args.alpha_direct
    )
    PL_RIS = (
        BS_Gain
        * User_Gain
        * RIS_Gain
        * args.n_RIS_ele**2
        * dimen_RIS**2
        / 4
        / np.pi
        * (3 * 10**8 / fc / 4 / np.pi / d_UR) ** 2
        * (3 * 10**8 / fc / 4 / np.pi / d_RB) ** 2
    )
    # channels
    h_d = (
        np.random.randn(args.n_receive_antennas, args.n_clients)
        + 1j * np.random.randn(args.n_receive_antennas, args.n_clients)
    ) / 2**0.5
    h_d = h_d @ np.diag(PL_direct**0.5) / ref
    H_RB = (
        np.random.randn(args.n_receive_antennas, args.n_RIS_ele)
        + 1j * np.random.randn(args.n_receive_antennas, args.n_RIS_ele)
    ) / 2**0.5
    h_UR = (
        np.random.randn(args.n_RIS_ele, args.n_clients)
        + 1j * np.random.randn(args.n_RIS_ele, args.n_clients)
    ) / 2**0.5
    h_UR = h_UR @ np.diag(PL_RIS**0.5) / ref

    G = np.zeros([args.n_receive_antennas, args.n_RIS_ele, args.n_clients], dtype=complex)
    for j in range(args.n_clients):
        G[:, :, j] = H_RB @ np.diag(h_UR[:, j])
    x = x0

    start = time.time()
    print("\niargs.nitial x = ", x)
    print()
    [x_store, obj_new, f_store, theta_store] = gibbs.optimize(h_d, G, x, sigma)
    print("final x = ", x_store[args.Jmax])
    print("final obj = ", obj_new)
    end = time.time()
    print("Running time: {} seconds\n".format(end - start))
    
    theta_optim = theta_store[:, args.Jmax]

    h_optim = np.zeros([args.n_receive_antennas, args.n_clients], dtype=complex)
    for i in range(args.n_clients):
        h_optim[:, i] = h_d[:, i] + G[:, :, i] @ theta_optim

    selected_optim = x_store[args.Jmax]
    selected_clnts_idx = np.where(selected_optim == 1)[0] # get the index of the selected clients
    selected_clnts = clients_list[selected_clnts_idx]
    
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
        "cloud_model_param": cloud_model_param,
        "args.noiseless": args.noiseless
    }
    if not args.noiseless:
        inputs["x"] = selected_optim
        inputs["f"] = f_store[:, args.Jmax]
        inputs["h"] = h_optim
        inputs["sigma"] = sigma
        
    server.aggregate(inputs)
    
    avg_model = inputs["avg_model"]
    all_model = inputs["all_model"]
    cloud_model = inputs["cloud_model"]
    cloud_model_param = inputs["cloud_model_param"]

    # get the test accuracy
    algorithm.evaluate(data_obj, cent_x, cent_y, avg_model, all_model, device, tst_perf_sel, trn_perf_sel, tst_perf_all, trn_perf_all, t)

save_performance(args.communication_rounds, tst_perf_all, algorithm.name, data_obj.name, args.n_clients, args.noiseless)