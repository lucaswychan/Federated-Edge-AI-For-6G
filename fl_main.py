import copy
import os
import time

import numpy as np
import torch

from air_comp import AirComp
from algorithm.fedavg import FedAvg
from algorithm.feddyn import FedDyn
from args import args_parser
from channel import Channel
from client import Client
from dataset import DatasetObject
from model import Model
from optimize import *
from utils import *


def main():
    # get all the constant parameters
    args = args_parser()

    # data for train and test
    storage_path = 'LEAF/shakespeare/data/'
    # data_obj =  DatasetObject(dataset='CIFAR10', n_client=args.n_clients, unbalanced_sgm=0, rule='Dirichlet', rule_arg=0.6)
    data_obj = DatasetObject(dataset='CIFAR10', n_client=args.n_clients, rule='iid', unbalanced_sgm=0)
    client_x_all = data_obj.clnt_x
    client_y_all = data_obj.clnt_y
    cent_x = np.concatenate(client_x_all, axis=0)
    cent_y = np.concatenate(client_y_all, axis=0)
    dataset = data_obj.dataset

    ###

    # the weight corresponds to the number of data that the client i has
    # the more data the client has, the larger the weight is
    weight_list   = np.asarray([len(client_y_all[i]) for i in range(args.n_clients)])
    weight_list   = weight_list / np.sum(weight_list) * args.n_clients  # FedDyn initialization

    # global parameters
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # algorithm parameters
    model_func    = lambda : Model(args.model_name)
    init_model    = model_func()

    init_par_list = get_model_params([init_model])[0] # parameters of the iargs.nitial model
    n_param       = len(init_par_list)

    # FedDyn parameters
    alpha_coef    = 1e-2
    
    #
    np.random.seed(args.rand_seed)

    ###
    # Channel setup

    # User-BS Path loss exponent
    fc            = 915 * 10**6  # carrier frequency, wavelength lambda=3.0*10**8/fc
    BS_Gain       = 10 ** (5.0 / 10)  # BS antenna gain
    RIS_Gain      = 10 ** (5.0 / 10)  # RIS antenna gain
    User_Gain     = 10 ** (0.0 / 10)  # User antenna gain
    dimen_RIS     = 1.0 / 10  # dimension length of RIS element/wavelength
    BS            = np.array([-50, 0, 10])  # Cartesian coordinate of BS
    RIS           = np.array([0, 0, 10])    # Cartesian coordinate of RIS
    #
    x0            = np.ones([args.n_clients], dtype=int)
    #
    channel       = Channel(SNR=args.SNR, n_clients=args.n_clients, location_range=args.location_range, fc=fc, alpha_direct=args.alpha_direct, n_RIS_ele=args.n_RIS_ele, n_receive_ant=args.n_receive_ant, User_Gain=User_Gain, x0=x0, BS=BS, BS_Gain=BS_Gain, RIS=RIS, RIS_Gain=RIS_Gain, dimen_RIS=dimen_RIS)

    gibbs         = Gibbs(n_clients=args.n_clients, n_receive_ant=args.n_receive_ant, n_RIS_ele=args.n_RIS_ele, Jmax=args.Jmax, K=weight_list, RISON=args.RISON, tau=args.tau, nit=args.nit, threshold=args.threshold)

    air_comp      = AirComp(n_receive_ant=args.n_receive_ant, K=weight_list, transmit_power=args.transmit_power)

    ###
    # FL system components

    algorithm = FedDyn(args.act_prob, args.lr, args.lr_decay_per_round, args.batch_size, args.epoch, args.weight_decay, model_func, init_model, data_obj, n_param, args.max_norm, air_comp, args.noiseless, args.save_period, args.print_per, alpha_coef)

    # algorithm = FedAvg(args.act_prob, args.lr, args.lr_decay_per_round, args.batch_size, args.epoch, args.weight_decay, model_func, init_model, data_obj, n_param, args.max_norm, air_comp, args.noiseless, args.save_period, args.print_per)

    clients_list = np.array([Client(algorithm=algorithm,
                                    device=device, 
                                    weight=weight_list[i], 
                                    train_data_X=client_x_all[i], 
                                    train_data_Y=client_y_all[i], 
                                    model=init_model, 
                                    client_param=np.copy(init_par_list)
                                ) for i in range(args.n_clients)])
    ###

    if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, args.model_name)):
        print("New directory!")
        os.mkdir('Output/%s/' %(data_obj.name))
        torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, args.model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, args.model_name)))   

    ###
    # Model Initialization (feel free to add new parameters in this section if using this frameworks)

    local_param_list = np.zeros((args.n_clients, n_param)).astype('float32')  # for feddyn

    trn_perf_sel = np.zeros((args.comm_rounds, 2)); trn_perf_all = np.zeros((args.comm_rounds, 2))
    tst_perf_sel = np.zeros((args.comm_rounds, 2)); tst_perf_all = np.zeros((args.comm_rounds, 2))

    # average model
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # all clients model
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # cloud (server) model  (for feddyn)
    cloud_model = model_func().to(device)
    cloud_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cloud_model_param = get_model_params([cloud_model], n_param)[0]

    ###
    # Start Training
    print()
    print("=" * 80)
    print('Device: %s' %device)
    print("Model: %s" %args.model_name)
    print("Epochs: %d" %args.epoch)
    print("Number of clients: %d" %args.n_clients)
    print("Number of communication rounds: %d" %args.comm_rounds)
    print("Training starts with algorithm: %s\n" %algorithm.name)
    print("The system is {}".format("noiseless" if args.noiseless else "noisy"))
    print("=" * 80, end="\n\n")

    for t in range(args.comm_rounds):
        
        print("This is the {0}-th trial".format(t+1))

        h_d, G, x, sigma = channel.generate()

        start = time.time()
        x_store, obj_new, f_store, theta_store = gibbs.optimize(h_d, G, x, sigma)
        end = time.time()
        print("Running time of Gibbs Optimization: {} seconds\n".format(end - start))
        
        theta_optim = theta_store[:, args.Jmax]

        h_optim = np.zeros([args.n_receive_ant, args.n_clients], dtype=complex)
        for i in range(args.n_clients):
            h_optim[:, i] = h_d[:, i] + G[:, :, i] @ theta_optim

        selected_optim = x_store[args.Jmax]
        selected_clnts_idx = np.where(selected_optim == 1)[0] # get the index of the selected clients
        selected_clnts = clients_list[selected_clnts_idx]
        
        print("Selected Clients Index: {}".format(x_store[args.Jmax]))
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
            
            fedavg_inputs = {
                "curr_round": t,
                "avg_model": avg_model,
            }
            
            client.local_train(feddyn_inputs)
            
            # update the parameters (For FedDyn)
            local_param_list[selected_clnts_idx[i]] = feddyn_inputs["local_param"]
        
        feddyn_inputs = {
            "clients_list": clients_list,
            "selected_clnts_idx": selected_clnts_idx,
            "local_param_list": local_param_list,
            "avg_model": avg_model,
            "all_model": all_model,
            "cloud_model": cloud_model,
            "cloud_model_param": cloud_model_param,
        }
        
        fedavg_inputs = {
            "clients_list": clients_list,
            "selected_clnts_idx": selected_clnts_idx,
            "weight_list": weight_list,
            "avg_model": avg_model,
            "all_model": all_model,
        }
        # pass the AirComp optimization parameters
        if not args.noiseless:
            feddyn_inputs["x"] = selected_optim
            feddyn_inputs["f"] = f_store[:, args.Jmax]
            feddyn_inputs["h"] = h_optim
            feddyn_inputs["sigma"] = sigma
            
            # fedavg_inputs["x"] = selected_optim
            # fedavg_inputs["f"] = f_store[:, args.Jmax]
            # fedavg_inputs["h"] = h_optim
            # fedavg_inputs["sigma"] = sigma
            
        algorithm.aggregate(feddyn_inputs)
        
        # update the parameters after aggregation
        avg_model = feddyn_inputs["avg_model"]
        all_model = feddyn_inputs["all_model"]
        
        # For FedDyn
        cloud_model = feddyn_inputs["cloud_model"]
        cloud_model_param = feddyn_inputs["cloud_model_param"]

        # get the test accuracy
        algorithm.evaluate(data_obj, cent_x, cent_y, avg_model, all_model, device, tst_perf_sel, trn_perf_sel, tst_perf_all, trn_perf_all, t)

    save_performance(args.comm_rounds, tst_perf_all, algorithm.name, data_obj.name, args.n_clients, args.noiseless)
    
    
if __name__ == "__main__":
    main()