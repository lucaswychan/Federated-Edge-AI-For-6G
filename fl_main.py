from client import Client
from server import Server
from dataset import DatasetObject
from algorithm import FedDyn
from model import Model
from utils import *
import numpy as np
import torch
import copy

n_clients = 10
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

model_name           = 'cifar10' # Model type
algorithm            = FedDyn()
algorithm_name       = algorithm.name
communication_rounds = 3
save_period          = 1
weight_decay         = 1e-3
batch_size           = 50
act_prob             = 0.4
lr_decay_per_round   = 1
epoch                = 5
learning_rate        = 0.1
print_per            = 5
rand_seed            = 0
alpha_coef           = 1e-2
device               = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: %s' %device)

model_func = lambda : Model(model_name)
init_model = model_func()
init_par_list = get_mdl_params([init_model])[0]
n_param = len(init_par_list)

###

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

n_save_instances = int(communication_rounds / save_period)
fed_mdls_sel = list(range(n_save_instances)) # Avg active clients
fed_mdls_all = list(range(n_save_instances)) # Avg all clients
fed_mdls_cld = list(range(n_save_instances)) # Cloud models 

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
cloud_model_param = get_mdl_params([cloud_model], n_param)[0]

print("Training starts with algorithm: %s" %algorithm_name)

for t in range(communication_rounds):

    # random client selection
    inc_seed = 0
    while(True):
        # Fix randomness in client selection
        np.random.seed(t + rand_seed + inc_seed)
        act_list    = np.random.uniform(size=n_clients)
        act_clients = act_list <= act_prob
        selected_clnts_idx = np.sort(np.where(act_clients)[0])
        selected_clnts = clients_list[selected_clnts_idx]
        inc_seed += 1
        if len(selected_clnts) != 0:
            break
    print('Selected Clients: %s' %(', '.join(['%2d' %clnt for clnt in selected_clnts_idx])))
    
    cloud_model_param_tensor = torch.tensor(cloud_model_param, dtype=torch.float32, device=device)
    
    ###
    feddyn_inputs = {}
    # partial training
    for i, client in enumerate(selected_clnts):
        # Train locally 
        print('---- Training client %d' %selected_clnts_idx[i])
        
        feddyn_inputs = {
            "data_obj": data_obj, 
            "act_prob": act_prob, 
            "learning_rate": learning_rate, 
            "batch_size": batch_size, 
            "epoch": epoch, 
            "communication_rounds": communication_rounds, 
            "curr_round": t,
            "print_per": print_per, 
            "weight_decay": weight_decay,  
            "model_func": model_func, 
            "init_model": init_model, 
            "alpha_coef": alpha_coef, 
            "save_period": save_period, 
            "lr_decay_per_round": lr_decay_per_round, 
            "n_param": n_param,
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
        "model_func": model_func, 
        "avg_model": avg_model,
        "all_model": all_model,
        "cloud_model": cloud_model,
        "cloud_model_param": cloud_model_param
    }
    
    server.update(inputs)
    
    avg_model = inputs["avg_model"]
    all_model = inputs["all_model"]
    cloud_model = inputs["cloud_model"]
    cloud_model_param = inputs["cloud_model_param"]

    ###
    loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, device)
    tst_perf_sel[t] = [loss_tst, acc_tst]
    print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
    ###
    loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, device)
    trn_perf_sel[t] = [loss_tst, acc_tst]
    print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
    ###
    loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset, device)
    tst_perf_all[t] = [loss_tst, acc_tst]
    print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
    ###
    loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, device)
    trn_perf_all[t] = [loss_tst, acc_tst]
    print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(t+1, acc_tst, loss_tst))
    
    if ((t+1) % save_period == 0):
        fed_mdls_sel[t//save_period] = avg_model
        fed_mdls_all[t//save_period] = all_model
        fed_mdls_cld[t//save_period] = cloud_model
    # 3. Each client sends local model to server
    # 4. Server aggregates all local models into global model
    # 5. Repeat