import copy

import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt

from dataset import Dataset


def get_model_params(model_list, n_par=None):
    # count the number of parameters of a given model
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    # extract the parameters of a given model
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)  # param_mat =  [[ 0.09114207 -0.10681842  0.10701807 ...  0.07207876  0.00579278   -0.0345436 ]]

def set_client_from_params(model, params, device):
    dict_param = copy.deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    model.load_state_dict(dict_param)    
    return model

def get_acc_loss(data_x, data_y, model, dataset_name, device, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for _ in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_model_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst


def plot_performance(communication_rounds, tst_perf_all, algorithm_name, data_obj_name):
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(communication_rounds)+1, tst_perf_all[:,1], label=algorithm_name, linewidth=2.5, color='red')
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
    plt.grid()
    plt.xlim([0, communication_rounds+1])
    plt.title(data_obj_name, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('Output/%s/plot.pdf' %data_obj_name, dpi=1000, bbox_inches='tight')