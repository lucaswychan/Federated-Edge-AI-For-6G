from abc import abstractmethod
from model import Model
from utils import *
from dataset import Dataset
import copy
import torch
from torch.utils import data

class Algorithm:
    def __init__(self):
        self.name = "Algorithm"
        
    @abstractmethod
    def local_train(self):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def local_eval(self):
        pass

###

class FedDyn(Algorithm):
    def __init__(self):
        self.name = "FedDyn"
        self.max_norm = 10
    
    def local_train(self, client, inputs: dict):
        model = client.model
        trn_x = client.train_data_X
        trn_y = client.train_data_Y
        self.device = client.device

        client.model = inputs["model_func"]().to(self.device)
        model = client.model # = self.model
        # Warm start from current avg model
        model.load_state_dict(copy.deepcopy(dict(inputs["cloud_model"].named_parameters())))
        for params in model.parameters():
            params.requires_grad = True

        # Scale down
        alpha_coef_adpt = inputs["alpha_coef"] / client.weight # adaptive alpha coef
        local_param_list_curr = torch.tensor(inputs["local_param"], dtype=torch.float32, device=self.device) # = local_grad_vector
        print("local_param_list_curr = ", local_param_list_curr)
        print("cloud_model_param_tensor = ", inputs["cloud_model_param_tensor"])
        client.model = self.train_model(model, inputs["model_func"], alpha_coef_adpt, inputs["cloud_model_param_tensor"], local_param_list_curr, trn_x, trn_y, inputs["learning_rate"] * (inputs["lr_decay_per_round"] ** inputs["curr_round"]), inputs["batch_size"], inputs["epoch"], inputs["print_per"], inputs["weight_decay"], inputs["data_obj"].dataset)
        curr_model_par = get_mdl_params([client.model], inputs["n_param"])[0] # get the model parameter after running FedDyn
        print("curr_model_par = ", curr_model_par)

        # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
        inputs["local_param"] += curr_model_par - inputs["cloud_model_param"]  # after training, dynamically update the weight withthe cloud model parameters
        client.client_param = curr_model_par
    
    
    def train_model(self, model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
        n_trn = trn_x.shape[0]
        trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
        model.train(); model = model.to(self.device)
        
        n_par = get_mdl_params([model_func()]).shape[1]
        
        for e in range(epoch):
            # Training
            epoch_loss = 0
            trn_gen_iter = trn_gen.__iter__()
            for i in range(int(np.ceil(n_trn/batch_size))):
                batch_x, batch_y = trn_gen_iter.__next__()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                y_pred = model(batch_x)
                
                ## Get f_i estimate 
                loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
                loss_f_i = loss_f_i / list(batch_y.size())[0]
                
                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                
                loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
                loss = loss_f_i + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_norm) # Clip gradients
                optimizer.step()
                epoch_loss += loss.item() * list(batch_y.size())[0]

            if (e+1) % print_per == 0:
                epoch_loss /= n_trn
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
                print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
                model.train()
        
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
                
        return model

    def update(self, inputs: dict):
        clients_list = inputs["clients_list"]
        all_clients_param_list = np.array([client.client_param for client in clients_list])
        
        avg_mdl_param = np.mean(all_clients_param_list[inputs["selected_clnts_idx"]], axis = 0)
        inputs["cloud_model_param"] = avg_mdl_param + np.mean(inputs["local_param_list"], axis=0)
        
        device = clients_list[0].device

        inputs["avg_model"] = set_client_from_params(inputs["model_func"](), avg_mdl_param, device)
        inputs["all_model"] = set_client_from_params(inputs["model_func"](), np.mean(all_clients_param_list, axis = 0), device)
        inputs["cloud_model"] = set_client_from_params(inputs["model_func"]().to(device), inputs["cloud_model_param"], device) 
        
    
    def local_eval(self, inputs: dict):
        pass 

###

class FedAvg(Algorithm):
    def __init__(self):
        self.name = "FedAvg"
    
    def local_train(self, inputs: dict):
        pass
    
    def local_eval(self, inputs: dict):
        pass

###

class FedProx(Algorithm):
    def __init__(self):
        self.name = "FedProx"
    
    def local_train(self, inputs: dict):
        pass
    
    def local_eval(self, inputs: dict):
        pass