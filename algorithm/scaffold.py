"""
@inproceedings{
    acar2021federated,
    title={Federated Learning Based on Dynamic Regularization},
    author={Durmus Alp Emre Acar and Yue Zhao and Ramon Matas and Matthew Mattina and Paul Whatmough and Venkatesh Saligrama},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=B7v4QMR6Z9w}
}
"""

import copy

import torch
from torch.utils import data

from algorithm.algorithm_base import Algorithm
from client import Client
from dataset import Dataset
from server import Server
from utils import *


class SCAFFOLD(Algorithm):
    def __init__(
        self,
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
        n_minibatch,
        global_learning_rate,
    ):
        super().__init__(
            "SCAFFOLD",
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
        )

        self.n_minibatch = n_minibatch
        self.global_learning_rate = global_learning_rate

    # override
    def local_train(self, client: Client, inputs: dict):
        self.device = client.device

        client.model = self.model_func().to(self.device)
        client.model.load_state_dict(
            copy.deepcopy(dict(inputs["avg_model"].named_parameters()))
        )

        for params in client.model.parameters():
            params.requires_grad = True

        # Scale down c
        state_params_diff_curr = torch.tensor(
            -inputs["state_param"] + inputs["general_state_param"] / client.weight,
            dtype=torch.float32,
            device=self.device,
        )
        print("state_params_diff_curr = ", state_params_diff_curr)
        client.model = self.__train_model(
            client.model,
            client.train_data_X,
            client.train_data_Y,
            inputs["curr_round"],
            state_params_diff_curr,
        )

        curr_model_par = get_model_params([client.model], self.n_param)[
            0
        ]  # get the model parameter after running SCAFFOLD
        print("curr_model_par = ", curr_model_par)

        new_c = (
            inputs["state_param"]
            - inputs["general_state_param"]
            + 1 / self.n_minibatch / self.lr * (inputs["prev_params"] - curr_model_par)
        )
        # Scale up delta c
        inputs["delta_c_sum"] += (new_c - inputs["state_param"]) * client.weight
        inputs["state_param"] = new_c

        client.client_param = curr_model_par

    def __train_model(self, model, trn_x, trn_y, curr_round, state_params_diff):
        decayed_lr = self.lr * (self.lr_decay_per_round**curr_round)

        n_trn = trn_x.shape[0]
        trn_gen = data.DataLoader(
            Dataset(trn_x, trn_y, train=True, dataset_name=self.dataset_name),
            batch_size=self.batch_size,
            shuffle=True,
        )
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        optimizer = torch.optim.SGD(
            model.parameters(), lr=decayed_lr, weight_decay=self.weight_decay
        )
        model.train()
        model = model.to(self.device)

        n_iter_per_epoch = int(np.ceil(n_trn / self.batch_size))
        self.epoch = np.ceil(self.n_minibatch / n_iter_per_epoch).astype(np.int64)
        count_step = 0
        is_done = False

        step_loss = 0
        n_data_step = 0
        for e in range(self.epoch):
            # Training
            if is_done:
                break
            trn_gen_iter = trn_gen.__iter__()
            for _ in range(int(np.ceil(n_trn / self.batch_size))):
                count_step += 1
                if count_step > self.n_minibatch:
                    is_done = True
                    break
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
                        local_par_list = torch.cat(
                            (local_par_list, param.reshape(-1)), 0
                        )

                loss_algo = torch.sum(local_par_list * state_params_diff)
                loss = loss_f_i + loss_algo

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=self.max_norm
                )  # Clip gradients
                optimizer.step()
                step_loss += loss.item() * list(batch_y.size())[0]
                n_data_step += list(batch_y.size())[0]

                if (count_step) % self.print_per == 0:
                    step_loss /= n_data_step
                    if self.weight_decay != None:
                        # Add L2 loss to complete f_i
                        params = get_model_params([model], self.n_param)
                        step_loss += (self.weight_decay) / 2 * np.sum(params * params)
                    print("Step %3d, Training Loss: %.4f" % (count_step, step_loss))
                    step_loss = 0
                    n_data_step = 0
                    model.train()

        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()

        return model

    # override
    def aggregate(self, server: Server, inputs: dict):
        clients_list = inputs["clients_list"]
        selected_clnts_idx = inputs["selected_clnts_idx"]

        clients_param_list = np.array([client.client_param for client in clients_list])
        n_clients = len(clients_list)

        avg_mdl_param = (
            inputs["avg_mdl_param"]
            if not self.noiseless
            else np.mean(clients_param_list[selected_clnts_idx], axis=0)
        )
        avg_mdl_param = (
            self.global_learning_rate * avg_mdl_param
            + (1 - self.global_learning_rate) * inputs["prev_params"]
        )

        inputs["general_state_param"] += 1 / n_clients * inputs["delta_c_sum"]

        server.avg_model = set_model(
            self.model_func().to(server.device), avg_mdl_param, server.device
        )
        server.all_model = set_model(
            self.model_func(), np.mean(clients_param_list, axis=0), server.device
        )
