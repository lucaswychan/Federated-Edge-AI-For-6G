import copy

import torch
from torch.utils import data

from algorithm.algorithm_base import Algorithm
from client import Client
from dataset import Dataset
from server import Server
from utils import *


class FedAvg(Algorithm):
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
    ):
        super().__init__(
            "FedAvg",
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

    # override
    def local_train(self, client: Client, inputs: dict):
        self.device = client.device

        client.model = self.model_func().to(self.device)
        client.model.load_state_dict(
            copy.deepcopy(dict(inputs["avg_model"].named_parameters()))
        )

        for params in client.model.parameters():
            params.requires_grad = True

        print(
            "client model parameters : ",
            get_model_params([client.model], self.n_param)[0],
        )
        client.model = self.__train_model(
            client.model, client.train_data_X, client.train_data_Y, inputs["curr_round"]
        )
        updated_param = get_model_params([client.model], self.n_param)[0]
        print("after updating, client model parameters : ", updated_param)

        client.client_param = updated_param

    def __train_model(self, model, trn_x, trn_y, curr_round):
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

        for e in range(self.epoch):
            # Training

            trn_gen_iter = trn_gen.__iter__()
            for _ in range(int(np.ceil(n_trn / self.batch_size))):
                batch_x, batch_y = trn_gen_iter.__next__()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y.reshape(-1).long())
                loss = loss / list(batch_y.size())[0]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=self.max_norm
                )  # Clip gradients
                optimizer.step()

            if (e + 1) % self.print_per == 0:
                loss_trn, acc_trn = get_acc_loss(
                    trn_x, trn_y, model, self.dataset_name, self.weight_decay
                )
                print(
                    "Epoch %3d, Training Accuracy: %.4f, Loss: %.4f"
                    % (e + 1, acc_trn, loss_trn)
                )
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
        weight_list = inputs["weight_list"]

        clients_param_list = np.array([client.client_param for client in clients_list])
        weight_list = weight_list.reshape((-1, 1))
        print("weight_list.shape = ", weight_list.shape)

        avg_mdl_param = (
            inputs["avg_mdl_param"]
            if not self.noiseless
            else np.sum(
                clients_param_list[selected_clnts_idx]
                * weight_list[selected_clnts_idx]
                / np.sum(weight_list[selected_clnts_idx]),
                axis=0,
            )
        )

        print("avg_mdl_param = ", avg_mdl_param)

        server.avg_model = set_model(self.model_func(), avg_mdl_param, server.device)
        server.all_model = set_model(
            self.model_func(),
            np.sum(clients_param_list * weight_list / np.sum(weight_list), axis=0),
            server.device,
        )
