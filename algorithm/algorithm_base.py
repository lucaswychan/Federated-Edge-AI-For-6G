from abc import abstractmethod

from utils import get_acc_loss


class Algorithm:
    def __init__(
        self,
        name,
        lr,
        lr_decay_per_round,
        batch_size,
        epoch,
        weight_decay,
        model_func,
        data_obj,
        n_param,
        max_norm,
        air_comp,
        noiseless,
        save_period,
        print_per,
    ):
        self.name = name
        self.lr = lr
        self.lr_decay_per_round = lr_decay_per_round
        self.batch_size = batch_size
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.model_func = model_func
        self.data_obj = data_obj
        self.n_param = n_param
        self.max_norm = max_norm
        self.air_comp = air_comp
        self.noiseless = noiseless
        self.save_period = save_period
        self.print_per = print_per

    def evaluate(
        self,
        cent_x,
        cent_y,
        avg_model,
        all_model,
        device,
        tst_perf_sel,
        trn_perf_sel,
        tst_perf_all,
        trn_perf_all,
        t,
    ):
        loss_tst, acc_tst = get_acc_loss(
            self.data_obj.tst_x, self.data_obj.tst_y, avg_model, self.data_obj.dataset, device
        )
        tst_perf_sel[t] = [loss_tst, acc_tst]
        print(
            "\n**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
            % (t + 1, acc_tst, loss_tst)
        )

        loss_tst, acc_tst = get_acc_loss(
            cent_x, cent_y, avg_model, self.data_obj.dataset, device
        )
        trn_perf_sel[t] = [loss_tst, acc_tst]
        print(
            "**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f"
            % (t + 1, acc_tst, loss_tst)
        )

        loss_tst, acc_tst = get_acc_loss(
            self.data_obj.tst_x, self.data_obj.tst_y, all_model, self.data_obj.dataset, device
        )
        tst_perf_all[t] = [loss_tst, acc_tst]
        print(
            "**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f"
            % (t + 1, acc_tst, loss_tst)
        )

        loss_tst, acc_tst = get_acc_loss(
            cent_x, cent_y, all_model, self.data_obj.dataset, device
        )
        trn_perf_all[t] = [loss_tst, acc_tst]
        print(
            "**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f\n"
            % (t + 1, acc_tst, loss_tst)
        )

    @abstractmethod
    def local_train(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass
