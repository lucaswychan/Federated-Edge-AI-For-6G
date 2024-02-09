import copy
import os
import time

import numpy as np
import torch

from air_comp import AirComp
from algorithm.algorithm_factory import AlgorithmFactory
from args import args_parser
from channel import Channel
from client import Client
from dataset import DatasetObject
from model import Model
from optimize import Gibbs
from utils import get_model_params, save_performance


def main():
    # get all the constant parameters
    args = args_parser()

    ############################################################################################################
    # data for train and test
    storage_path = "LEAF/shakespeare/data/"
    # data_obj =  DatasetObject(dataset='CIFAR10', n_client=args.n_clients, unbalanced_sgm=0, rule='Dirichlet', rule_arg=0.6)
    data_obj = DatasetObject(
        dataset="CIFAR10", n_client=args.n_clients, rule="iid", unbalanced_sgm=0
    )
    client_x_all = data_obj.clnt_x
    client_y_all = data_obj.clnt_y
    cent_x = np.concatenate(client_x_all, axis=0)
    cent_y = np.concatenate(client_y_all, axis=0)
    dataset = data_obj.dataset

    ############################################################################################################

    # the weight corresponds to the number of data that the client i has
    # the more data the client has, the larger the weight is
    weight_list = np.asarray([len(client_y_all[i]) for i in range(args.n_clients)])
    # FedDyn and SCAFFOLD initialization
    if args.algorithm_name == "FedDyn" or args.algorithm_name == "SCAFFOLD":
        weight_list = weight_list / np.sum(weight_list) * args.n_clients

    # global parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # algorithm parameters
    model_func = lambda: Model(args.model_name)
    init_model = model_func()

    init_par_list = get_model_params([init_model])[
        0
    ]  # parameters of the iargs.nitial model
    n_param = len(init_par_list)

    #
    np.random.seed(args.rand_seed)

    ############################################################################################################
    # Channel setup

    # User-BS Path loss exponent
    fc = 915 * 10**6  # carrier frequency, wavelength lambda=3.0*10**8/fc
    BS_Gain = 10 ** (5.0 / 10)  # BS antenna gain
    RIS_Gain = 10 ** (5.0 / 10)  # RIS antenna gain
    User_Gain = 10 ** (0.0 / 10)  # User antenna gain
    dimen_RIS = 1.0 / 10  # dimension length of RIS element/wavelength
    BS = np.array([-50, 0, 10])  # Cartesian coordinate of BS
    RIS = np.array([0, 0, 10])  # Cartesian coordinate of RIS
    #
    x0 = np.ones([args.n_clients], dtype=int)
    #
    channel = Channel(
        SNR=args.SNR,
        n_clients=args.n_clients,
        location_range=args.location_range,
        fc=fc,
        alpha_direct=args.alpha_direct,
        n_RIS_ele=args.n_RIS_ele,
        n_receive_ant=args.n_receive_ant,
        User_Gain=User_Gain,
        x0=x0,
        BS=BS,
        BS_Gain=BS_Gain,
        RIS=RIS,
        RIS_Gain=RIS_Gain,
        dimen_RIS=dimen_RIS,
    )

    gibbs = Gibbs(
        n_clients=args.n_clients,
        n_receive_ant=args.n_receive_ant,
        n_RIS_ele=args.n_RIS_ele,
        Jmax=args.Jmax,
        K=weight_list,
        RISON=args.RISON,
        tau=args.tau,
        nit=args.nit,
        threshold=args.threshold,
    )

    air_comp = AirComp(
        n_receive_ant=args.n_receive_ant,
        K=weight_list,
        transmit_power=args.transmit_power,
    )

    ############################################################################################################

    # create the algorihtm object
    args.data_obj = data_obj
    args.n_param = n_param
    args.air_comp = air_comp
    args.model_func = model_func
    args.init_model = init_model

    algorithm_factory = AlgorithmFactory(args)
    algorithm = algorithm_factory.create_algorithm(args.algorithm_name)

    clients_list = np.array(
        [
            Client(
                algorithm=algorithm,
                device=device,
                weight=weight_list[i],
                train_data_X=client_x_all[i],
                train_data_Y=client_y_all[i],
                model=init_model,
                client_param=np.copy(init_par_list),
            )
            for i in range(args.n_clients)
        ]
    )

    ############################################################################################################

    if not os.path.exists(
        "Output/%s/%s_init_mdl.pt" % (data_obj.name, args.model_name)
    ):
        print("New directory!")
        os.mkdir("Output/%s/" % (data_obj.name))
        torch.save(
            init_model.state_dict(),
            "Output/%s/%s_init_mdl.pt" % (data_obj.name, args.model_name),
        )
    else:
        # Load model
        init_model.load_state_dict(
            torch.load("Output/%s/%s_init_mdl.pt" % (data_obj.name, args.model_name))
        )

    ############################################################################################################
    # Model Initialization (feel free to add new parameters in this section if using this frameworks)

    # For FedDyn
    if args.algorithm_name == "FedDyn":
        local_param_list = np.zeros((args.n_clients, n_param)).astype("float32")
        # cloud (server) model  (for FedDyn)
        cloud_model = model_func().to(device)
        cloud_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        cloud_model_param = get_model_params([cloud_model], n_param)[0]

    # For SCAFFOLD
    elif args.algorithm_name == "SCAFFOLD":
        state_param_list = np.zeros((args.n_clients + 1, n_param)).astype(
            "float32"
        )  # including cloud state

    trn_perf_sel = np.zeros((args.comm_rounds, 2))
    trn_perf_all = np.zeros((args.comm_rounds, 2))
    tst_perf_sel = np.zeros((args.comm_rounds, 2))
    tst_perf_all = np.zeros((args.comm_rounds, 2))

    # average model
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # all clients model
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    ############################################################################################################
    # Start Communicatoin
    print()
    print("=" * 80)
    print("Device: %s" % device)
    print("Model: %s" % args.model_name)
    print("Epochs: %d" % args.epoch)
    print("Number of clients: %d" % args.n_clients)
    print("Number of communication rounds: %d" % args.comm_rounds)
    print("Training starts with algorithm: %s\n" % algorithm.name)
    print("The system is {}".format("noiseless" if args.noiseless else "noisy"))
    print("=" * 80, end="\n\n")

    for t in range(args.comm_rounds):

        print("This is the {0}-th trial".format(t + 1))
        
        ############################################################################################################
        
        if not args.noiseless:
            print("\nRunning Gibbs Optimization...\n")
            # generate the channel
            h_d, G, x, sigma = channel.generate()

            start = time.time()

            # get the optimized parameters based on RIS-FL
            x_optim, f_optim, h_optim = gibbs.optimize(h_d, G, x, sigma)

            end = time.time()
            print("Running time of Gibbs Optimization: {} seconds\n".format(end - start))
        
        else:
            print("\nRunning Random Selection...\n")
            inc_seed = 0
            x_optim = np.array([0])
            while np.sum(x_optim) == 0:
                # Fix randomness in client selection
                np.random.seed(t + args.rand_seed + inc_seed)
                active_clients = np.random.uniform(size=args.n_clients)
                x_optim = active_clients <= args.act_prob
                inc_seed += 1
                
        ############################################################################################################
        # get the selected clients

        selected_clnts_idx = np.where(x_optim == 1)[0]  # get the index of the selected clients
        selected_clnts = clients_list[selected_clnts_idx]

        print("Selected Clients Index: {}".format(x_optim))
        print("Selected Clients: %s\n" % (", ".join(["%2d" % clnt for clnt in selected_clnts_idx])))

        ############################################################################################################
        # set up the required parameters for the algorithms

        # FedDyn
        if args.algorithm_name == "FedDyn":
            cloud_model_param_tensor = torch.tensor(
                cloud_model_param, dtype=torch.float32, device=device
            )

        # FedProx
        elif args.algorithm_name == "FedProx":
            avg_model_param = get_model_params([avg_model], n_param)[0]
            avg_model_param_tensor = torch.tensor(
                avg_model_param, dtype=torch.float32, device=device
            )

        # SCAFFOLD
        elif args.algorithm_name == "SCAFFOLD":
            delta_c_sum = np.zeros(n_param)
            prev_params = get_model_params([avg_model], n_param)[0]

        ############################################################################################################
        # clients training

        for i, client in enumerate(selected_clnts):
            # Train locally
            print("---- Training client %d" % selected_clnts_idx[i])

            inputs = {
                "curr_round": t,
                "avg_model": avg_model,
            }

            if args.algorithm_name == "FedDyn":
                inputs["cloud_model"] = cloud_model
                inputs["cloud_model_param"] = cloud_model_param
                inputs["cloud_model_param_tensor"] = cloud_model_param_tensor
                inputs["local_param"] = local_param_list[selected_clnts_idx[i]]

            elif args.algorithm_name == "FedProx":
                inputs["avg_model_param_tensor"] = avg_model_param_tensor

            elif args.algorithm_name == "SCAFFOLD":
                inputs["state_param"] = state_param_list[selected_clnts_idx[i]]
                inputs["general_state_param"] = state_param_list[-1]
                inputs["prev_params"] = prev_params
                inputs["delta_c_sum"] = delta_c_sum

            client.local_train(inputs)

            # update the parameters (For FedDyn)
            if args.algorithm_name == "FedDyn":
                local_param_list[selected_clnts_idx[i]] = inputs["local_param"]

            # update the parameters (For SCAFFOLD)
            elif args.algorithm_name == "SCAFFOLD":
                delta_c_sum = inputs["delta_c_sum"]
                state_param_list[selected_clnts_idx[i]] = inputs["state_param"]

        ############################################################################################################
        # aggregation

        inputs = {
            "clients_list": clients_list,
            "selected_clnts_idx": selected_clnts_idx,
            "avg_model": avg_model,
            "all_model": all_model,
            "weight_list": weight_list,
        }

        if args.algorithm_name == "FedDyn":
            inputs["local_param_list"] = local_param_list
            inputs["cloud_model"] = cloud_model
            inputs["cloud_model_param"] = cloud_model_param

        elif args.algorithm_name == "SCAFFOLD":
            inputs["prev_params"] = prev_params
            inputs["delta_c_sum"] = delta_c_sum
            inputs["general_state_param"] = state_param_list[-1]

        # pass the AirComp optimization parameters
        if not args.noiseless:
            inputs["x"] = x_optim
            inputs["f"] = f_optim
            inputs["h"] = h_optim
            inputs["sigma"] = sigma

        algorithm.aggregate(inputs)

        # update the parameters after aggregation
        avg_model = inputs["avg_model"]
        all_model = inputs["all_model"]

        # For FedDyn
        if args.algorithm_name == "FedDyn":
            cloud_model = inputs["cloud_model"]
            cloud_model_param = inputs["cloud_model_param"]

        # For SCAFFOLD
        elif args.algorithm_name == "SCAFFOLD":
            state_param_list[-1] = inputs["general_state_param"]

        # get the test accuracy
        algorithm.evaluate(
            data_obj,
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
        )

        ############################################################################################################

    save_performance(
        args.comm_rounds,
        tst_perf_all,
        algorithm.name,
        data_obj.name,
        args.n_clients,
        args.noiseless,
    )


############################################################################################################

if __name__ == "__main__":
    main()
