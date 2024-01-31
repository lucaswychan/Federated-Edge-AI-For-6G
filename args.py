import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # FedDyn setup
    parser.add_argument('--n_clients', type=int, default=30, help="number of clients")
    
    parser.add_argument('--comm_rounds', type=int, default=50, help="number of communication rounds")
    
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    
    parser.add_argument('--act_prob', type=float, default=0.6, help="probability of active clients")
    
    parser.add_argument('--lr_decay_per_round', type=float, default=1, help="learning rate decay per round")
    
    parser.add_argument('--batch_size', type=int, default=50, help="batch size")
    
    parser.add_argument('--epoch', type=int, default=5, help="local epoch for client training")
    
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight decay")
    
    parser.add_argument('--model_name', type=str, default='cifar10', help="model name")
    
    parser.add_argument('--rand_seed', type=int, default=1, help="random seed")
    
    parser.add_argument('--save_period', type=int, default=1, help="save period")
    
    parser.add_argument('--print_per', type=int, default=5, help="print period")
    
    
    # RIS FL setup
    parser.add_argument('--n_RIS_ele', type=int, default=40, help="number of RIS elements")
    
    parser.add_argument('--n_receive_ant', type=int, default=5, help="number of receive antennas")
    
    parser.add_argument('--alpha_direct', type=float, default=3.76, help="path loss component")
    
    parser.add_argument('--SNR', type=float, default=90.0, help="noise variance/0.1W in dB")
    
    parser.add_argument('--location_range', type=int, default=30, help="location range between clients and RIS")
    
    parser.add_argument('--Jmax', type=int, default=50, help="number of maximum Gibbs Outer loops")
    
    parser.add_argument('--RISON', type=bool, default=True, help="whether RIS is on")
    
    parser.add_argument('--tau', type=float, default=1.0, help="tau, the SCA regularization term")
    
    parser.add_argument('--nit', type=int, default=100, help="I_max, number of maximum SCA loops")
    
    parser.add_argument('--threshold', type=float, default=1e-2, help="epsilon, SCA early stopping criteria")
    
    parser.add_argument('--transmit_power', type=float, default=0.1, help="transmit power")
    
    parser.add_argument('--noiseless', type=bool, default=False, help="whether the channel is noiseless")
    
    args = parser.parse_args()
    
    return args