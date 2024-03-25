<div align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-F63939?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://en.wikipedia.org/wiki/Federated_learning">
    <img src="https://img.shields.io/badge/Federated%20Learning-3333FF?style=for-the-badge&logoColor=white" alt="FL">
  </a>
  <a href="https://arxiv.org/pdf/2102.00742.pdf">
    <img src="https://img.shields.io/badge/RIS-00CC66?style=for-the-badge&logoColor=white" alt="RIS">
  </a>
</div>

<hr/>

# Federated Edge AI for 6G

## Abstract
Federated Learning (FL) is a decentralized approach to machine learning that addresses the crucial issue of data privacy. However, decentralization presents challenges such as data and system heterogeneity, as well as communication channel difficulties. By tapping into the potential of state-of-the-art 6G technology and leveraging advancements in FL computation algorithms, it is possible to effectively overcome the computational and communication complexities inherent in the FL system. Techniques such as FedDyn (Federated Dynamic Regularization) and RIS-FL (Reconfigurable Intelligence Surface-assisted Federated Learning) have been chosen to address the computation and communication problems resulting from the decentralization of FL, with named RIS-AirFedDyn. Through these methods, it is possible to enhance the efficiency and performance of FL systems while maintaining data privacy and security.

<hr/>

## Credit
<div style="font-size:1rem">
<a href="https://arxiv.org/abs/2111.04263">
  Federated Learning Based on Dynamic Regularization
</a>
<p>
  Code :  &nbsp 
  <a href="https://github.com/AntixK/FedDyn">
     FedDyn
  </a>
</p>
<a href="https://arxiv.org/abs/2011.10282">
  Reconfigurable Intelligent Surface Enabled Federated Learning: A Unified Communication-Learning Design Approach
</a>
<p>
  Code :  &nbsp 
  <a href="https://github.com/liuhang1994/RIS-FL">
     RIS-FL
  </a>
</p>
</div>

<hr/>

## Dependencies
* Python >= 3.6
* numpy==1.21.6 (For Dirichlet Case)
* torch
* torchvision
* cvxpy
* matplotlib
  
Or you can install all the packages via
```
pip install -r requirements.txt
```

<hr/>

## Instructions
There are four algorithms available to play with, which are FedDyn, FedAvg, FedProx, and SCAFFOLD  
The default algorithm is FedDyn, but you can feel free to change the algorithm by adding ```--algorithm_name={FedAvg, FedProx, FedDyn, SCAFFOLD}```  
  
For more details on the parameters, please visit [Parameters](#parameters)
```
python3 fl_main.py
```

<hr/>

## Add new algorithm
This whole code structure enjoys the advantage OOP brings, so adding new algorithms on top of the existing codes is a piece of cake.
1. Create a new Python file under the ```algorithm``` directory
<br></br>
E.g. ```fedsplit.py```
<br></br>
2. Construct a new class inheriting ```Algorithm``` in the corresponding Python file.
<br></br>
E.g.
```python
class FedSplit(Algorithm):
    def __init__(self, lr, lr_decay_per_round, batch_size, epoch, weight_decay, model_func, n_param, max_norm, noiseless, dataset_name, save_period, print_per, new_parameter):
        super().init("FedSplit", lr, lr_decay_per_round, batch_size, epoch, weight_decay, model_func, n_param, max_norm, noiseless, dataset_name, save_period, print_per)

        self.new_parameter = new_parameter
```
<br></br>
3. Override the method ```local_train``` and ```aggregate```.
<br></br>
E.g.
```python
class FedSplit(Algorithm):
    .
    .
    .
    def local_train(self, client: Client, inputs: dict):
        # client local training

    def __train_model(self, ...):
        # helper function for local_train (Optional)

    def aggregate(self, server: Server, inputs: dict):
        # aggregate the global model
```
<br></br>
4. Add the corresponding class in ```AlgorithmFactory```, which is in ```algorithm/algorithm_factory.py```
<br></br>
E.g.
```python
class AlgorithmFactory:
    def __init__(self, args):
        ...

    def create_algorithm(self, algorithm_name) -> Algorithm:
        ...
        elif algorithm_name == "FedAvg":
            ...

        elif algorithm_name == "FedSplit":
            new_parameter = 0.3

            algorithm = FedSplit(
                self.args.lr,
                self.args.lr_decay_per_round,
                self.args.batch_size,
                self.args.epoch,
                self.args.weight_decay,
                self.args.model_func,
                self.args.n_param,
                self.args.max_norm,
                self.args.noiseless,
                self.args.data_obj.dataset,
                self.args.save_period,
                self.args.print_per,
                new_parameter
            )
```
<br></br>
5. Create the required parameters for this algorithm in ```fl_main.py```
<br></br>
E.g.
```python
# these lines should be in fl_main.py
required_parameter = np.ones((args.n_clients, n_param))
.
.
.
inputs["required_parameter"] = required_parameter
```
Note that if ```inputs["required_parameter"]``` is updated in ```client.local_train``` or ```server.aggregate```, it should be explicitly updated in ```fl_main.py```  
i.e.  
```python
# these lines should be in fl_main.py
if args.algorithm_name == "FedSplit":
    required_parameter = inputs["required_parameter"]
```
<br></br>
6. Have fun to play with your algorithm !
<br></br>

<hr/>

## Parameters
There are various parameters required by the algorithms. 
For more details you can visit ```args.py```
| Parameter Name  | Meaning| Default Value| Type | Choice/Range |
| ---------- | -----------|-----------|-----------|-----------|
| algorithm_name   | algorithm for training   |FedDyn   |str   | FedDyn, FedAvg, FedProx, SCAFFOLD |
|  n_clients  | number of clients   | 30  |int   | [1, inf) | 
| comm_rounds   | number of communication rounds   |50   |int   | [1, inf) | 
| lr   | learning rate   |0.03   |float   | (0, inf) | 
| act_prob   | probability of randomly choosing active clients   |0.9   |float   |[0,1] | 
| lr_decay_per_round   | learning rate decay per round   |0.99   |float   | [0, inf) | 
| batch_size   | number of data per batch   |50   |int   | [1, total data size] | 
| epoch   | local epoch for client training   |5   |int   | [1, inf) | 
| weight_decay   | weight decay  |0.01   |float   | (0, inf) | 
| max_norm   | max norm for gradient clipping   |10.0   |float  | (0, inf) | 
| model_name   | model for training. The name is also the corresponding dataset name   |cifar10   |str   | linear, mnist, emnist, cifar10, cifar100, resnet18, shakespeare |
| rule   | the rule of data partitioning   |iid   |str   |iid, dirichlet |
|  rand_seed  | random seed   |1   |int   | [0, inf) | 
| save_period   | period to save the models   |1   |int   | [1, comm_rounds] |
| print_per   | period to print the training result   |5   |int   | [1, epoch] | 
| n_RIS_ele   | number of RIS elements   |40   |int   | [0, inf) |
| n_receive_ant   | number of receive antennas   |5   |int   | [0, inf) | 
| alpha_direct   | path loss component   |3.76   |float   | [0, inf) | 
| SNR   | noise variance/0.1W in dB   |90.0   |float   | [0, inf) | 
| location_range   | location range between clients and RIS   |30   |int   | [0, inf) | 
| Jmax   | number of maximum Gibbs Outer loops   |50   |int   | [1, inf) |
| tau   | the SCA regularization term   |0.03   |float   | [0, inf) | 
| nit   | I_max, number of maximum SCA loops   |100   |int   | [1, inf) | 
| threshold   | epsilon, SCA early stopping criteria   |0.01   |float   | [0, inf) | 
| transmit_power   | transmit power of clients   |0.003   |float   | [0, inf) | 
| noiseless   | whether the channel is noiseless   |False   |bool   | True, False | 

<hr/>
