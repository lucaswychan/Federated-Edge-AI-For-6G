from client import Client
from dataset import DatasetObject

num_clients = 100
data_obj = DatasetObject(dataset='CIFAR10', n_client=num_clients, rule='iid', unbalanced_sgm=0)

###
model_name         = 'cifar10' # Model type
communication_rounds = 100
save_period        = 20
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 0.6
lr_decay_per_round = 1
epoch              = 5
learning_rate      = 0.1
print_per          = 5


clients_list = [Client() for _ in range(num_clients)]

for t in range(communication_rounds):
    # 1. Server broadcasts global model to all clients
    # 2. Each client trains on local data
    for client in clients_list:
        client.local_train()
    # 3. Each client sends local model to server
    # 4. Server aggregates all local models into global model
    # 5. Repeat