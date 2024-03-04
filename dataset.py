import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class DatasetObject:
    def __init__(self, dataset, n_client, rule, unbalanced_sgm=0, rule_arg=""):
        self.dataset = dataset
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else "%.3f" % rule_arg
        self.name = "%s_%d_%s_%s" % (
            self.dataset,
            self.n_client,
            self.rule,
            rule_arg_str,
        )
        self.name += "_%f" % unbalanced_sgm if unbalanced_sgm != 0 else ""
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = "Data"
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists("%s/%s" % (self.data_path, self.name)):
            # Get Raw data
            if self.dataset == "mnist":
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )
                trnset = torchvision.datasets.MNIST(
                    root="%s/Raw" % self.data_path,
                    train=True,
                    download=True,
                    transform=transform,
                )
                tstset = torchvision.datasets.MNIST(
                    root="%s/Raw" % self.data_path,
                    train=False,
                    download=True,
                    transform=transform,
                )

                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=len(trnset), shuffle=False, num_workers=1
                )
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=len(tstset), shuffle=False, num_workers=1
                )
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            elif self.dataset == "cifar10":
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
                        ),
                    ]
                )

                trnset = torchvision.datasets.CIFAR10(
                    root="%s/Raw" % self.data_path,
                    train=True,
                    download=True,
                    transform=transform,
                )
                tstset = torchvision.datasets.CIFAR10(
                    root="%s/Raw" % self.data_path,
                    train=False,
                    download=True,
                    transform=transform,
                )

                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=len(trnset), shuffle=False, num_workers=1
                )
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=len(tstset), shuffle=False, num_workers=1
                )
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 10

            elif self.dataset == "cifar100":
                print(self.dataset)
                # mean and std are validated here: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                        ),
                    ]
                )
                trnset = torchvision.datasets.CIFAR100(
                    root="%s/Raw" % self.data_path,
                    train=True,
                    download=True,
                    transform=transform,
                )
                tstset = torchvision.datasets.CIFAR100(
                    root="%s/Raw" % self.data_path,
                    train=False,
                    download=True,
                    transform=transform,
                )
                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=len(trnset), shuffle=False, num_workers=0
                )
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=len(tstset), shuffle=False, num_workers=0
                )
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 100

            elif self.dataset == "emnist":
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )
                trnset = torchvision.datasets.EMNIST(
                    root="%s/Raw" % self.data_path,
                    split="letters",
                    train=True,
                    download=True,
                    transform=transform,
                )
                tstset = torchvision.datasets.EMNIST(
                    root="%s/Raw" % self.data_path,
                    split="letters",
                    train=False,
                    download=True,
                    transform=transform,
                )
                
                # filter the labels with limitation of 10
                filtered_indices = trnset.targets.clone().detach() <= 10
                trnset.targets = trnset.targets[filtered_indices] - 1
                trnset.data = trnset.data[filtered_indices]
                    
                
                filtered_indices = tstset.targets.clone().detach() <= 10
                tstset.targets = tstset.targets[filtered_indices] - 1
                tstset.data = tstset.data[filtered_indices]

                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=len(trnset), shuffle=False, num_workers=1
                )
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=len(tstset), shuffle=False, num_workers=1
                )
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            trn_itr = trn_load.__iter__()
            tst_itr = tst_load.__iter__()
            # labels are of shape (n_data,)
            trn_x, trn_y = trn_itr.__next__()
            tst_x, tst_y = tst_itr.__next__()

            trn_x = trn_x.numpy()
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)

            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            if self.unbalanced_sgm != 0:
                # Draw from lognormal distribution
                clnt_data_list = np.random.lognormal(
                    mean=np.log(n_data_per_clnt),
                    sigma=self.unbalanced_sgm,
                    size=self.n_client,
                )
                clnt_data_list = (
                    clnt_data_list / np.sum(clnt_data_list) * len(trn_y)
                ).astype(int)
                diff = np.sum(clnt_data_list) - len(trn_y)

                # Add/Subtract the excess number starting from first client
                if diff != 0:
                    for clnt_i in range(self.n_client):
                        if clnt_data_list[clnt_i] > diff:
                            clnt_data_list[clnt_i] -= diff
                            break
            else:
                clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
            ###

            if self.rule == "dirichlet":
                cls_priors = np.random.dirichlet(
                    alpha=[self.rule_arg] * self.n_cls, size=self.n_client
                )
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [
                    np.zeros(
                        (clnt_data_list[clnt__], self.channels, self.height, self.width)
                    ).astype(np.float32)
                    for clnt__ in range(self.n_client)
                ]
                clnt_y = [
                    np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64)
                    for clnt__ in range(self.n_client)
                ]

                while np.sum(clnt_data_list) != 0:
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print("Remaining Data: %d" % np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[
                            idx_list[cls_label][cls_amount[cls_label]]
                        ]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[
                            idx_list[cls_label][cls_amount[cls_label]]
                        ]

                        break

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
                prior_real_diff = np.abs(cls_means - cls_priors)
                print("--- Max deviation from prior: %.4f" % np.max(prior_real_diff))
                print("--- Min deviation from prior: %.4f" % np.min(prior_real_diff))

            elif (
                self.rule == "iid"
                and self.dataset == "cifar100"
                and self.unbalanced_sgm == 0
            ):
                assert len(trn_y) // 100 % self.n_client == 0
                # Only have the number clients if it divides 500
                # Perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros(
                    (self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32
                )
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client):  # devices
                    for j in range(100):  # class
                        clnt_x[
                            i,
                            n_cls_sample_per_device
                            * j : n_cls_sample_per_device
                            * (j + 1),
                            :,
                            :,
                            :,
                        ] = trn_x[
                            500 * j
                            + n_cls_sample_per_device * i : 500 * j
                            + n_cls_sample_per_device * (i + 1),
                            :,
                            :,
                            :,
                        ]
                        clnt_y[
                            i,
                            n_cls_sample_per_device
                            * j : n_cls_sample_per_device
                            * (j + 1),
                            :,
                        ] = trn_y[
                            500 * j
                            + n_cls_sample_per_device * i : 500 * j
                            + n_cls_sample_per_device * (i + 1),
                            :,
                        ]

            elif self.rule == "iid":

                clnt_x = [
                    np.zeros(
                        (clnt_data_list[clnt__], self.channels, self.height, self.width)
                    ).astype(np.float32)
                    for clnt__ in range(self.n_client)
                ]
                clnt_y = [
                    np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64)
                    for clnt__ in range(self.n_client)
                ]

                clnt_data_list_cum_sum = np.concatenate(
                    ([0], np.cumsum(clnt_data_list))
                )
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[
                        clnt_data_list_cum_sum[clnt_idx_] : clnt_data_list_cum_sum[
                            clnt_idx_ + 1
                        ]
                    ]
                    clnt_y[clnt_idx_] = trn_y[
                        clnt_data_list_cum_sum[clnt_idx_] : clnt_data_list_cum_sum[
                            clnt_idx_ + 1
                        ]
                    ]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            self.clnt_x = clnt_x
            self.clnt_y = clnt_y

            self.tst_x = tst_x
            self.tst_y = tst_y

            # Save data
            os.mkdir("%s/%s" % (self.data_path, self.name))

            np.save("%s/%s/clnt_x.npy" % (self.data_path, self.name), clnt_x)
            np.save("%s/%s/clnt_y.npy" % (self.data_path, self.name), clnt_y)

            np.save("%s/%s/tst_x.npy" % (self.data_path, self.name), tst_x)
            np.save("%s/%s/tst_y.npy" % (self.data_path, self.name), tst_y)

        else:
            print("Data is already downloaded in the folder.")
            self.clnt_x = np.load(
                "%s/%s/clnt_x.npy" % (self.data_path, self.name), allow_pickle=True
            )
            self.clnt_y = np.load(
                "%s/%s/clnt_y.npy" % (self.data_path, self.name), allow_pickle=True
            )
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load(
                "%s/%s/tst_x.npy" % (self.data_path, self.name), allow_pickle=True
            )
            self.tst_y = np.load(
                "%s/%s/tst_y.npy" % (self.data_path, self.name), allow_pickle=True
            )

            if self.dataset == "mnist":
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10
            elif self.dataset == "cifar10":
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 10
            elif self.dataset == "cifar100":
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 100
            elif self.dataset == "fashion_mnist":
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10
            elif self.dataset == "emnist":
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

        print("Class frequencies:")
        count = 0
        for clnt in range(self.n_client):
            print(
                "Client %3d: " % clnt
                + ", ".join(
                    [
                        "%.3f" % np.mean(self.clnt_y[clnt] == cls)
                        for cls in range(self.n_cls)
                    ]
                )
                + ", Amount:%d" % self.clnt_y[clnt].shape[0]
            )
            count += self.clnt_y[clnt].shape[0]

        print("Total Amount:%d" % count)
        print("--------")

        print(
            "      Test: "
            + ", ".join(
                ["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]
            )
            + ", Amount:%d" % self.tst_y.shape[0]
        )


def generate_syn_logistic(
    dimension,
    n_clnt,
    n_cls,
    avg_data=4,
    alpha=1.0,
    beta=0.0,
    theta=0.0,
    iid_sol=False,
    iid_dat=False,
):

    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    samples_per_user = (
        np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)
    ).astype(int)
    print("samples per user")
    print(samples_per_user)
    print("sum %d" % np.sum(samples_per_user))

    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat:  # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))

    if iid_sol:  # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))

    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(
            -1, 1
        )

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


class DatasetSynthetic:
    def __init__(
        self,
        alpha,
        beta,
        theta,
        iid_sol,
        iid_data,
        n_dim,
        n_clnt,
        n_cls,
        avg_data,
        name_prefix,
    ):
        self.dataset = "synt"
        self.name = name_prefix + "_"
        self.name += "%d_%d_%d_%d_%f_%f_%f_%s_%s" % (
            n_dim,
            n_clnt,
            n_cls,
            avg_data,
            alpha,
            beta,
            theta,
            iid_sol,
            iid_data,
        )

        data_path = "Data"
        if not os.path.exists("%s/%s/" % (data_path, self.name)):
            # Generate data
            print("Sythetize")
            data_x, data_y = generate_syn_logistic(
                dimension=n_dim,
                n_clnt=n_clnt,
                n_cls=n_cls,
                avg_data=avg_data,
                alpha=alpha,
                beta=beta,
                theta=theta,
                iid_sol=iid_sol,
                iid_dat=iid_data,
            )
            os.mkdir("%s/%s/" % (data_path, self.name))
            np.save("%s/%s/data_x.npy" % (data_path, self.name), data_x)
            np.save("%s/%s/data_y.npy" % (data_path, self.name), data_y)
        else:
            # Load data
            print("Load")
            data_x = np.load(
                "%s/%s/data_x.npy" % (data_path, self.name), allow_pickle=True
            )
            data_y = np.load(
                "%s/%s/data_y.npy" % (data_path, self.name), allow_pickle=True
            )

        for clnt in range(n_clnt):
            print(
                ", ".join(["%.4f" % np.mean(data_y[clnt] == t) for t in range(n_cls)])
            )

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=""):
        self.name = dataset_name
        if self.name == "mnist" or self.name == "synt" or self.name == "emnist":
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == "cifar10" or self.name == "cifar100":
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype("float32")

        elif self.name == "shakespeare":

            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == "mnist" or self.name == "synt" or self.name == "emnist":
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == "cifar10" or self.name == "cifar100":
            img = self.X_data[idx]
            if self.train:
                img = (
                    np.flip(img, axis=2).copy() if (np.random.rand() > 0.5) else img
                )  # Horizontal flip
                if np.random.rand() > 0.5:
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(
                        np.float32
                    )
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1 : dim_1 + 32, dim_2 : dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == "shakespeare":
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y
