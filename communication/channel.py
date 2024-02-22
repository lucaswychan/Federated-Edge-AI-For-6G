"""
@article{
    liu2021reconfigurable,
    title={Reconfigurable Intelligent Surface Enabled Federated Learning: A Unified Communication-Learning Design Approach},
    author={Liu, Hang and Yuan, Xiaojun and Zhang, Ying-Jun Angela},
    journal={arXiv preprint arXiv:2011.10282},
    year={2021},
    eprint={2011.10282},
    archivePrefix={arXiv},
    primaryClass={cs.IT}
}
"""

import numpy as np


class Channel(object):
    def __init__(
        self,
        SNR,
        n_clients,
        location_range,
        fc,
        alpha_direct,
        n_RIS_ele,
        n_receive_ant,
        User_Gain,
        x0,
        BS,
        BS_Gain,
        RIS,
        RIS_Gain,
        dimen_RIS,
    ):
        self.SNR = SNR
        self.n_clients = n_clients
        self.location_range = location_range
        self.fc = fc
        self.alpha_direct = alpha_direct
        self.n_RIS_ele = n_RIS_ele
        self.n_receive_ant = n_receive_ant
        self.User_Gain = User_Gain
        self.x = x0

        self.BS = BS
        self.BS_Gain = BS_Gain

        self.RIS = RIS
        self.RIS_Gain = RIS_Gain
        self.dimen_RIS = dimen_RIS

    def generate(self):
        ref = (1e-10) ** 0.5
        sigma_n = np.power(10, -self.SNR / 10)
        sigma = sigma_n / ref**2  # [100,100+range]

        # set 2
        dx2 = (
            np.random.rand(int(self.n_clients - np.round(self.n_clients / 2)))
            * self.location_range
            + 200
        )
        dx1 = (
            np.random.rand(int(np.round(self.n_clients / 2))) * self.location_range
            - self.location_range
        )  # [-location_range , 0]

        dx = np.concatenate((dx1, dx2))
        np.random.shuffle(dx)

        dy = np.random.rand(self.n_clients) * 20 - 10
        d_UR = (
            (dx - self.RIS[0]) ** 2 + (dy - self.RIS[1]) ** 2 + self.RIS[2] ** 2
        ) ** 0.5
        d_RB = np.linalg.norm(self.BS - self.RIS)
        d_direct = (
            (dx - self.BS[0]) ** 2 + (dy - self.BS[1]) ** 2 + self.BS[2] ** 2
        ) ** 0.5
        PL_direct = (
            self.BS_Gain
            * self.User_Gain
            * (3 * 10**8 / self.fc / 4 / np.pi / d_direct) ** self.alpha_direct
        )
        PL_RIS = (
            self.BS_Gain
            * self.User_Gain
            * self.RIS_Gain
            * self.n_RIS_ele**2
            * self.dimen_RIS**2
            / 4
            / np.pi
            * (3 * 10**8 / self.fc / 4 / np.pi / d_UR) ** 2
            * (3 * 10**8 / self.fc / 4 / np.pi / d_RB) ** 2
        )
        # channels
        h_d = (
            np.random.randn(self.n_receive_ant, self.n_clients)
            + 1j * np.random.randn(self.n_receive_ant, self.n_clients)
        ) / 2**0.5
        h_d = h_d @ np.diag(PL_direct**0.5) / ref

        H_RB = (
            np.random.randn(self.n_receive_ant, self.n_RIS_ele)
            + 1j * np.random.randn(self.n_receive_ant, self.n_RIS_ele)
        ) / 2**0.5

        h_UR = (
            np.random.randn(self.n_RIS_ele, self.n_clients)
            + 1j * np.random.randn(self.n_RIS_ele, self.n_clients)
        ) / 2**0.5
        h_UR = h_UR @ np.diag(PL_RIS**0.5) / ref

        G = np.zeros(
            [self.n_receive_ant, self.n_RIS_ele, self.n_clients], dtype=complex
        )
        for j in range(self.n_clients):
            G[:, :, j] = H_RB @ np.diag(h_UR[:, j])

        return h_d, G, self.x, sigma
