# -*- coding: utf-8 -*-

import numpy as np


class AirComp(object):
    def __init__(self, n_receive_antennas, K, transmit_power):
        self.n_receive_antennas = n_receive_antennas
        self.K = K
        self.transmit_power = transmit_power

    def transmission(self, d, signal, x, f, h, sigma):
        index = x == 1
        N = self.n_receive_antennas
        K = self.K[index]  # K_m
        K2 = K**2   # K_m^2


        inner = f.conj() @ h[:, index]  # fH h_m(theta)
        inner2 = np.abs(inner) ** 2 # fH h_m(theta) ^ 2

        g = signal
        # mean and variance
        mean = np.mean(g, axis=1)
        g_bar = K @ mean

        var = np.var(g, axis=1)
        var[var < 1e-3] = 1e-3
        var_sqrt = var**0.5

        eta = np.min(self.transmit_power * inner2 / K2 / var) # from (17)
        eta_sqrt = eta**0.5
        b = K * eta_sqrt * var_sqrt * inner.conj() / inner2 # from (17) p_m

        noise_power = sigma * self.transmit_power

        n = (
            (np.random.randn(N, d) + 1j * np.random.randn(N, d))
            / (2) ** 0.5
            * noise_power**0.5
        )

        x_signal = np.tile(b / var_sqrt, (d, 1)).T * (g - np.tile(mean, (d, 1)).T)
        y = h[:, index] @ x_signal + n
        w = np.real((f.conj() @ y / eta_sqrt + g_bar))  # delete to divide by sum(K) as it is not necessary  # (11)

        return w
