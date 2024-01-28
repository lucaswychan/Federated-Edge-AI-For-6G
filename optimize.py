# -*- coding: utf-8 -*-
import copy
import numpy as np

np.set_printoptions(precision=6, threshold=1e3)
from scipy.optimize import minimize


class Gibbs:
    def __init__(self, n_clients, n_receive_antennas, n_RIS_ele, Jmax, K, RISON, tau, nit, threshold):
        self.n_clients = n_clients
        self.n_receive_antennas = n_receive_antennas
        self.n_RIS_ele = n_RIS_ele
        self.Jmax = Jmax
        self.K = K
        self.RISON = RISON
        
        # for sca_fmincon
        self.tau = tau
        self.nit = nit
        self.threshold = threshold
        
        self.verbose = 0
        
    def optimize(self, h_d, G, x0, sigma):
        N = self.n_receive_antennas
        L = self.n_RIS_ele
        M = self.n_clients
        
        K = self.K / np.mean(self.K)  # normalize K to speed up floating computation
        K2 = K**2
        Ksum2 = sum(K) ** 2
        x = x0
        # inital the return values
        obj_new = np.zeros(self.Jmax + 1)
        f_store = np.zeros([N, self.Jmax + 1], dtype=complex)
        theta_store = np.zeros([L, self.Jmax + 1], dtype=complex)
        x_store = np.zeros([self.Jmax + 1, M], dtype=int)

        # the first loop
        ind = 0
        [obj_new[ind], x_store[ind, :], f, theta] = self.find_obj_inner(
            x, K, K2, Ksum2, h_d, G, None, None, sigma
        )

        theta_store[:, ind] = copy.deepcopy(theta)
        f_store[:, ind] = copy.deepcopy(f)
        beta = min(1, obj_new[ind])
        alpha = 0.9
        if self.verbose > 1:
            print("The inital guess: {}, obj={:.6f}".format(x, obj_new[ind]))
        elif self.verbose == 1:
            print("The inital guess obj={:.6f}".format(obj_new[ind]))
        f_loop = np.tile(f, (M + 1, 1))
        theta_loop = np.tile(theta, (M + 1, 1))

        for j in range(self.Jmax):
            if self.verbose > 1:
                print(
                    "This is the {}-th Gibbs sampling iteration, beta= {:.6f}".format(
                        j + 1, beta
                    )
                )
            
            # store the possible transition solution and their objectives
            X_sample = np.zeros([M + 1, M], dtype=int)
            Temp = np.zeros(M + 1)
            # the first transition => no change
            X_sample[0, :] = copy.deepcopy(x)
            Temp[0] = copy.deepcopy(obj_new[ind])
            f_loop[0] = copy.deepcopy(f)
            theta_loop[0] = copy.deepcopy(theta)
            # 2--M+1-th transition, change only 1 position
            for m in range(M):

                # filp the m-th position
                x_sam = copy.deepcopy(x)
                x_sam[m] = copy.deepcopy((x_sam[m] + 1) % 2)
                X_sample[m + 1, :] = copy.deepcopy(x_sam)
                Temp[m + 1], _, f_loop[m + 1], theta_loop[m + 1] = self.find_obj_inner(
                    x_sam,
                    K,
                    K2,
                    Ksum2,
                    h_d,
                    G,
                    f_loop[m + 1],
                    theta_loop[m + 1],
                    sigma,
                )
                if self.verbose > 1:
                    print("sol:{} with obj={:.6f}".format(x_sam, Temp[m + 1]))
            temp2 = Temp

            Lambda = np.exp(-1 * temp2 / beta)
            Lambda = Lambda / sum(Lambda)
            while np.isnan(Lambda).any():
                beta = beta / alpha
                Lambda = np.exp(-1.0 * temp2 / beta)
                Lambda = Lambda / sum(Lambda)
            if self.verbose > 1:
                print("The obj distribution: {}".format(temp2))
                print("The Lambda distribution: {}".format(Lambda))
                
            kk_prime = np.random.choice(M + 1, p=Lambda)
            x = copy.deepcopy(X_sample[kk_prime, :])
            f = copy.deepcopy(f_loop[kk_prime])
            theta = copy.deepcopy(theta_loop[kk_prime])
            ind += 1
            obj_new[ind] = copy.deepcopy(Temp[kk_prime])
            x_store[ind, :] = copy.deepcopy(x)
            theta_store[:, ind] = copy.deepcopy(theta)
            f_store[:, ind] = copy.deepcopy(f)
            
            if self.verbose > 1:
                print(
                    "Choose the solution {}, with objective {:.6f}".format(x, obj_new[ind])
                )

            if self.verbose:
                print(
                    "This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}".format(
                        j + 1, beta, obj_new[ind]
                    )
                )

            beta = max(alpha * beta, 1e-4)

        return x_store, obj_new, f_store, theta_store
    
    def find_obj_inner(self, x, K, K2, Ksum2, h_d, G, f0, theta0, sigma):
        N = self.n_receive_antennas
        L = self.n_RIS_ele
        M = self.n_clients
        
        if sum(x) == 0:
            obj = np.inf

            theta = np.ones([L], dtype=complex)
            f = h_d[:, 0] / np.linalg.norm(h_d[:, 0])
            if not self.RISON:
                theta = np.zeros([L])
        else:
            index = x == 1
            
            f, theta, _ = self.sca_fmincon(
                h_d[:, index], G[:, :, index], f0, theta0, x, K2[index]
            )

            h = np.zeros([N, M], dtype=complex)
            for i in range(M):
                h[:, i] = h_d[:, i] + G[:, :, i] @ theta
            gain = K2 / (np.abs(np.conjugate(f) @ h) ** 2) * sigma
            obj = (
                np.max(gain[index]) / (sum(K[index])) ** 2
                + 4 / Ksum2 * (sum(K[~index])) ** 2
            )
        return obj, x, f, theta
    
    # Algorithm 1
    def sca_fmincon(self, h_d, G, f, theta, x, K2):   # (25)
        N = self.n_receive_antennas
        L = self.n_RIS_ele
        I = sum(x)
        
        if theta is None:
            theta = np.ones([L], dtype=complex)
        if not self.RISON:
            theta = np.zeros([L], dtype=complex)
        result = np.zeros(self.nit)
        h = np.zeros([N, I], dtype=complex)
        for i in range(I):
            h[:, i] = h_d[:, i] + G[:, :, i] @ theta

        if f is None:
            f = h[:, 0] / np.linalg.norm(h[:, 0])

        obj = min(np.abs(np.conjugate(f) @ h) ** 2 / K2)

        for it in range(self.nit):
            obj_pre = copy.deepcopy(obj)
            a = np.zeros([N, I], dtype=complex)
            b = np.zeros([L, I], dtype=complex)
            c = np.zeros([1, I], dtype=complex)
            F_cro = np.outer(f, np.conjugate(f))
            for i in range(I): # (26)
                a[:, i] = self.tau * K2[i] * f + np.outer(h[:, i], np.conjugate(h[:, i])) @ f
                if self.RISON:
                    b[:, i] = self.tau * K2[i] * theta + G[:, :, i].conj().T @ F_cro @ h[:, i]
                    c[:, i] = (
                        np.abs(np.conjugate(f) @ h[:, i]) ** 2
                        + 2 * self.tau * K2[i] * (L + 1)
                        + 2
                        * np.real(
                            (theta.conj().T) @ (G[:, :, i].conj().T) @ F_cro @ h[:, i]
                        )
                    )
                else:
                    c[:, i] = np.abs(np.conjugate(f) @ h[:, i]) ** 2 + 2 * self.tau * K2[i]

            fun = lambda mu: np.real(
                2 * np.linalg.norm(a @ mu) + 2 * np.linalg.norm(b @ mu, ord=1) - c @ mu   # (29)
            )

            cons = {"type": "eq", "fun": lambda mu: K2 @ mu - 1}
            bnds = ((0, None) for i in range(I))
            res = minimize(fun, 1 / K2, bounds=tuple(bnds), constraints=cons)
            if ~res.success:
                pass

            fn = a @ res.x
            thetan = b @ res.x
            fn = fn / np.linalg.norm(fn)

            if self.RISON:
                thetan = thetan / np.abs(thetan)
                theta = thetan
            f = fn
            for i in range(I):
                h[:, i] = h_d[:, i] + G[:, :, i] @ theta
            obj = min(np.abs(np.conjugate(f) @ h) ** 2 / K2)   # (24)
            result[it] = copy.deepcopy(obj)
            if self.verbose > 2:
                print(
                    "  Iteration {} Obj {:.6f} Opt Obj {:.6f}".format(
                        it, result[it], res.fun[0]
                    )
                )

            if np.abs(obj - obj_pre) / min(1, abs(obj)) <= self.threshold:
                break
        
        if self.verbose == 1:
            print(
                " SCA Take {} iterations with final obj {:.6f}".format(it + 1, result[it])
            )

        result = result[0:it]
        return f, theta, result