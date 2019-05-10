#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

from ..base import BaseCVM


class Tetrahedron(BaseCVM):
    """docstring for tetrahedron"""

    def __init__(self, meta: dict, *, series=None, experiment=None, verbose=True):
        super().__init__(meta, series=series, experiment=experiment, verbose=verbose)

        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.t_ = np.zeros((2, 2, 2, 2), np.float64)
        self.en = np.zeros((2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)
        self.eta_sum = np.float64(0.0)

    def update_energy(self, e_ints, **kwargs):
        ###############################################
        # configuration
        ###############################################

        # pure energy of 2body 1st
        e1 = np.zeros((2, 2), np.float64)
        e1[0, 1] = e1[1, 0] = 0.5 * (e1[0, 0] + e1[1, 1] - e_ints['1st'])

        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = e_ints['triple']

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = e_ints['tetra']

        # energy ε
        it = np.nditer(self.en, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.en[i, j, k, l] = \
                0.5 * (e1[i, j] + e1[i, k] + e1[i, l] +
                       e1[j, k] + e1[j, l] + e1[k, l]) + \
                de31[i, j, k] + de31[i, k, l] + \
                de31[i, j, l] + de31[j, k, l] + \
                de41[i, j, k, l]
            # print('en{} is: {}'.format(it.multi_index, self.en[i, j, k, l]))
            it.iternext()

        # chemical potential
        self.mu[0] = (self.en[0, 0, 0, 0] - self.en[1, 1, 1, 1])
        self.mu[1] = -self.mu[0]

    def reset(self):
        """
        initialize
        """

        it = np.nditer(self.y_, flags=['multi_index'])
        while not it.finished:
            i, j = it.multi_index
            self.y_[i, j] = self.x_[i] * self.x_[j]
            it.iternext()

    def _eta_tetra(self, i, j, k, l):
        """
        η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l)]
                    * X^(-5/8)
                    * Y^(1/2)
        X = x_i * x_j * x_k * x_l
        Y = y_ij * y_ik * y_il * y_jk * y_jl * y_kl
        """
        # exp
        exp = np.exp(-self.beta * self.en[i, j, k, l] + (self.beta / 8) *
                     (self.mu[i] + self.mu[j] + self.mu[k] + self.mu[l]))

        # X
        X = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l]

        # Y
        Y = self.y_[i, j] * self.y_[i, k] * self.y_[i, l] * \
            self.y_[j, k] * self.y_[l, j] * self.y_[k, l]

        return exp * np.power(X, -5 / 8) * np.power(Y, 1 / 2)

    def process(self):
        # counts
        self.count += 1

        # calculate eta
        eta_sum = np.float64(0)
        t_ = np.zeros((2, 2, 2, 2), np.float64)
        it = np.nditer(t_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            t_[i, j, k, l] = self._eta_tetra(i, j, k, l)
            eta_sum += t_[i, j, k, l]
            it.iternext()

        # normalization
        self.checker = np.float64(0)
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        it = np.nditer(t_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            t_[i, j, k, l] /= eta_sum
            self.checker += np.absolute(t_[i, j, k, l] - self.t_[i, j, k, l])

            # t_
            self.t_[i, j, k, l] = t_[i, j, k, l]

            # y_
            self.y_[i, j] += self.t_[i, j, k, l]

            # x_
            self.x_[i] += self.t_[i, j, k, l]
            it.iternext()
