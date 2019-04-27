#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

from ..base import BaseCVM


class Pair(BaseCVM):
    """docstring for process"""

    def __init__(self, inp):
        super().__init__(inp)
        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.en = np.zeros((2, 2), np.float64)
        self.mu = np.zeros((2), np.float64)
        self.beta = np.float64(0)
        self.eta_sum = np.float64(0)

        # init
        self.__init()

        # run
        self.__run()
        print(self.count)
        inp.output['x_'] = np.array(self.x_).reshape(2).tolist()

    def __init(self):
        self.x_[0] = self.x_1
        self.x_[1] = 1 - self.x_1

        en[1, 1] = en[0, 1] = 0.5 * (en[0, 0] + en[1, 1] - self.int_pair)
        self.lam = np.float64(0.0)
        self.eta_sum = zeros((inp.x_.size, inp.x_.size))
        self.en = np.matrix([[0., inp.en[0, 0]], [inp.en[0, 0], 0.]])
        self.mu = np.matrix([[2 * inp.mu, 0], [0, -2 * inp.mu]])
        self.en = -0.5 * self.en

    def __eta_ij(self):
        """
        η_ij = (x_*x_j)^((2ω-1)/2ω) * exp( -βe_ij + (β/2ω)(mu_i + mu_j) )
        """
        entro = np.power((self.x_ * self.x_.T), ((2 * self.omega - 1) / (2 * self.omega)))
        energy = np.exp(-self.beta * self.en + (self.beta / (2 * self.omega)) * self.mu)
        return np.multiply(entro, energy)

    def __y_ij(self):
        """
        y_ij = η_ij * exp(β*λ/ω)
        """
        eta_sum = self.__eta_ij()
        eta_sum = np.sum(eta_sum)
        self.lam = -np.log(eta_sum) * self.omega / self.beta
        return eta_sum * np.power(eta_sum, -1)

    def __run(self):
        """
        TODO: need doc
        """
        lam = self.lam
        y_ij = self.__y_ij()
        self.count += 1
        print('lambda is: {}'.format(self.lam))
        self.x_ = y_ij.sum(1)
        print(y_ij)
        print(self.x_)
        print('\n')
        if np.absolute(self.lam - lam) > 1e-6:
            self.__run()
