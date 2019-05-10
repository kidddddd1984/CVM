#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from ..base import BaseCVM


class TetraOctahedron(BaseCVM):
    """docstring for tetraOctahedron"""

    def __init__(self, meta: dict, *, series=None, experiment=None, verbose=True):
        super().__init__(meta, series=series, experiment=experiment, verbose=verbose)

        ####################
        # define var
        ####################
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        self.z_ = np.zeros((2, 2, 2), np.float64)
        self.enT = np.zeros((2, 2, 2, 2), np.float64)
        self.enO = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
        self.beta = np.float64(0.0)
        self.mu = np.zeros((2), np.float64)
        self.af_ = np.zeros((2, 2, 2), np.float64)
        self.main_condition = np.float64(1e-3)
        self.sub_condition = np.float64(1e-2)

    def update_energy(self, e_ints, **kwargs):
        ###############################################
        # configuration
        ###############################################

        # pure energy of 2body-1st
        en1 = np.zeros((2, 2), np.float64)
        en1[0, 1] = en1[1, 0] = 0.5 * (en1[0, 0] + en1[1, 1] - e_ints['1st'])

        #############################################
        # tetrahedron
        #############################################
        # 3body-1st interaction energy
        de31 = np.zeros((2, 2, 2), np.float64)
        de31[1, 1, 1] = e_ints['triple']

        # 4body-1st interaction energy
        de41 = np.zeros((2, 2, 2, 2), np.float64)
        de41[1, 1, 1, 1] = e_ints['tetra']

        # energy ε
        it = np.nditer(self.enT, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            self.enT[i, j, k, l] = \
                0.5 * (en1[i, j] + en1[i, k] + en1[i, l] +
                       en1[j, k] + en1[j, l] + en1[l, k]) +\
                (de31[i, j, k] + de31[i, k, l] +
                 de31[i, j, l] + de31[j, k, l]) +\
                de41[i, j, k, l]
            it.iternext()
        # =============================================

        ########################
        # octahedron
        ########################
        # pure energy of 2body-1st
        en2 = np.zeros((2, 2), np.float64)
        en2[0, 1] = en2[1, 0] = \
            0.5 * (en2[0, 0] + en2[1, 1] - e_ints['2nd'])

        # energy ε
        it = np.nditer(self.enO, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n = it.multi_index
            self.enO[i, j, k, l, m, n] = en2[i, k] + en2[j, l] + en2[n, m]
            it.iternext()
        # ==============================================

        # chemical potential
        self.mu[0] = (self.enT[0, 0, 0, 0] + self.enO[0, 0, 0, 0, 0, 0]) - \
            (self.enT[1, 1, 1, 1] + self.enO[1, 1, 1, 1, 1, 1])
        self.mu[1] = -self.mu[0]

    def reset(self):

        self.af_ = np.zeros((2, 2, 2), np.float64)
        self.main_condition = np.float64(1e-3)
        self.sub_condition = np.float64(1e-2)

        it = np.nditer(self.z_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            self.z_[i, j, k] = self.x_[i] * self.x_[j] * self.x_[k]
            self.y_[i, j] = self.x_[i] * self.x_[j]
            it.iternext()

    def __eta_tetra(self, i, j, k, l):
        """
        η_ijkl = exp[-β*e_ijkl + (β/8)(mu_i + mu_j + mu_k + mu_l) + Alpha]
                    * X^(1/8)
                    * Y^(-1/2)
                    * Z^(1/1)
        Alpha = af_ijk + af_ijl + af_ikl + af_jkl
        X = x_i * x_j * x_k * x_l
        Y = y_ij * y_ik * y_il * y_jk * y_jl * y_kl
        Z = z_ijk * z_ikl * z_ijl * z_jkl
        """
        # exp
        exp = np.exp(-self.beta * self.enT[i, j, k, l] + (self.beta / 8) *
                     (self.mu[i] + self.mu[j] + self.mu[k] + self.mu[l]) + self.af_[i, j, k] +
                     self.af_[i, j, l] + self.af_[i, k, l] + self.af_[j, k, l])

        # X
        X = self.x_[i] * self.x_[j] * self.x_[k] * self.x_[l]

        # Y
        Y = self.y_[i, j] * self.y_[i, k] * self.y_[i, l] * \
            self.y_[j, k] * self.y_[l, j] * self.y_[k, l]

        # Z
        Z = self.z_[i, j, k] * self.z_[i, k, l] *\
            self.z_[i, j, l] * self.z_[j, k, l]

        return exp * np.power(X, 1 / 8) * np.power(Y, -1 / 2) * Z

    def _eta_octa(self, i, j, k, l, m, n):
        """
        η_ijklmn = exp[-β*e_ijklmn -
                        (af_ijm + af_ijn + af_jkm + af_jkn +
                        af_klm + af_kln + af_ilm + af_iln)]
        """
        # Alpha
        af = self.af_[i, j, m] + self.af_[i, j, n] + self.af_[j, k, m] +\
            self.af_[j, k, n] + self.af_[k, l, m] + self.af_[k, l, n] +\
            self.af_[i, l, m] + self.af_[i, l, n]

        # exp
        return np.exp(-self.beta * self.enO[i, j, k, l, m, n] - af)

    def _eta_TO(self):
        # counts
        self.count += 1

        # tetrahedron
        self.zt_ = np.zeros((2, 2, 2), np.float64)
        wt_ = np.zeros((2, 2, 2, 2), np.float64)
        it = np.nditer(wt_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            wt_[i, j, k, l] = self.__eta_tetra(i, j, k, l)
            self.zt_[i, j, k] += wt_[i, j, k, l]
            it.iternext()

        # octahedron
        self.zo_ = np.zeros((2, 2, 2), np.float64)
        wo_ = np.zeros((2, 2, 2, 2, 2, 2), np.float64)
        it = np.nditer(wo_, flags=['multi_index'])
        while not it.finished:
            i, j, k, l, m, n = it.multi_index
            wo_[i, j, k, l, m, n] = self._eta_octa(i, j, k, l, m, n)
            self.zo_[i, j, m] += wo_[i, j, k, l, m, n]
            it.iternext()

        # alpha
        sub_checker = np.float64(0)
        daf = np.zeros((2, 2, 2), np.float64)
        it = np.nditer(self.af_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            daf[i, j, k] = 0.15 * np.log(self.zo_[i, j, k] / self.zt_[i, j, k])
            self.af_[i, j, k] += daf[i, j, k]
            sub_checker += np.absolute(daf[i, j, k])
            it.iternext()

        return sub_checker

    def process(self):
        # check sub consistant
        sub_checker = self._eta_TO()
        while sub_checker > self.sub_condition:
            sub_checker = self._eta_TO()

        # get concentration
        eta_sum = np.float64(0)
        self.checker = np.float64(0)
        self.x_ = np.zeros((2), np.float64)
        self.y_ = np.zeros((2, 2), np.float64)
        it = np.nditer(self.zt_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            self.zt_[i, j, k] = (self.zo_[i, j, k] + 2 * self.zt_[i, j, k]) / 3
            eta_sum += self.zt_[i, j, k]
            it.iternext()

        it = np.nditer(self.z_, flags=['multi_index'])
        while not it.finished:
            i, j, k = it.multi_index
            self.zt_[i, j, k] /= eta_sum
            self.checker += np.absolute(self.z_[i, j, k] - self.zt_[i, j, k])

            # z_
            self.z_[i, j, k] = self.zt_[i, j, k]

            # y_
            self.y_[i, j] += self.z_[i, j, k]

            # x_
            self.x_[i] += self.z_[i, j, k]
            it.iternext()
