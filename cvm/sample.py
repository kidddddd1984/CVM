#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from .utils import UnitConvert as uc
from .vibration import ClusterVibration as cv


class Sample(object):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    def __init__(self,
                 coord_num,
                 boltzmann_cons=8.6173303e-5,
                 ry2eV=13.605698066,
                 *,
                 patch=None,
                 **series):
        super().__init__()
        self.bzc = boltzmann_cons
        self.conv = ry2eV
        self.coord_num = np.array(coord_num)
        self.condition = np.float32(series['condition'])
        self.x_1 = np.float64(series['x_1'])
        # chemical potential

        self._lattice_func = None
        self._int_func = None
        self._patch = patch

        self._normalize = deepcopy(series['normalize'])
        self._no_imp_depen = series['no_imp_depen']
        self._fix_r0 = uc.lc2ad(np.float64(series['fix_r0'])) if 'fix_r0' in series else None
        self._host_en = np.array(series['host_en'], np.float64)
        self._label = series['label']
        self._latts = np.array(series['lattice_c'], np.float64)

        # generate temperature steps
        self._temp = self._gen_temp(series['temp'])

        # generate raw interaction energies
        self._raw_ints, self._pair_labels = self._gen_raw_ints(series['energies'])

        # generate normalized energies
        self._normalized_ens = self._gen_normalized_ens(series['energies'])

        # generate free energy functions base on debye-grüneisen model
        self._int_func = self._gen_int_func()

    @property
    def host_energies(self):
        return self._host_en.copy()

    @property
    def raw_ints_ev(self):
        return pd.DataFrame(
            data=self.conv * self._raw_ints, index=self._pair_labels, columns=self._latts)

    @property
    def raw_ints_ry(self):
        return pd.DataFrame(data=self._raw_ints, index=self._pair_labels, columns=self._latts)

    def _gen_raw_ints(self, energies):
        xs = uc.lc2ad(self._latts)

        def pair_label(datas, start=1):
            label = 'pair' + str(start)

            while label in datas:
                yield label
                start += 1
                label = 'pair' + str(start)

        # get interaction energies
        def raw_int(data):
            tmp = np.zeros(len(xs))
            for d in data:
                tmp += d['coefficient'] * np.array(d['energy'])
            return tmp

        pair_labels = [n for n in pair_label(energies)]
        if 'cut_pair' in energies:
            pair_labels = pair_labels[:-energies['cut_pair']]
        return np.array([raw_int(energies[n]) for n in pair_labels]), pair_labels

    def _gen_normalized_ens(self, energies):
        energies = deepcopy(energies)
        int_diffs = self._gen_normalize_diff()

        # 1st total energy
        energies['pair1'][0]['energy'] = np.array(energies['pair1'][0]['energy']) + int_diffs[0]
        energies['pair2'][0]['energy'] = np.array(energies['pair2'][0]['energy']) + int_diffs[1]

        return energies

    def get_r0(self, T, c, *, wc=True):
        if self._fix_r0 is not None:
            return self._fix_r0 if wc else uc.ad2lc(self._fix_r0)

        xs = uc.lc2ad(self._latts)
        host = self._host_en * self.conv

        def phase_en_func(*datas, num=4):
            for data in datas:
                ys = np.array(data['energy'], np.float64) * self.conv / num
                mass = np.array(data['mass'], np.float64)
                yield cv.free_energy(xs, ys, host, mass, self.bzc)

        def lattice_func(*formulas, bounds=None, ratio=None, k=2):
            if bounds is None:
                bounds = (xs[0], xs[-1])

            if ratio is None:
                ratio = [0, 0.25, 0.5, 0.75, 1]

            def _lattice_gene(T, c):
                _lattice_minimums = list()
                for formula in formulas:
                    _lattice_min = minimize_scalar(
                        lambda r: formula(r, T), bounds=bounds, method='bounded')
                    _lattice_minimums.append(_lattice_min.x)

                _lattice_func = UnivariateSpline(ratio, _lattice_minimums[::-1], k=k)
                return _lattice_func(0.0) if self._no_imp_depen else _lattice_func(c)

            return _lattice_gene

        # if None, build new one
        # here, we have to use single-asterisk representation to change the generator to tuple
        if self._lattice_func is None:
            self._lattice_func = lattice_func(*phase_en_func(*self._normalized_ens['tetra'], num=4))

        r0 = self._lattice_func(T, c)
        return r0 if wc else uc.ad2lc(r0)

    def _gen_int_func(self):
        xs = uc.lc2ad(self._latts)
        host = self._host_en * self.conv
        int_pair1 = cv.int_energy(
            xs,
            self._normalized_ens['pair1'],
            host,
            bzc=self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)
        int_pair2 = cv.int_energy(
            xs,
            self._normalized_ens['pair2'],
            host,
            bzc=self.bzc,
            num=6,
            conv=self.conv,
            noVib=False)
        int_trip = cv.int_energy(
            xs,
            self._normalized_ens['triple'],
            host,
            bzc=self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)
        int_tetra = cv.int_energy(
            xs,
            self._normalized_ens['tetra'],
            host,
            bzc=self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)

        return (int_pair1, int_pair2), int_trip, int_tetra

    def _gen_temp(self, temp):
        l = len(temp)  # get length of 'temp'
        if l == 1:
            temp = np.array(temp, np.single)
        elif l == 3:
            temp = np.linspace(temp[0], temp[1], temp[2])
        else:
            raise NameError('temperature was configured in error format')
        return temp

    @property
    def temperature(self):
        return self._temp.copy()

    def __call__(self, T, c):
        """Get interaction energies at concentration c.
        
        Parameters
        ----------
        c : float
            Concentration of impurity.
        
        Returns
        -------
        tuple
            Named tuple contains calculated interaction energies.
        """

        r0 = self.get_r0(T, c)
        int_pair1 = self._int_func[0][0]
        int_pair2 = self._int_func[0][1]
        int_trip = self._int_func[1]
        int_tetra = self._int_func[2]

        pair1 = int_pair1(r0, T)
        pair2 = int_pair2(r0, T)
        trip = int_trip(r0, T)
        tetra = int_tetra(r0, T)
        if self._patch is None:
            return (pair1, pair2), trip, tetra
        tmp = self._patch(uc.ad2lc(r0))
        return (pair1 + tmp[0][0], pair2 + tmp[0][1]), trip + tmp[1], tetra + tmp[2]

    def _gen_normalize_diff(self):
        """
        2nd parameter refer to the neighbor that transfer to
        """

        transfers: list = self._normalize
        _int_diff = np.zeros_like(self._raw_ints)

        for trans in transfers:
            length = len(trans)

            to = 1
            start = to + 1
            end = _int_diff.shape[0]
            percent = 1

            if length > 0:
                to = trans[0]
            if length > 1:
                end = trans[1]
            if length > 2:
                start = trans[2]
            if length > 3:
                percent = trans[3]

            # prepare range
            if start > _int_diff.shape[0] or end > _int_diff.shape[0] or start > end:
                raise IndexError('index error')
            _range = range(start - 1, end)

            # print(_range)
            for index in _range:
                if index == to - 1:
                    pass
                else:
                    _int_diff[to - 1] += self.coord_num[index] * self._raw_ints[
                        index] * percent / self.coord_num[to - 1]

        return _int_diff
