#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from .utils import UnitConvert as uc
from .utils import mixed_atomic_weight
from .vibration import ClusterVibration
from .normalizer import Normalizer


class Sample(object):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    def __init__(
            self,
            label,
            *,
            temperature=None,
            energies=None,
            clusters=None,
            global_relax=True,
            mean='arithmetic',
            vibration=True,
            skip=False,
            x_1=0.001,
            condition=1e-07,
            host='host',
            lattice='lattice',
            r_0=None,
            normalizer=None,
            patch=None,
    ):
        super().__init__()
        self.label = label
        self.mean = mean
        self.vibration = vibration
        self.condition = condition
        self.skip = skip
        self.x_1 = x_1
        self.r_0 = r_0
        self.global_relax = global_relax
        self.patch = patch

        # ##########
        # private vars
        # ##########
        self._host = host
        self._lattice = lattice
        self._debye_funcs = defaultdict(None)
        self._lattice_func = None
        self._int_func = None
        self._clusters = None
        self._normalizer = None
        self._temp = None

        if energies is not None:
            self.make_debye_func(energies)
        if normalizer is not None:
            self.normalizer = Normalizer(**normalizer)
        if temperature is not None:
            self.temperature = temperature
        if clusters is not None:
            self.clusters = clusters

        # generate free energy functions base on debye-grüneisen model
        self._int_func = self._gen_int_func()

    def make_debye_func(self, energies):
        if isinstance(energies, pd.DataFrame):
            # calculate debye function
            energy_shift = energies[self._host]
            xs = energies[self._lattice]
            energies = energies.drop(columns=[self._host, self._lattice])

            for c in energies:
                ys = energies[c]
                mass = mixed_atomic_weight(c, mean=self.mean)
                self._debye_funcs[c] = ClusterVibration(xs,
                                                        ys,
                                                        mass,
                                                        energy_shift=energy_shift,
                                                        vibration=self.vibration)

        else:
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

    @property
    def temperature(self):
        return self._temp.copy()

    @temperature.setter
    def temperature(self, temp):
        l = len(temp)  # get length of 'temp'
        if l == 1:
            self._temp = np.array(temp, np.single)
        elif l == 3:
            self._temp = np.linspace(temp[0], temp[1], temp[2])
        else:
            raise NameError('temperature was configured in error format')

    @property
    def clusters(self):
        return deepcopy(self._clusters)

    @clusters.setter
    def clusters(self, val):
        if isinstance(val, dict):
            self._clusters = deepcopy(val)
        else:
            raise TypeError('clusters must be type of <dict> but got %s' % val.__class__.__name__)

    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, val):
        if isinstance(val, Normalizer):
            self._normalizer = val
        else:
            raise TypeError('normalizer must be type of <Normalizer> but got %s' %
                            val.__class__.__name__)

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
                    _lattice_min = minimize_scalar(lambda r: formula(r, T),
                                                   bounds=bounds,
                                                   method='bounded')
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
        int_pair1 = cv.int_energy(xs,
                                  self._normalized_ens['pair1'],
                                  host,
                                  bzc=self.bzc,
                                  num=4,
                                  conv=self.conv,
                                  noVib=False)
        int_pair2 = cv.int_energy(xs,
                                  self._normalized_ens['pair2'],
                                  host,
                                  bzc=self.bzc,
                                  num=6,
                                  conv=self.conv,
                                  noVib=False)
        int_trip = cv.int_energy(xs,
                                 self._normalized_ens['triple'],
                                 host,
                                 bzc=self.bzc,
                                 num=4,
                                 conv=self.conv,
                                 noVib=False)
        int_tetra = cv.int_energy(xs,
                                  self._normalized_ens['tetra'],
                                  host,
                                  bzc=self.bzc,
                                  num=4,
                                  conv=self.conv,
                                  noVib=False)

        return (int_pair1, int_pair2), int_trip, int_tetra

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

    @classmethod
    def int_energy(cls, xs, datas, host, bzc, num, conv, *, noVib=False):
        """
        generate interaction energy
        """
        parts = []
        for data in datas:
            coeff = np.int(data['coefficient'])
            mass = np.float64(data['mass'])
            ys = np.array(data['energy'], np.float64) * conv / num
            part = cls.free_energy(xs, ys, host, mass, bzc, noVib=noVib)
            parts.append((coeff, part))

        def __int(r, T):
            int_en = np.float64(0)
            for part in parts:
                int_en += part[0] * part[1](r, T)
            return int_en * num

        return __int
