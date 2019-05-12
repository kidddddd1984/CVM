#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .normalizer import Normalizer
from .utils import UnitConvert
from .vibration import ClusterVibration


class Sample(defaultdict):
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
            mean='arithmetic',
            vibration=True,
            skip=False,
            x_1=0.001,
            condition=1e-07,
            host='host',
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
        self.patch = patch

        # ##########
        # private vars
        # ##########
        self._host = host
        self._r_0 = r_0
        self._ens = None
        self._lattice_func = None
        self._int_func = None
        self._clusters = None
        self._normalizer = None
        self._temp = None

        if normalizer is not None:
            self.set_normalizer(normalizer)
        if temperature is not None:
            self.set_temperature(temperature)
        if clusters is not None:
            self.set_clusters(**clusters)
        if energies is not None:
            self.set_energies(energies)

    def set_energies(self, energies):

        if isinstance(energies, pd.DataFrame):
            self._ens = energies

            # calculate debye function
            energy_shift = energies[self._host]
            xs = UnitConvert.lc2ad(energies.index.values)
            energies = energies.drop(columns=[self._host])

            for c in energies:
                ys = energies[c]
                self[c] = ClusterVibration(
                    c, xs, ys, energy_shift=energy_shift, mean=self.mean, vibration=self.vibration)
                if self._normalizer and c in self._normalizer:
                    ys = energies[c] + self._normalizer[c]
                    self[c + '~'] = ClusterVibration(
                        c,
                        xs,
                        ys,
                        energy_shift=energy_shift,
                        mean=self.mean,
                        vibration=self.vibration)

        else:
            raise TypeError(
                'energies must be <pd.DataFrame> but got %s' % energies.__class__.__name__)

    def set_temperature(self, temp):
        l = len(temp)  # get length of 'temp'
        if l == 1:
            self._temp = np.array(temp, np.single)
        elif l == 3:
            self._temp = np.linspace(temp[0], temp[1], temp[2])
        else:
            raise NameError('temperature was configured in error format')

    @property
    def energies(self):
        return self._ens

    @property
    def clusters(self):
        return deepcopy(self._clusters)

    def set_clusters(self, **val):
        self._clusters = deepcopy(val)

    @property
    def normalizer(self):
        return self._normalizer

    def set_normalizer(self, val):
        if isinstance(val, Normalizer):
            pass
        elif isinstance(val, dict):
            val = Normalizer(**val)
        else:
            raise TypeError('normalizer must be a dict or has type of <Normalizer> but got %s' %
                            val.__class__.__name__)

        self._normalizer = val
        if self._ens is not None:
            energy_shift = self._ens[self._host]
            xs = UnitConvert.lc2ad(self._ens.index.values)
            for k, v in self._normalizer.items():
                if k in self._ens:
                    ys = (self._ens[k] + v)
                    self[k + '~'] = ClusterVibration(
                        k,
                        xs,
                        ys,
                        energy_shift=energy_shift,
                        mean=self.mean,
                        vibration=self.vibration)

    def ie(self, T, r=None):
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

        def _int(cluster):
            ret_ = 0
            for k, v in cluster.items():
                ret_ += self[k](T, r) * v
            return ret_

        ret = {}
        for k, v in self._clusters.items():
            ret[k] = _int(v)

        return ret

    def __call__(self, *, temperature=None):

        def r_0_func(t):
            x_mins = []
            c_mins = []

            for k, v in self._r_0.items():
                _, x_min = self[k](t, min_x='ws')
                x_mins.append(x_min)
                c_mins.append(v)

            tmp = np.array([c_mins, x_mins])
            index = np.argsort(tmp[0])
            return UnivariateSpline(tmp[0, index], tmp[1, index], k=4)

        if temperature is not None:
            self.set_temperature(temperature)

        for t in self._temp:
            if isinstance(self._r_0, dict):
                yield t, r_0_func(t)
            else:
                yield t, lambda _: self._r_0
