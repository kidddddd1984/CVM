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
        self._debye_funcs = defaultdict(None)
        self._lattice_func = None
        self._int_func = None
        self._clusters = None
        self._normalizer = None
        self._temp = None

        if energies is not None:
            self.make_debye_func(energies)
        if temperature is not None:
            self.set_temperature(temperature)
        if normalizer is not None:
            self.normalizer = Normalizer(**normalizer)
        if clusters is not None:
            self.clusters = clusters

    def make_debye_func(self, energies):
        if isinstance(energies, pd.DataFrame):
            self._ens = energies

            # calculate debye function
            energy_shift = energies[self._host]
            xs = energies.index.values
            energies = energies.drop(columns=[self._host])

            for c in energies:
                mass, num = mixed_atomic_weight(c, mean=self.mean)
                ys = energies[c] / num
                self._debye_funcs[c] = ClusterVibration(xs,
                                                        ys,
                                                        mass,
                                                        energy_shift=energy_shift,
                                                        vibration=self.vibration)

        else:
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

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

    def __getitem__(self, i):
        return self._debye_funcs[i]

    def __call__(self, *, temperature=None):

        def r_0_func(t):
            x_mins = []
            c_mins = []

            for k, v in self._r_0.items():
                _, x_min = self[k](t)
                x_mins.append(x_min)
                c_mins.append(v)

            return UnivariateSpline(c_mins, x_mins, k=4)

        if temperature is not None:
            self.set_temperature(temperature)
        for t in self._temp:
            if self._r_0 is not dict:
                yield t, self._r_0
            yield t, r_0_func(t)
