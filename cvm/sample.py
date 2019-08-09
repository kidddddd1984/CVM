#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import defaultdict, namedtuple
from copy import deepcopy

from typing import Union, Callable
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from .normalizer import Normalizer
from .utils import UnitConvert, parse_formula, mixed_atomic_weight, logspace
from .vibration import ClusterVibration

__all__ = ['Sample']


class Sample(defaultdict):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    def __init__(self,
                 label: str,
                 host: str,
                 impurity: str,
                 *,
                 temperature: list = None,
                 energies: pd.DataFrame = None,
                 clusters: dict = None,
                 mean: str = 'arithmetic',
                 vibration: bool = True,
                 skip: bool = False,
                 x_1: float = 0.001,
                 condition: float = 1e-07,
                 r_0: Union[float, dict] = None,
                 normalizer: Union[dict, Normalizer] = None):
        """
        
        Parameters
        ----------
        label : str
            The Sample label.
        host : str, optional
            The column name of host energies in ``energies``.
        host : str, optional
            The column name of impurity energies in ``energies``.
        temperature : list, optional
            Temperature steps follow the format [start, stop, # of steps].
            By default ``None``.
        energies : pd.DataFrame, optional
            A :py:class:`pandas.DataFrame` object contain raw energies.
            By default ``None``.
        clusters : dict, optional
            Set how to calculate interaction energy.
            By default ``None``.
        mean : str, optional
            Specific how to mix atom weights.
            Can be 'arithmetic', 'harmonic', and 'geometric'.
            By default 'arithmetic'.
        vibration : bool, optional
            Specific whether or not to import the thermal vibration effect.
            By default ``True``.
        skip : bool, optional
            Set to true to skip this series sample.
            By default ``False``.
        x_1 : float, optional
            The initialization concentration of impurity.
            By default ``0.001``.
        condition : float, optional
            Convergence condition.
            By default ``1e-07``.
        r_0 : Union[float, dict, None], optional
            Set how to estimate r_0 from the given T and c.
            If ``None``, r_0 will be calculated from each phase respectively.
            If constant, will ignore T and c.
            If dict, will do a parabolic curve fitting.
            By default ``None``.
        normalizer : Union[dict, Normalizer], optional
            Configuration of a :py:class:`.Normalizer` or an instance.
            If given, this will be used to normalize the long-range interaction energy.
            By default ``None``.
        """
        super().__init__()
        self.label = label
        self.mean = mean
        self.vibration = vibration
        self.condition = condition
        self.skip = skip
        self.host = host
        self.impurity = impurity
        self.x_1 = x_1

        # ##########
        # private vars
        # ##########
        self._en_min = defaultdict(float)
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
            self._temp = self.set_temperature(temperature)
        if clusters is not None:
            self.set_clusters(**clusters)
        if energies is not None:
            self.set_energies(energies)

    def set_energies(self, energies):

        if isinstance(energies, pd.DataFrame):
            self._ens = energies

            # calculate debye function
            host = energies[self.host]
            impurity = energies[self.impurity]
            xs = UnitConvert.lc2ad(energies.index.values)

            # get minimum from a polynomial
            poly_min = minimize_scalar(UnivariateSpline(xs, host, k=4),
                                       bounds=(xs[0], xs[-1]),
                                       method='bounded')
            self._en_min[self.host] = poly_min.fun

            poly_min = minimize_scalar(UnivariateSpline(xs, impurity, k=4),
                                       bounds=(xs[0], xs[-1]),
                                       method='bounded')
            self._en_min[self.impurity] = poly_min.fun

            for c in energies:
                if c in [self.host, self.impurity]:
                    continue
                comp = parse_formula(c)
                ys = energies[c]
                mass, num = mixed_atomic_weight(c, mean=self.mean)

                for k, v in comp.items():
                    ys -= self._en_min[k] * v

                self[c] = ClusterVibration(
                    label=c,
                    xs=xs,
                    ys=host * num + ys,
                    mass=mass,
                    num=num,
                )
                setattr(self, c, self[c])

                if self._normalizer and c in self._normalizer:
                    ys += self._normalizer[c]
                    c = f'{c}_'
                    self[c] = ClusterVibration(
                        label=c,
                        xs=xs,
                        ys=host * num + ys,
                        mass=mass,
                        num=num,
                    )
                    setattr(self, c, self[c])

        else:
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

    def set_temperature(self, temp):
        if isinstance(temp, dict):
            if 'log_scale' not in temp or not temp['log_scale']:
                return np.linspace(temp['start'], temp['stop'], temp['steps'])
            return logspace(temp['start'], temp['stop'], temp['steps'])
        if isinstance(temp, (list, np.ndarray)):
            return deepcopy(temp)
        if isinstance(temp, (float, int)):
            return [temp]
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

    @property
    def temperatures(self):
        return self._temp

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
            host = self._ens[self.host]
            impurity = self._ens[self.impurity]
            xs = UnitConvert.lc2ad(self._ens.index.values)

            for c, v in self._normalizer.items():
                if c in self._ens:
                    ys = self._ens[c] + v
                    comp = parse_formula(c)
                    mass, num = mixed_atomic_weight(c, mean=self.mean)

                    for k_, v_ in comp.items():
                        ys -= self._en_min[k_] * v_

                    ys += host * num
                    c = f'{c}_'
                    self[c] = ClusterVibration(
                        label=c,
                        xs=xs,
                        ys=ys,
                        mass=mass,
                        num=num,
                    )
                    setattr(self, c, self[c])

    def __call__(self,
                 *,
                 T: float = None,
                 r: float = None,
                 convert_r: bool = False,
                 vibration: bool = None,
                 energy_patch: Callable[[float, float], namedtuple] = None,
                 **kwargs):
        """Get interaction energies at given T, and r.
        
        Parameters
        ----------
        T : float
            Temperature.
        r : float, optional
            Atomic distance. By default ``None``.
        vibration: bool
            Specific whether or not to import the thermal vibration effect.
        convert_r: bool, optional
            If parameter ``r`` is given in lattice constance, must set this flag to ``True`` to execute an auto-unit-conversion.
        energy_patch: Callable[[float, float], namedtuple], optional
            A patch that will be used to correct the returns of interaction energy.
            By default ``None``.
        
        Returns
        -------
        namedtuple: namedtuple
            Named tuple contains calculated interaction energies.
        """

        del kwargs

        if convert_r:
            r = UnitConvert.lc2ad(r)

        if vibration is None:
            vibration = self.vibration

        if T is None:
            vibration = False

        def _int(cluster, r_):
            ret_ = 0
            for k, v in cluster.items():
                if vibration:
                    ret_ += self[k](T=T, r=r_) * v
                else:
                    ret_ += self[k](r=r_) * v

            return ret_

        ret = {}
        for k, v in self._clusters.items():
            try:
                ret[k] = _int(v, r)
            except KeyError as e:
                raise KeyError(f'configuration of `{k}` in parameter <series.clusters> '
                               f'reference an unknown phase {e}')

        if energy_patch is not None:
            patch = energy_patch(T, r)

            for k, v in patch.items():
                ret[k] += v

        return namedtuple('interaction_energy', self._clusters.keys())(**ret)

    def ite(self, *, temperature: list = None, k: int = 3, vibration: bool = None, **kwargs):
        """Iterate over each temperature

        Parameters
        ----------
        temperature : list, optional
            Reset temperature steps, by default None
        k: int
            Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline. 
        vibration: bool
            Specific whether or not to import the thermal vibration effect.

        Yields
        -------
        T: float
            Temperature at current step.
        r_func: Callable[[float], float]
            A function receiving impurity concentration c and
            returns corresponding atomic distance r.
        """
        del kwargs

        if vibration is None:
            vibration = self.vibration

        def r_0_func(t):
            x_mins = []
            c_mins = []

            for k_, v in self._r_0.items():
                if vibration:
                    _, x_min = self[k_](T=t, min_x='ws')
                else:
                    _, x_min = self[k_](min_x='ws')
                x_mins.append(x_min)
                c_mins.append(v)

            tmp = np.array([c_mins, x_mins])
            index = np.argsort(tmp[0])
            return UnivariateSpline(tmp[0, index], tmp[1, index], k=k)

        if temperature is not None:
            temperature = self.set_temperature(temperature)
        else:
            temperature = self._temp

        for t in temperature:
            if isinstance(self._r_0, dict):
                yield t, r_0_func(t)
            else:
                yield t, lambda _: UnitConvert.lc2ad(self._r_0)

    def __repr__(self):
        s1 = '  | \n  |-'
        s2 = '  | '
        header = [f'{self.label}--<skip: {self.skip}>:']

        return f'\n{s1}'.join(header + [f'\n{s2}'.join(str(self.normalizer).split('\n'))] +
                              [f'\n{s2}'.join(str(v).split('\n')) for v in self.values()])
