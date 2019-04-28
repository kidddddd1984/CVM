#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from .sample import Sample


class BaseCVM(ABC):
    """
    Abstract CVM class
    ====================

    All cvm calculation must inherit this class and
    implement run(self) method
    """

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset all conditions for new calculation.
        """

    @abstractmethod
    def update_energy(self, e_ints, **kwargs):
        """
        Update energies.
        """

    @abstractmethod
    def process(self, **kwargs):
        """
        Main loop
        """

    def __init__(self,
                 meta: dict,
                 *series,
                 experiment=None,
                 boltzmann_cons=8.6173303e-5,
                 ry2eV=13.605698066,
                 verbose=True):
        super().__init__()
        self.count = 0
        self.verbose = verbose
        self._samples = []
        self.beta = None

        if not isinstance(meta, dict):
            raise TypeError('meta information must be a dict')

        meta = {k.lower(): v.lower() for k, v in meta.items()}
        meta['timestamp'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta = meta

        self.expt = experiment

        # Boltzmann constant
        self.bzc = np.float32(boltzmann_cons)

        # coversion
        self.conv = np.float32(ry2eV)

        ##################
        # init series
        ##################
        for s in series:
            if 'skip' in s and s['skip']:
                continue
            self._build_sample(s)

    @property
    def samples(self):
        return deepcopy(self._samples)

    def add_sample(self, val):
        if not isinstance(val, Sample):
            raise TypeError('sample must be a Sample instance')
        self._samples.append(val)

    def _build_sample(self, series):
        self._samples.append(
            Sample(
                [
                    12,
                    6,
                    24,
                    12,
                    24,
                    8,
                    48,
                    6,
                    12,  # 9th-a
                    24,  # 9th-b
                    4,
                    24,
                    24,
                    48,  # 13th-a
                    24,  # 13th-b
                    48,
                    12,
                    24,  # 16th-a
                    24,  # 16th-b
                    24,  # 17th-a
                    6,  # 17th-b
                    48,  # 18th-a
                    24,  # 18th-b
                    24,
                    48  # 20th
                ],
                self.bzc,
                self.conv,
                **series))

    @classmethod
    def from_samples(cls,
                     meta: dict,
                     *samples,
                     experiment=None,
                     boltzmann_cons=8.6173303e-5,
                     ry2eV=13.605698066,
                     verbose=True):
        ret = cls(
            meta,
            experiment=experiment,
            boltzmann_cons=boltzmann_cons,
            ry2eV=ry2eV,
            verbose=verbose)

        for s in samples:
            ret.add_sample(s)

        return ret

    def __call__(self, *, reset_paras={}, update_en_paras={}, process_paras={}):
        """
        Run the calculation.
        
        Parameters
        ----------
        reset_paras : dict, optional
            The parameters will be passed to ``self.reset`` method, by default empty.
        update_en_paras : dict, optional
            The parameters will be passed to ``self.update_energy`` method, by default empty.
        process_paras : dict, optional
            The parameters will be passed to ``self.process`` method, by default empty.
        """
        # temperature iteration
        for sample in self._samples:
            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            for T in sample.temperature:

                # β = 1/kt
                self.beta = np.float64(pow(self.bzc * T, -1))

                # reset
                self.reset(**reset_paras)
                while self.checker > sample.condition:
                    e_int = sample(T, self.x_[1])
                    self.update_energy(e_int, **update_en_paras)
                    self.process(**process_paras)

                yield T, self.x_[1], self.count, e_int
