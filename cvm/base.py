#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy

import pandas as pd
import numpy as np
from pathlib import Path
from .sample import Sample
from .utils import parse_input_set


class BaseCVM(defaultdict, metaclass=ABCMeta):
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

    def __init__(self, meta: dict, *, series=None, experiment=None, verbose=True):
        super().__init__()
        self.count = 0
        self.checker = np.float64(1.0)
        self.verbose = verbose
        self.beta = None

        if not isinstance(meta, dict):
            raise TypeError('meta information must be a dict')

        meta = {k.lower(): v.lower() for k, v in meta.items()}
        meta['timestamp'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta = meta

        self.expt = experiment

        ##################
        # init series
        ##################
        for s in series:
            if 'skip' in s and s['skip']:
                continue
            tmp = Sample(**s)
            self[tmp.label] = tmp

    def add_sample(self, val):
        if not isinstance(val, Sample):
            raise TypeError('sample must be a Sample instance')
        self[val.label] = val

    @classmethod
    def from_samples(cls, meta: dict, *samples, experiment=None, verbose=True):
        ret = cls(meta, experiment=experiment, verbose=verbose)

        for s in samples:
            ret.add_sample(s)

        return ret

    @classmethod
    def from_input_set(cls, path_of_set, *, is_ry_unit=True, is_lattice_unit=True):
        inp = parse_input_set(path_of_set, is_ry_unit=is_ry_unit, is_lattice_unit=is_lattice_unit)
        return cls(**inp)

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
        for label, sample in self.items():
            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            for T, r_0 in sample():

                # β = 1/kt
                self.beta = np.power(8.6173303e-5 * T, -1)

                # reset
                self.count = 0
                self.checker = np.float64(1.0)
                self.reset(**reset_paras)
                while self.checker > sample.condition:
                    e_int = sample.ie(T, r_0(self.x_[1]))
                    self.update_energy(e_int, **update_en_paras)
                    self.process(**process_paras)

                yield label, T, self.x_[1], self.count, e_int
