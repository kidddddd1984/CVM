#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt

from abc import ABC, abstractmethod
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

    def __init__(self, inp, *, verbose=True):
        super().__init__()
        self.count = 0
        self.verbose = verbose
        self.beta = None
        self.series = []
        self.meta = dict(
            host=inp['host'].lower(),
            impurity=inp['impurity'].lower(),
            suffix=inp['suffix'].lower(),
            prefix=inp['prefix'].lower(),
            description=inp['description'],
            data=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.expt = None if 'experiment' not in inp else inp['experiment']

        # Boltzmann constant
        self.bzc = np.float32(8.6173303e-5) if 'bzc' not in inp else np.float32(inp['bzc'])

        # coversion
        self.conv = np.float32(13.605698066) if 'conversion' not in inp else np.float32(
            inp['conversion'])

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise RuntimeError('need at least one calculation set')

        for item in inp['series'][::-1]:
            if 'skip' in item and item['skip']:
                continue
            self.series.append(
                Sample(
                    self.bzc,
                    self.conv,
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
                    **item))

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
        for sample in self.series:
            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            for T in sample.temperature:

                # β = 1/kt
                self.beta = np.float64(pow(self.bzc * T, -1))

                # reset
                self.reset(**reset_paras)
                while self.checker > sample.condition:
                    e_int = sample.get_ints(T, self.x_[1])
                    self.update_energy(e_int, **update_en_paras)
                    self.process(**process_paras)

                yield T, self.x_[1], self.count, e_int
