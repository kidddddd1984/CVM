# Copyright 2019 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime as dt
import os
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict, namedtuple
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .sample import Sample
from .utils import parse_input_set, UnitConvert

__all__ = ['BaseCVM']


class BaseCVM(defaultdict, metaclass=ABCMeta):
    """
    CVM is a general CVM (Cluster Variation Method) calculation framework using  NIM (Natural Iteration Method). 
 
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
        The main loop to execute a NIM calculation step-by-step..
        """

    def __init__(self, meta: dict, *, series: List[dict] = None, experiment: dict = None):
        """BaseCVM
        
        Parameters
        ----------
        meta : dict
            Meta information.
            It can be anything that will be used to describe this calculation set.
            This information will be passed over the whole calculation.
        series : List[dict], optional
            Sample data contains raw energies, and other calculation needed parameters, by default None
        experiment : dict, optional
            The experiment data, by default None
        
        """
        super().__init__()
        self.count = 0
        self.checker = np.float64(1.0)
        self.beta = None
        self._results: list = []

        if not isinstance(meta, dict):
            raise TypeError('meta information must be a dict')

        meta = {k.lower(): v.lower() for k, v in meta.items()}
        meta['timestamp'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta = meta

        self.experiment = experiment

        ##################
        # init series
        ##################
        for s in series:
            tmp = Sample(**s)
            self.add_sample(tmp)

    @property
    def results(self) -> pd.DataFrame:
        """Get calculated results.
        
        Returns
        -------
        results
        """
        if not self._results:
            return None
        return pd.DataFrame(self._results).set_index('label')

    def add_sample(self, sample: Sample):
        """Add sample data
        
        Parameters
        ----------
        sample : Sample
            Instance of :py:class:`.Sample`.
        
        """
        if not isinstance(sample, Sample):
            raise TypeError('sample must be a Sample instance')
        self[sample.label] = sample

        setattr(self, f'tag_{sample.label}', self[sample.label])

    @classmethod
    def from_samples(cls, meta: dict, *samples: Sample, experiment: dict = None):
        """Build a CVM instance from series samples.
        
        Parameters
        ----------
        meta : dict
            Meta information.
            It can be anything that will be used to describe this calculation set.
            This information will be passed over the whole calculation.
        samples : Sample
            List of Series sample.
        experiment : dict, optional
            The experiment data, by default None

        Returns
        -------
        BaseCVM
            New CVM instance.
        """
        ret = cls(meta, experiment=experiment)
        for s in samples:
            ret.add_sample(s)
        return ret

    @classmethod
    def from_input_set(cls, path_of_set: str):
        """Build a CVM instance from input set.
        
        Parameters
        ----------
        path_of_set : str
            The path of input set dir.
        
        Returns
        -------
        BaseCVM
            New CVM instance.
        """
        inp = parse_input_set(path_of_set)
        return cls(**inp)

    def run(self,
            *labels: str,
            verbose: bool = False,
            early_stopping: float = 0.2,
            sample_paras: dict = {},
            reset_paras: dict = {},
            update_en_paras: dict = {},
            process_paras: dict = {}):
        """
        Run CVM calculation.
        
        Parameters
        ----------
        labels: str
            If not ``None``, only execute listed series samples.
        verbose : bool, optional
            Set to ``True`` to show extra information when running calculations, by default False
        sample_ite_paras : dict, optional
            The parameters will be passed to ``sample.ite`` and ``sample.__call__`` method, by default empty.        
        reset_paras : dict, optional
            The parameters will be passed to ``self.reset`` method, by default empty.
        update_en_paras : dict, optional
            The parameters will be passed to ``self.update_energy`` method, by default empty.
        process_paras : dict, optional
            The parameters will be passed to ``self.process`` method, by default empty.
        """

        if len(labels) == 0:
            labels = self.keys()

        for l in labels:
            with tqdm(total=len(self[l].temperatures), desc=l) as pbar:
                for _ in self(
                        l,
                        verbose=verbose,
                        early_stopping=early_stopping,
                        sample_paras=sample_paras,
                        reset_paras=reset_paras,
                        update_en_paras=update_en_paras,
                        process_paras=process_paras):
                    pbar.update()

    def __call__(self,
                 *labels: str,
                 verbose: bool = False,
                 early_stopping: float = 0.2,
                 sample_paras: dict = {},
                 reset_paras: dict = {},
                 update_en_paras: dict = {},
                 process_paras: dict = {}):
        """
        Run CVM calculation as generator.
        Will yield calculation details on each step.

        Parameters
        ----------
        labels: str
            If not ``None``, only execute listed series samples.
        verbose : bool, optional
            Set to ``True`` to show extra information when running calculations, by default False
        sample_ite_paras : dict, optional
            The parameters will be passed to ``sample.ite`` and ``sample.__call__`` method,
            by default empty.
        reset_paras : dict, optional
            The parameters will be passed to ``self.reset`` method, by default empty.
        update_en_paras : dict, optional
            The parameters will be passed to ``self.update_energy`` method, by default empty.
        process_paras : dict, optional
            The parameters will be passed to ``self.process`` method, by default empty.
        """

        ret = namedtuple(
            'status', 'label, temperature, concentration, lattice_param, num_of_ite, int_energy')

        self._results = []
        # temperature iteration
        for label, sample in self.items():
            if sample.skip:
                continue
            if labels and label not in labels:
                continue

            self.x_[1] = sample.x_1
            self.x_[0] = 1 - sample.x_1

            for T, r_0 in sample.ite(**sample_paras):

                # β = 1/kt
                self.beta = np.power(8.6173303e-5 * T, -1)

                # reset
                self.count = 0
                self.checker = np.float64(1.0)
                self.reset(**reset_paras)
                while self.checker > sample.condition:
                    r_0_ = r_0(self.x_[1])
                    e_int = sample(T=T, r=r_0_, **sample_paras)
                    self.update_energy(e_int, **update_en_paras)
                    self.process(**process_paras)

                    if early_stopping > 0 and self.x_[1] >= early_stopping:
                        sys.stderr.write(
                            '...Early stopping applied because the impurity concentration is greater than 0.5.'
                            f'Current temperature is: {T}, concentration is: {self.x_[1]}')
                        return

                    # add result
                    self._results.append(
                        OrderedDict(
                            label=label,
                            temperature=T,
                            concentration=self.x_[1],
                            lattice_param=UnitConvert.ad2lc(r_0_) if r_0_ is not None else r_0_,
                            num_of_ite=self.count,
                            **e_int._asdict()))

                    if verbose:
                        yield ret(
                            label=label,
                            temperature=T,
                            concentration=self.x_[1],
                            lattice_param=UnitConvert.ad2lc(r_0_) if r_0_ is not None else r_0_,
                            num_of_ite=self.count,
                            int_energy=e_int)

                if not verbose:
                    yield ret(
                        label=label,
                        temperature=T,
                        concentration=self.x_[1],
                        lattice_param=UnitConvert.ad2lc(r_0_) if r_0_ is not None else r_0_,
                        num_of_ite=self.count,
                        int_energy=e_int)

    def __repr__(self):
        s1 = '  | \n  |-'
        s2 = '  | '
        s3 = '  |-'
        header = [self.__class__.__name__ + ':']

        flag = os.getenv('simple_print')

        meta = f'\n{s3}'.join(['meta:'] + [f'{k}: {v}' for k, v in self.meta.items()])
        if not flag:
            return f'\n{s1}'.join(header + [f'\n{s2}'.join(str(meta).split('\n'))] +
                                  [f'\n{s2}'.join(str(v).split('\n')) for v in self.values()])
        else:
            return f'\n{s1}'.join(header + [f'\n{s2}'.join(str(meta).split('\n'))] +
                                  ['{}'.format(str(v).split('\n')[0][:-1]) for v in self.values()])
