#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
import threading

import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import UnivariateSpline

from .cluster_vibration import ClusterVibration as cv
from .unit_convert import *
from .sample import Sample


class CVM(threading.Thread):
    """
    Abstract CVM class
    ====================

    All cvm calculation must inherit this class and
    implement run(self) method
    """

    __slots__ = (
        'count',  # count for self-consistent
        'output',  # output
        'bzc',  # Boltzmann constant
        'conv',  # unit conversion
        'series',  # calculation series
    )

    def __init__(self, inp):
        super(CVM, self).__init__()
        self.count = 0
        self.series = []
        self.output = {'meta': {}, 'results': [], 'experiment': {}}

        ##################
        # init output
        ##################
        # experiment
        self.output['experiment'] = inp['experiment']

        # Meta
        date_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output['meta']['host'] = inp['host'].lower()
        self.output['meta']['impurity'] = inp['impurity'].lower()
        self.output['meta']['suffix'] = inp['suffix'].lower()
        self.output['meta']['prefix'] = inp['prefix'].lower()
        self.output['meta']['description'] = inp['description']
        self.output['meta']['structure'] = inp['structure']
        self.output['meta']['date'] = date_time_str

        # Boltzmann constant
        self.bzc = np.float32(inp['bzc'])

        # coversion
        self.conv = np.float32(inp['conversion'])

        ##################
        # init series
        ##################
        if 'series' not in inp or len(inp['series']) == 0:
            raise NameError('need a defination of calculation series')

        for item in inp['series'][::-1]:
            if 'skip' in item and item['skip']:
                continue
            self.series.append(self.gene_series(item))

    def gene_series(self, item):
        # sample holds all data for calculation
        sample = Sample(
            item['label'],
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
            ])

        # initialzed impurity Concentration
        sample.x_1 = np.float64(item['x_1'])

        # convergence condition
        sample.condition = np.float32(item['condition'])

        # chemical potential
        if len(item['delta_mu']) <= 1:
            sample.mu = np.array(item['delta_mu'], np.float64)
        else:
            sample.mu = np.linspace(item['delta_mu'][0], item['delta_mu'][1],
                                    item['delta_mu'][2])

        # Temperature
        if len(item['temp']) == 0:
            raise NameError('must set temperature')
        if len(item['temp']) == 1:
            sample.temp = np.array(item['temp'], np.single)
        elif len(item['temp']) == 3:
            sample.temp = np.linspace(item['temp'][0], item['temp'][1],
                                      item['temp'][2])
        elif len(item['temp']) == 4:
            raise NameError('can not use this function right now')
        else:
            raise NameError('temperature was configured with error format')
        sample.res['temp'] = sample.temp

        # =================================================
        # Interation energies with thermal vibration
        # by combined with Morse potential and Debye model
        # ==================================================
        datas = item['datas']
        xs = lc2ad(np.array(datas['lattice_c']))

        # transter
        int_ens = []
        transfer = item['transfer']
        pair_label = [n for n in self.gene_pair_label(datas)]
        if 'cut_pair' in datas:
            pair_label = pair_label[:-datas['cut_pair']]
        for n in pair_label:
            int_ens.append(self.gene_raw_int(datas[n], np.zeros(len(xs))))

        # =====================================>
        # int_ens = np.array(int_ens)

        # # add distortion energies
        # if 'distortion' in datas:
        #     for i, v in enumerate(datas['distortion']):
        #         print(v)
        #         distortion = np.float64(v) / self.conv
        #         int_ens[i] += distortion
        #         # print(int_ens[i])
        # <==========================================

        copy_int = np.array(int_ens)
        effective_int = np.array(sample.effctive_en(int_ens, *transfer))
        int_diffs = effective_int - copy_int

        # # add distortion energies
        # if 'distortion' in datas:
        #     for i, v in enumerate(datas['distortion']):
        #         distortion = np.float64(v) / self.conv
        #         pair = 'pair%s' % (i + 1)
        #         datas[pair][0]['energy'] = np.array(
        #             datas[pair][0]['energy']) + distortion + int_diffs[i]

        # 1st total energy
        distortion = np.full(len(xs), np.float64(datas['distortion'][0]))
        distortion /= self.conv
        datas['pair1'][0]['energy'] = np.array(
            datas['pair1'][0]['energy']) + int_diffs[0] + distortion

        distortion = np.full(len(xs), np.float64(datas['distortion'][1]))
        distortion /= self.conv
        datas['pair2'][0]['energy'] = np.array(
            datas['pair2'][0]['energy']) + int_diffs[1] + distortion

        host = np.array(datas['host_en']) * self.conv
        int_pair1 = cv.int_energy(
            xs,
            datas['pair1'],
            # datas['pair1'][0]['energy'] + int_diffs[0],
            host,
            self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)
        int_pair2 = cv.int_energy(
            xs,
            datas['pair2'],
            # datas['pair2'][0]['energy'] + int_diffs[1],
            host,
            self.bzc,
            num=6,
            conv=self.conv,
            noVib=False)
        int_trip = cv.int_energy(
            xs,
            datas['triple'],
            host,
            self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)
        int_tetra = cv.int_energy(
            xs,
            datas['tetra'],
            host,
            self.bzc,
            num=4,
            conv=self.conv,
            noVib=False)

        # Host with vibration
        # equilibrium lattice will evaluate from formula
        phase_ens_func = self.gene_phase_en_func(
            xs, host, (), *datas['tetra'], num=4)
        if 'no_imp_depen' in datas and datas['no_imp_depen'] is True:
            lattice_func = self.gene_lattice_func(
                phase_ens_func, noImpDepen=True)
        else:
            lattice_func = self.gene_lattice_func(phase_ens_func)

        def _gene_ints(T, c, only_r0=False):
            if 'fix_a0' in datas:
                r_0 = lc2ad(np.float64(datas['fix_a0']))
            else:
                r_0 = lattice_func(T, c)

            if not only_r0:
                pair1 = np.array(int_pair1(r_0, T), np.float64)
                pair2 = np.array(int_pair2(r_0, T), np.float64)
                trip = np.array(int_trip(r_0, T), np.float64)
                tetra = np.array(int_tetra(r_0, T), np.float64)
                return (pair1, pair2), trip, tetra

            return r_0, c

        sample.gene_ints = _gene_ints

        return sample

    def gene_lattice_func(self, formulas, **kwagrs):
        if 'bounds' in kwagrs:
            _bounds = kwagrs['bounds']
        else:
            _bounds = (lc2ad(7.0), lc2ad(9.0))

        if 'ratio' in kwagrs and len(kwagrs['ratio']) == len(formulas):
            _ratio = kwagrs['ratio']
        else:
            _ratio = [0, 0.25, 0.5, 0.75, 1]

        def _lattice_gene(T, c):
            _lattice_minimums = list()
            for formula in formulas:
                _lattice_min = minimize_scalar(
                    lambda r: formula(r, T), bounds=_bounds, method='bounded')
                _lattice_minimums.append(_lattice_min.x)

            _lattice_func = UnivariateSpline(
                _ratio, _lattice_minimums[::-1], k=2)
            return _lattice_func(0.0) if 'noImpDepen' in kwagrs and kwagrs['noImpDepen'] is True \
                                       else _lattice_func(c)

        return _lattice_gene

    def gene_phase_en_func(self, xs, host, acc, *datas, **kwagrs):
        if not datas:
            return acc

        data = datas[0]
        conv = self.conv
        num = kwagrs['num']
        bzc = self.bzc
        ys = np.array(data['energy'], np.float64) * conv / num
        mass = np.array(data['mass'], np.float64)
        _en_func = cv.free_energy(xs, ys, host, mass, bzc)
        return self.gene_phase_en_func(xs, host, acc + (_en_func, ),
                                       *datas[1:], **kwagrs)

    # get interaction energies
    @classmethod
    def gene_raw_int(cls, data, acc):
        if not data:
            return acc
        part = data[0]['coefficient'] * np.array(data[0]['energy'])
        return cls.gene_raw_int(data[1:], acc + part)

    @classmethod
    def gene_pair_label(cls, datas, start=1):
        while True:
            label = 'pair' + str(start)
            if label in datas:
                start += 1
                yield label
                continue
            break

    def run(self):
        raise NotImplementedError(
            'must implement this inherited abstract method')
