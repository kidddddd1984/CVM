#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class Sample(object):
    """
    Sample is a object which store a group of configuration data for calculation
    """

    __slots__ = (
        'label',  # label for a calculation sample
        'x_1',  # initialization of impurity concentration
        'condition',  # Convergence condition
        'int_pair',  # interaction energy
        'int_trip',  # interaction energy
        'int_tetra',  # interaction energy
        'mu',  # Chemical potential
        'temp',  # Temperature (K)
        'res',  # result
    )

    def __init__(self, label):
        super(Sample, self).__init__()
        self.label = label
        self.res = {
            'label': label,
            'temp': [],
            'c': [],
        }

    def effctive_en(self, to=1, end=0, start=2):
        """
        2nd parameter refer to the neighbour that transfer to
        """
        # coordination number
        _coord_num = np.array([
            12,
            6,
            24,
            12,
            24,
            8,
            48,
            6,
            24,  # 9th-a
            12,  # 9th-b
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
            6,   # 17th-b
            48,  # 18th-a
            24,  # 18th-b
            24,
            48  # 20th
        ])

        # prepare range
        length = len(self.int_pair)
        if start > length or end > length or start >= end:
            raise IndexError('index error')
        if end == 0:
            _range = range(start - 1, length)
        else:
            _range = range(start - 1, end)

        # calculation pair interaction
        _int = np.float64(self.int_pair[to - 1])
        # print(_range)
        for index in _range:
            if index == to - 1:
                pass
            else:
                _int += _coord_num[index] * self.int_pair[index] /\
                    _coord_num[to - 1]
                print('approximation until %sth is: %s eV' %
                      (index + 1, _int))
        self.int_pair[to - 1] = _int