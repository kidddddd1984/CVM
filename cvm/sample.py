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
        'int',  # interaction energy
        'mu',  # Chemical potential
        'temp',  # Temperature (K)
        'transfer',
        'coord_num',
        'gene_ints',
        'res',  # result
    )

    def __init__(self, label, coord_num):
        super(Sample, self).__init__()
        self.label = label
        self.coord_num = np.array(coord_num)
        self.int = []
        self.res = {
            'label': label,
            'temp': [],
            'c': [],
            'inter_en': [],
        }

    def effctive_en(self, int_pair, *args):
        """
        2nd parameter refer to the neighbor that transfer to
        """
        if not args:
            return int_pair

        arg = args[0]
        length = len(arg)

        to = 1 if not arg[0] else arg[0]
        end = 0 if length < 2 else arg[1]
        start = to + 1 if length < 3 else arg[2]
        percent = 1 if length < 4 else arg[3]

        # prepare range
        length = len(int_pair[0])
        if start > length or end > length or start > end:
            raise IndexError('index error')
        if end == 0:
            _range = range(start - 1, length)
        else:
            _range = range(start - 1, end)

        # print(_range)
        _int = np.zeros(length)
        for index in _range:
            if index == to - 1:
                pass
            else:
                _int += self.coord_num[index] * int_pair[index] * percent /\
                    self.coord_num[to - 1]
                # print('approximation until %sth is: %s eV' %
                #       (index + 1, _int))
        int_pair[to - 1] += _int

        return self.effctive_en(int_pair, *args[1:])
