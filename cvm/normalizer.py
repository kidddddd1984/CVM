# Copyright 2019 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from copy import deepcopy

from collections import defaultdict

__all__ = ['Normalizer']


class Normalizer(defaultdict):

    def __init__(self, energies, clusters, targets):
        super().__init__()
        if not isinstance(energies, pd.DataFrame):
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

        _ints = []
        for f in clusters:
            tmp = 0
            for k, v in f.items():
                tmp += energies[k].values * v
            _ints.append(tmp)
        self._ints = pd.DataFrame(data=_ints, columns=energies.index.tolist())

        self.targets = deepcopy(targets)
        for k, v in self.targets.items():
            self[k] = self._energy_diff(**v)

    def _energy_diff(self, steps, ratios):
        """
        2nd parameter refer to the neighbor that transfer to
        """

        _int_diff = 0

        for step in steps:
            length = len(step)

            to = 1
            start = to + 1
            end = self._ints.shape[0]
            percent = 1

            if length > 0:
                to = step[0]
            if length > 1:
                end = step[1]
            if length > 2:
                start = step[2]
            if length > 3:
                percent = step[3]

            # prepare range
            if start > self._ints.shape[0] or end > self._ints.shape[0] or start > end:
                raise IndexError('index error')
            _range = range(start - 1, end)

            # print(_range)
            for index in _range:
                if index == to - 1:
                    pass
                else:
                    _int_diff += ratios[index] * self._ints.values[index] * percent / ratios[to - 1]

        return _int_diff

    def __repr__(self):
        s1 = '  |-'
        header = [self.__class__.__name__ + ':']

        return f'\n{s1}'.join(header +
                              ['{}: {}'.format(k, v['steps']) for k, v in self.targets.items()])
