import numpy as np
import pandas as pd

from collections import defaultdict


class Normalizer():

    def __init__(self, energies, formulas, targets):
        if not isinstance(energies, pd.DataFrame):
            raise TypeError('energies must be <pd.DataFrame> but got %s' %
                            energies.__class__.__name__)

        _ints = []
        for f in formulas:
            tmp = 0
            for k, v in f.items():
                tmp += energies[k].values * v
            _ints.append(tmp)
        self._ints = np.asarray(_ints)

        self._diff = defaultdict(None)
        for k, v in targets.items():
            self._diff[k] = self._energy_diff(**v)

    def __item__(self, i):
        return self._diff[i]

    def _energy_diff(self, steps, ratios):
        """
        2nd parameter refer to the neighbor that transfer to
        """

        _int_diff = 0

        for step in steps:
            length = len(step)

            to = 1
            start = to + 1
            end = _int_diff.shape[0]
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
                    _int_diff += ratios[index] * self._ints[index] * percent / ratios[to - 1]

        return _int_diff
