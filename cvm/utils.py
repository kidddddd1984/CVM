#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
import json
import os
import re as regex
import sys
import tempfile
from pathlib import Path

import numpy as np

import ruamel.yaml


class UnitConvert:
    # lattice constan to atomic distance
    @staticmethod
    def lc2ad(d, n=4):
        return d * np.power((3 / (4 * n * np.pi)), 1 / 3)

    # atomic distance to lattice constan
    @staticmethod
    def ad2lc(d, n=4):
        return d / np.power((3 / (4 * n * np.pi)), 1 / 3)

    # a.u. press to Kbar
    @staticmethod
    def eV2Kbar(p):
        return p * 2.9421912e13 * 1e-8 / 27.21138505

    # a.u. temperature to K
    @staticmethod
    def au2K(t):
        return t * 3.1577464e5


def get_inp(path):
    # remove comment in json
    pattern = regex.compile(r"(/\*)+.+?(\*/)", regex.S)
    path = Path(path).expanduser().resolve()
    with open(str(path)) as f:
        _content = f.read()
        _content = pattern.sub('', _content)
    f = tempfile.TemporaryFile(mode='w+t')
    f.write(_content)
    f.seek(0)
    inp = json.load(f)
    f.close()
    return inp
