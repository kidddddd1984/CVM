#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


# lattice constan to atomic distance
def lc2ad(d, n=4):
    return d * np.power((3 / (4 * n * np.pi)), 1 / 3)


# atomic distance to lattice constan
def ad2lc(d, n=4):
    return d / np.power((3 / (4 * n * np.pi)), 1 / 3)


# a.u. press to Kbar
def eV2Kbar(p):
    return p * 2.9421912e13 * 1e-8 / 27.21138505


# a.u. temperature to K
def au2K(t):
    return t * 3.1577464e5
