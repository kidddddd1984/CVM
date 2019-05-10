#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime as dt
import json
import os
import re as regex
import sys
import tempfile
from pathlib import Path
from ruamel.yaml import YAML
import pandas as pd
import numpy as np
import collections
import re

from scipy.stats.mstats import gmean, hmean

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

    # eV. press to Kbar
    @staticmethod
    def eV2Kbar(p):
        return p * 2.9421912e13 * 1e-8 / 27.21138505

    # a.u. temperature to K
    @staticmethod
    def au2K(t):
        return t * 3.1577464e5

    # a.u. temperature to K
    @staticmethod
    def ry2eV(t):
        return t * 13.605698066


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


def mixed_atomic_weight(formula: str, *, mean='arithmetic'):
    atomic_weight = {
        'H': 1.008,
        'He': 4.0026019999999995,
        'Li': 6.94,
        'Be': 9.0121831,
        'B': 10.81,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'F': 18.99840316,
        'Ne': 20.1797,
        'Na': 22.98976928,
        'Mg': 24.305,
        'Al': 26.9815385,
        'Si': 28.085,
        'P': 30.973762,
        'S': 32.06,
        'Cl': 35.45,
        'Ar': 39.948,
        'K': 39.0983,
        'Ca': 40.078,
        'Sc': 44.955908,
        'Ti': 47.867,
        'V': 50.9415,
        'Cr': 51.9961,
        'Mn': 54.938044,
        'Fe': 55.845,
        'Co': 58.93319399999999,
        'Ni': 58.6934,
        'Cu': 63.54600000000001,
        'Zn': 65.38,
        'Ga': 69.723,
        'Ge': 72.63,
        'As': 74.921595,
        'Se': 78.971,
        'Br': 79.904,
        'Kr': 83.79799999999999,
        'Rb': 85.4678,
        'Sr': 87.62,
        'Y': 88.90584,
        'Zr': 91.22399999999999,
        'Nb': 92.90637,
        'Mo': 95.95,
        'Tc': 97.90720999999999,
        'Ru': 101.07,
        'Rh': 102.9055,
        'Pd': 106.42,
        'Ag': 107.8682,
        'Cd': 112.414,
        'In': 114.818,
        'Sn': 118.71,
        'Sb': 121.76,
        'Te': 127.6,
        'I': 126.90446999999999,
        'Xe': 131.293,
        'Cs': 132.905452,
        'Ba': 137.327,
        'La': 138.90547,
        'Ce': 140.116,
        'Pr': 140.90766000000002,
        'Nd': 144.24200000000002,
        'Pm': 144.91276000000002,
        'Sm': 150.36,
        'Eu': 151.964,
        'Gd': 157.25,
        'Tb': 158.92535,
        'Dy': 162.5,
        'Ho': 164.93033,
        'Er': 167.25900000000001,
        'Tm': 168.93421999999998,
        'Yb': 173.045,
        'Lu': 174.9668,
        'Hf': 178.49,
        'Ta': 180.94788,
        'W': 183.84,
        'Re': 186.207,
        'Os': 190.23,
        'Ir': 192.217,
        'Pt': 195.084,
        'Au': 196.966569,
        'Hg': 200.592,
        'Tl': 204.38,
        'Pb': 207.2,
        'Bi': 208.9804,
        'Po': 209.0,
        'At': 210.0,
        'Rn': 222.0,
        'Fr': 223.0,
        'Ra': 226.0,
        'Ac': 227.0,
        'Th': 232.0377,
        'Pa': 231.03588,
        'U': 238.02891,
        'Np': 237.0,
        'Pu': 244.0,
        'Am': 243.0,
        'Cm': 247.0,
        'Bk': 247.0,
        'Cf': 251.0,
        'Es': 252.0,
        'Fm': 257.0,
        'Md': 258.0,
        'No': 259.0,
        'Lr': 262.0,
        'Rf': 267.0,
        'Db': 268.0,
        'Sg': 271.0,
        'Bh': 274.0,
        'Hs': 269.0,
        'Mt': 276.0,
        'Ds': 281.0,
        'Rg': 281.0,
        'Cn': 285.0,
        'Nh': 286.0,
        'Fl': 289.0,
        'Mc': 288.0,
        'Lv': 293.0,
        'Ts': 294.0,
        'Og': 294.0
    }

    weights = []
    num = 0
    for k, v in parse_formula(formula).items():
        weights += [atomic_weight[k]] * int(v)
        num += int(v)

    if mean is 'arithmetic':
        return np.mean(weights), num

    if mean is 'harmonic':
        return hmean(weights), num

    if mean is 'geometric':
        return gmean(weights), num


def parse_input_set(path_of_set, *, is_ry_unit=True, is_lattice_unit=True):
    path = Path(path_of_set).expanduser().resolve()
    if not path.is_dir() or not (path / 'input.yml').exists():
        raise RuntimeError('can not parse input set')
    yaml = YAML()
    with open(str(path / 'input.yml'), 'r') as f:
        inp = yaml.load(f)

    if 'meta' not in inp:
        raise RuntimeError('can not find an entry named meta')

    if 'series' in inp:
        for s in inp['series']:
            ens = pd.read_csv(path / s['energies'], index_col=s['lattice'])
            del s['lattice']
            if is_ry_unit:
                ens = ens * 13.605698066
            if is_lattice_unit:
                ens.index = UnitConvert.lc2ad(ens.index)
            s['energies'] = ens

            if 'normalizer' in s:
                ens = pd.read_csv(path / s['normalizer']['energies'])
                if is_ry_unit:
                    ens = ens * 13.605698066
                s['normalizer']['energies'] = ens

    return inp


def parse_formula(formula):
    """
    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3

    Returns:
        Composition with that formula.

    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")

    def get_sym_dict(f, factor):
        sym_dict = collections.defaultdict(float)
        for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
            el = m.group(1)
            amt = 1
            if m.group(2).strip() != "":
                amt = float(m.group(2))
            sym_dict[el] += amt * factor
            f = f.replace(m.group(), "", 1)
        if f.strip():
            raise RuntimeError("{} is an invalid formula!".format(f))
        return sym_dict

    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt) for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)
