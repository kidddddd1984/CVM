#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def percent(x, pos=0):
    return u'{:3.1f}'.format(100 * x)


def process(opt):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)  # 800 * 500
    plt.subplots_adjust(top=0.97, bottom=0.14, left=0.12, right=0.97)

    # draw experiment
    y_exp = opt['experiment'][0]['temp']
    x_exp = opt['experiment'][0]['c']
    # ax.annotate('1200$K$', (0.005, 410), fontsize=13)
    ax.plot(
        x_exp, y_exp, 'x--', mew=3, mfc='w', ms=6, lw=1.5, label='Experiment')

    # theory
    # ax.plot(
    #     [0.0083, 0.019, 0.027, 0.035, .056], [800, 1200, 1400, 1600, 2050],
    #     'kx--', mew=3, mfc='w', ms=6, lw=1.5, label='Theory'
    # )

    # draw results
    line_type = ['o-', 'o--', 'o:']
    for i, res in enumerate(opt['results']):
        # plt.xlim(xmin=0.5, xmax=12.5)
        # plt.axhline(y=0, color='k', ls='-', lw=1.0)
        # label[i] = 'int= ' + '{:07.4}'.format(opt['Results'][i]['1st_int'])
        ax.plot(
            res['c'],
            res['temp'],
            # 'o-',
            line_type[i],
            color='darkorange',
            ms=4,
            lw=1.5,
            label=res['label'])

    # set formater
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    ax.set_xlim(0, 0.122)

    # for preview
    ax.grid(axis='y')
    ax.tick_params(labelsize=14.5)
    ax.set_ylabel(r'Temperature, $T$/K', size=16)
    ax.set_xlabel(
        r'Concentration of ' + opt['meta']['impurity'].capitalize() +
        r', $c$/at$\%$',
        size=16)
    ax.annotate(
        r'(b)CVMTO10, with thermal vibration effect', (0.003, 1830), size=15)
    ax.annotate(r'$T_\mathrm{FD}=1600$K', (0.004, 1630), size=17)
    ax.legend(loc='lower right', markerscale=1.2, fontsize=15)
    fig_name = "_".join(
        (opt['meta']['prefix'], opt['meta']['host'], opt['meta']['impurity'],
         opt['meta']['suffix'])).lstrip('_')
    plt.savefig(fig_name, dpi=600)  # 300 dpi
    plt.show()
