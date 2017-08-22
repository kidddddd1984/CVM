#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
CVM
===

Provides
--------
combined cluster variation method with real spaces cluster expansion to
calculated solubility limit.
"""

import json
import yaml
import sys
import os
import tempfile
import datetime as dt
import re as regex

from .A1 import tetrahedron as T
from .A1 import tetraOctahedron as TO
from .A1 import tetraSquare as TS
from .A1 import doubleTetrahedron as DT
from .A1 import quadrupleTetrahedron as QT

pattern = regex.compile(r"(/\*)+.+?(\*/)", regex.S)  # remove comment in json


class CvmCalc(object):

    """
    Configuration for CVM Calculation.
    ==================================

    Initialaztion
    -------------
    We use a INCAR as our calculation by inner the program we trade it as json.
    You can write a INCAR in JSON and and saved it as UTF-8 and pass the file
    to program using '-inp FileName'. Also you can pass a dictionary object
    contained you input to the first paramater directly.

    The output will be saved as output.json by default.However you can set the
    output file's name and path in the INCAR.

    You can format output by yourself even if you want to stream it to database
    directly. Just pass a backend function to the second paramater.
    """

    __slots__ = (
        'backend',
        'workerpool',
        'output_json',  # format output to json
        'backends',  # post processes
        'method_dict',  # store calculation methods as {'name': method}
    )

    def __init__(self, **kwargs):
        """
        keep None when you don't want to custom by yourself.
        """
        super(CvmCalc, self).__init__()
        self.workerpool = []
        self.method_dict = dict(
            T=T,
            DT=DT,
            TO=TO,
            TS=TS,
            QT=QT,
        )

        # add cvm to path
        cwd = os.getcwd() + '/'
        cmd_folder = os.path.dirname(__file__)
        if cmd_folder not in sys.path:
            sys.path.append(cmd_folder)

        # parse flags
        self.output_json = True if kwargs['output_json'] else False
        self.backends = kwargs['backend'] if 'backend' in kwargs else None
        if 'inp' in kwargs:
            self.__run(kwargs['inp'])
        else:
            with open(cwd + kwargs['inp_card']) as f:
                _content = f.read()
                _content = pattern.sub('', _content)
            f = tempfile.TemporaryFile(mode='w+t')
            f.write(_content)
            f.seek(0)
            inp = json.load(f)
            f.close()
            self.__run(inp)

    def __run(self, inp):
        #  calculation run on separate thread
        if 'methods' not in inp:
            raise NameError('need specify  calculation methods')
        for method in inp['methods']:
            worker = self.method_dict[method](inp)
            worker.start()
            self.workerpool.append(worker)

        # wait all done
        for worker in self.workerpool:
            worker.join()

        # join results
        worker = self.workerpool[0]
        for other in self.workerpool[1:]:
            worker.output['results'].extend(other.output['results'])

        log_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(os.getcwd() + '/log/' + log_name)
        if not os.path.exists('log/'):
            os.makedirs('log/')

        if self.backends:
            for backend in self.backends:
                try:
                    __import__(backend[:-3]).process(worker.output)
                except ImportError as e:
                    raise e

        if self.output_json:
            with open(log_path + '.json', 'w') as f:
                json.dump(worker.output, f, indent=2)
        else:
            with open(log_path + '.yaml', 'w') as f:
                yaml.dump(worker.output, f, default_flow_style=False, indent=3)
