# CVM

### Provides

CVM is a python package for solution limit calculation using **Cluster Variation Method** (CVM).

## Usage

cvm is a python module which can run in a python3 environment.
The simplest way to run a calculation can do something like this:

```bash
$ python -m cvm -h  # for help
$ python -m cvm <your input_card> [-b <some post process> -o]
```

`input_card` is necessary for a calculation. you can see the `*.json` in the root dir.
`-b/--backend` is optional arguments to do some post process with results. also see `backend.py` under this dir.

All the calculated results will be stored in `log` dir by the created date.
