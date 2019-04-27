import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import UnivariateSpline

from .utils import UnitConvert as uc


class ClusterVibration(object):
    """
    Tools class for cluster vibration
    """

    @classmethod
    def minimum(cls, xs, ys, **kwagrs):
        """
        xs: array
            atomic distance between nearest-neighbor
        ys: array
            energies change for ys, impuity cluster and correct respectivly
        """

        # get minimum from a polynomial
        poly_min = minimize_scalar(
            UnivariateSpline(xs, ys, k=4), bounds=(xs[0], xs[-1]), method='bounded')
        min_x = poly_min.x  # equilibrium atomic distance
        min_y = np.float(poly_min.fun)  # ground status energy

        return min_x, min_y

    @classmethod
    def free_energy(cls, xs, cluster, host, mass, bzc, correct=None, *, noVib=False):
        """
        xs: array
            atomic distance between nearest-neighbor
        correc, thost, cluster: array
            total energy of host
            energy different of impuity cluster
            correct between band calculation and impurity calculation
        mass: float
            atomic mass
        """

        # perpare energies and get polynomial minimum for morse fit
        ys = cluster + host if not correct else cluster + host - correct
        _, min_y = cls.minimum(xs, ys)

        # use polynomial minimum to obtain morse parameters
        ret = cls.fit_parameters(xs, ys - min_y, mass)
        morse = ret['morse']  # Morse potential based on 0 minimum
        D = ret['debye_func']  # Debye function
        theta_D = ret['debye_temperature_func']  # Debye temperature function

        if noVib:
            return lambda r, T: morse(r) + min_y

        # construct vibration withed energy formula
        return lambda r, T: morse(r) + min_y + (9 / 8) * bzc * theta_D(r)\
            - bzc * T * \
            (D(r, T) - 3 * np.log(1 - np.exp(-(theta_D(r) / T))))

    @classmethod
    def int_energy(cls, xs, datas, host, bzc, num, conv, *, noVib=False):
        """
        generate interaction energy
        """
        parts = []
        for data in datas:
            coeff = np.int(data['coefficient'])
            mass = np.float64(data['mass'])
            ys = np.array(data['energy'], np.float64) * conv / num
            part = cls.free_energy(xs, ys, host, mass, bzc, noVib=noVib)
            parts.append((coeff, part))

        def __int(r, T):
            int_en = np.float64(0)
            for part in parts:
                int_en += part[0] * part[1](r, T)
            return int_en * num

        return __int

    @classmethod
    def fit_parameters(cls, xdata, ydata, M, bounds=None):
        # generate morse potential
        def __morse_gene(xs, ys, bounds):

            def morse_mod(r, c1, c2, lmd, r0):
                return c1 - 2 * c2 * np.exp(-lmd * (r - r0))\
                    + c2 * np.exp(-2 * lmd * (r - r0))

            # morse parameters
            if not bounds:
                bounds = ([3, 3, 1, xs[0]], [8.5, 8.5, 2, xs[-1]])
            popt, _ = curve_fit(morse_mod, xs, ys, bounds=bounds)

            return popt[0], popt[1], popt[2], popt[3],\
                lambda r: morse_mod(r, popt[0], popt[1], popt[2], popt[3])

        # debye function
        # default n=3
        def __debye_func(x, n=3):
            try:
                from scipy.integrate import quad
            except Exception as e:
                raise e

            ret, _ = quad(lambda t: t**n / (np.exp(t) - 1), 0, x)
            return (n / x**n) * ret

        # generate debye temperature Θ_D
        def __debye_temp_gene(r0, lmd, B, M):  # (ΘD)0 r=r0
            D_0 = np.float64(41.63516) * np.power(r0 * B / M, 1 / 2)
            return D_0, lambda r: D_0 * np.power(r0 / r, 3 * lmd * r / 2)

        c1, c2, lmd, r0, morse_potential = __morse_gene(xdata, ydata, bounds)

        x0 = np.exp(-lmd * r0)
        B_0 = -(c2 * (lmd**3)) / (6 * np.pi * np.log(x0))
        gamma_0 = lmd * r0 / 2
        debye_temp_0, debye_temp_func = __debye_temp_gene(r0, lmd, uc.eV2Kbar(B_0), np.float(M))

        # parameters will be used to construt
        # free energy with thermal vibration effect
        return dict(
            c1=c1,
            c2=c2,
            lmd=lmd,
            r0=r0,
            x0=x0,
            gamma_0=gamma_0,
            equilibrium_lattice_constant=uc.ad2lc(r0),
            morse=morse_potential,
            bulk_moduli=uc.eV2Kbar(B_0),
            debye_temperature_0=debye_temp_0,
            debye_temperature_func=debye_temp_func,
            debye_func=lambda r, T: __debye_func(debye_temp_func(r) / T),
            debye_func_0=lambda T: __debye_func(debye_temp_0 / T),
        )

    @classmethod
    def show_parameter(cls, ret):
        c1 = ret['c1']
        c2 = ret['c2']
        lmd = ret['lmd']
        r0 = ret['r0']
        x0 = ret['x0']
        gamma_0 = ret['gamma_0']
        bulk_moduli = ret['bulk_moduli']
        debye_temp_0 = ret['debye_temperature_0']

        print("c1: {:f},  c2: {:f},  lambda: {:f}".format(c1, c2, lmd))
        print("r0: {:f},  x0: {:f}".format(r0, x0))
        print("Gruneisen constant: {:f}".format(gamma_0))
        print("Equilibrium lattice constant: {:f} a.u.".format(uc.ad2lc(r0)))
        print("Bulk Modulus: {:f} Kbar".format(bulk_moduli))
        print("Debye temperature: {:f} K\n\n".format(debye_temp_0))
        print("")
