import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize_scalar

from .utils import UnitConvert as uc

__all__ = ['ClusterVibration']


class ClusterVibration(object):
    """
    Tools class for cluster vibration
    """

    def __init__(self,
                 *,
                 label: str,
                 xs: list,
                 ys: list,
                 mass: float,
                 num: int,
                 morse_paras_bounds: list = None):
        """Calculate phase energy using debye-sg model

        Parameters
        ----------
        label : str
            Label in chemical composition format.
        xs : list
            List of atomic distance.
        ys : list
            List of phase energies.
        mass : float
            Mixed mass.
        num : int
            Number of atoms.
        morse_paras_bounds : list, optional
            parameter bounds for fitting, by default ``None``.
            See also, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        Raises
        ------
        ValueError
            `xs` and `ys` must have same shape.
        """

        if not len(xs) == len(ys):
            raise ValueError('xs and ys must have same dim.')
        self.label = label
        self.mass, self.num = mass, num
        self._xs = self._check_input(xs)
        self._ys = self._check_input(ys) / num

        # parameter bounds for fitting
        # order: c1, c2, lmd, r0
        # bounds: ([low, low, low, low], [high, high, high, high])
        # doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        if morse_paras_bounds is not None:
            self._morse_paras_bounds = morse_paras_bounds
        else:
            self._morse_paras_bounds = ([0, 0, 0, xs[0]], [10, 10, 2, xs[-1]])

        # fit parameters
        self._paras = self._fit_paras()

        # calculate equilibrium constant
        poly_min = minimize_scalar(
            self.morse_potential, bounds=(self._xs[0], self._xs[-1]), method='bounded')
        self._lattic_cons = poly_min.x
        self._ground_en = poly_min.fun

    @staticmethod
    def _check_input(array):
        if isinstance(array, list):
            return np.array(array, dtype=np.float64)
        if isinstance(array, pd.Series):
            return array.values
        if isinstance(array, np.ndarray):
            return array
        raise TypeError('input must be a array with shape (n,)')

    @property
    def c1(self) -> float:
        """Morse potential parameter `C1`.

        Returns
        -------
        float
        """
        return self._paras['c1']

    @property
    def c2(self) -> float:
        """Morse potential parameter `C2`.

        Returns
        -------
        float
        """
        return self._paras['c2']

    @property
    def lmd(self) -> float:
        """Morse potential parameter `lambda`.

        Returns
        -------
        float
        """
        return self._paras['lmd']

    @property
    def r_0(self) -> float:
        """Atomic distance at equilibrium status.

        Returns
        -------
        float
        """
        return self._paras['r_0']

    @property
    def equilibrium_lattice_cons(self) -> float:
        """Equilibrium lattice constant.

        Returns
        -------
        float
        """
        return uc.ad2lc(self._lattic_cons)

    @property
    def x_0(self) -> float:
        return self._paras['x_0']

    @property
    def gamma_0(self) -> float:
        return self._paras['gamma_0']

    @property
    def bulk_modulus(self) -> float:
        """Estimated bulk modulus.

        Returns
        -------
        float
        """
        return self._paras['B_0']

    def morse_potential(self, r: float) -> float:
        """Morse potential

        Parameters
        ----------
        r : float
            Atomic distance

        Returns
        -------
        float
            Potential energy.
        """
        return self.c1 - 2 * self.c2 * np.exp(-self.lmd * (r - self.r_0)) + \
            self.c2 * np.exp(-2 * self.lmd * (r - self.r_0))

    def debye_temperature(self, r: [float, str] = 'local') -> float:
        """Debye temperature function.

        Parameters
        ----------
        r : float, str, optional
            Atomic distance, by default None

        Returns
        -------
        float
        """
        D_0 = np.float64(41.63516) * np.power(self.r_0 * self.bulk_modulus / self.mass, 1 / 2)
        if r == 'local':
            return D_0
        return D_0 * np.power(self.r_0 / r, 3 * self.lmd * r / 2)

    # debye function
    # default n=3
    def debye_function(self, T: float, r: float = None, *, n: int = 3) -> float:
        """Debye function.

        Parameters
        ----------
        T : float
            Temperature.
        r : float, optional
            Atomic distance, by default None
        n : int, optional
            Dimssion, by default 3

        Returns
        -------
        float
        """
        x = self.debye_temperature(r) / T

        ret, _ = quad(lambda t: t**n / (np.exp(t) - 1), 0, x)
        return (n / x**n) * ret

    def __call__(
            self,
            *,
            T: float = None,
            r: [float, str] = 'local',
            min_x: str = None,
    ):
        """Get free energy.

        Parameters
        ----------
        T : float, optional
            Temperature. If ``None``, will set parameter ``vibration`` to ``False`` automatically.
        r : float or str, optional
            Atomic distance, If ``local``, will ues the default setting which be set when instancing.
            by default ``None``.
        min_x: str, optional
            By default ``None``.
            If ``None``, will ues the default setting which be set when instancing.
            If not ``None``, function will returns equilibrium lattice constant as second result.
            The string can be ``ws`` or ``lattice``.

        """
        bzc = 8.6173303e-5

        if isinstance(T, (float, int)) and T < 0:
            raise RuntimeError('T must a positive number')

        if isinstance(r, (float, int)) and T < 0:
            raise RuntimeError('r must a positive number or str `local`')

        if isinstance(r, (str)) and r != 'local':
            raise RuntimeError('r must a positive number or str `local`')

        if T is None and r != 'local':
            return (self.morse_potential(r)) * self.num

        # no vibration effects
        if T is None and r == 'local':
            if min_x:
                if min_x == 'ws':
                    return self._ground_en * self.num, self._lattic_cons
                if min_x == 'lattice':
                    return self._ground_en * self.num, uc.ad2lc(self._lattic_cons)
                raise ValueError("min_x can only be 'ws' or 'lattice' but got %s" % min_x)
            return self._ground_en * self.num

        if T is not None and r != 'local':
            return (self.morse_potential(r) + \
                (9 / 8) * bzc * self.debye_temperature(r) - \
                bzc * T * (self.debye_function(T, r) - \
                3 * np.log(1 - np.exp(-(self.debye_temperature(r) / T))))) * self.num

        # take count into vibration effects
        if T is not None and r == 'local':
            poly_min = minimize_scalar(
                lambda _r: self(T=T, r=_r), bounds=(self._xs[0], self._xs[-1]), method='bounded')

            if min_x:
                if min_x == 'ws':
                    return poly_min.fun * self.num, poly_min.x
                if min_x == 'lattice':
                    return poly_min.fun * self.num, uc.ad2lc(poly_min.x)
                raise ValueError("min_x can only be 'ws' or 'lattice' but got %s" % min_x)
            return poly_min.fun

    def _fit_paras(self):
        # morse potential
        def morse_pot(r, c1, c2, lmd, r0):
            return c1 - 2 * c2 * np.exp(-lmd * (r - r0))\
                + c2 * np.exp(-2 * lmd * (r - r0))

        # morse parameters
        popt, _ = curve_fit(morse_pot, self._xs, self._ys, bounds=self._morse_paras_bounds)
        c1, c2, lmd, r0 = popt[0], popt[1], popt[2], popt[3]

        x0 = np.exp(-lmd * r0)
        B0 = uc.eV2Kbar(-(c2 * (lmd**3)) / (6 * np.pi * np.log(x0)))
        gamma_0 = lmd * r0 / 2

        # parameters will be used to construt
        # free energy with thermal vibration effect
        return dict(
            c1=c1,
            c2=c2,
            lmd=lmd,
            r_0=r0,
            x_0=x0,
            gamma_0=gamma_0,
            B_0=B0,
        )

    def __repr__(self):
        s1 = '  |-'
        header = [f'{self.label}:']

        return f'\n{s1}'.join(header + [
            'c1: {:f},  c2: {:f},  lambda: {:f}'.format(self.c1, self.c2, self.lmd), \
            'r0: {:f},  x0: {:f}'.format(self.r_0, self.x_0), \
            'Gruneisen constant: {:f}'.format(self.gamma_0), \
            'Equilibrium lattice constant: {:f} a.u.'.format(self.equilibrium_lattice_cons), \
            'Bulk Modulus: {:f} Kbar'.format(self.bulk_modulus), \
            'Debye temperature: {:f} K'.format(self.debye_temperature())
        ])
