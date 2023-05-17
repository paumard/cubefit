#!/usr/bin/env python3
#    Copyright (C) 2023  Thibaut Paumard <thibaut.paumard@obspm.fr>
#            Julien Brulé
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Single-line spectral models

Each profile has the following signature:
  ydata, jacobian = profile(xdata, *parameters)

Wrappers are provided to use these profiles with
scipy.optimize.curve_fit (WrapToCurveFit) and to use functions
designed for curve fit with cubefit (WrapFromCurveFit), as well as to
use astropy.modeling profiles (WrapFromAstropy).
"""

import numpy as np
from math import sqrt
# import matplotlib.pyplot as plt

# from scipy import optimize

# for plotting
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename


class LineProfile:
    """
    One dimensional profile model.

    Parameters
    ----------
    x : array_like
        Independent variable, for instance wavelengths for which the
        profile must be computed.
    a : array_like
        Parameters of the Gaussian: (I0, x0, dx [, offset [, slope]]) with:

    See Also
    --------
    Gaussian, NGaussian, Moffat

    """
    def __init__(self, x, *a):
        self.x = x
        self.a = a

    def __str__(self):
        print("LineProfile:")
        print(f"\tprofile parameters: {self.a}")
        print(f"\ton domain min: {self.x[0]} max: {self.x[-1]}")


# DONE ajout boolean switch deriv if deriv return res, grad ?
# def gauss(x, *a, deriv):
def gauss(x, *a):
    """Compute a Gaussian profile.

    Parameters
    ----------
    x : array_like
        Independent variable, for instance wavelengths for which the
        profile must be computed.
    a : array_like
        Parameters of the Gaussian: (I0, x0, dx [, offset [, slope]]) with:
        I0: peak value
        x0: center
        dx: Gaussian standard deviation
        offset (optional): constant offset
        slope (optional): linear offset

    Returns
    -------
    ydata : array_like
        The values of the Gaussian with paramaters a computed at
        x. Same shape as x.
    jac : array like
        The Jacobian matrix of the model, with shape x.size × a.size
        (if x is a 1D array) or a.size (if x is a scalar).

    Notes
    -----
    Returns a Gaussian:
        I0*exp(-0.5*((x-x0)/dx)**2) [+a[3] [+a[4]*x]]
    Where:
        I0=a[0]
        x0=a[1]
        dx=a[2] (gaussian sigma)

    FHWM=sigma*2*sqrt(2*alog(2)); sum(gauss)=I0*sigma*sqrt(2*pi)

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 1, 0.5, 0.5, 0.1])
    >>> x = np.linspace(-10, 10, 5)
    >>> ret, ret_jac = gauss(x, *a)
    >>> print(f"{ret}")
    [-0.5         0.          0.63533528  1.          1.5       ]
    >>> print(f"{ret_jac}")
    [[ 7.95674389e-106 -3.50096731e-104  7.70212809e-103  1.00000000e+000
      -1.00000000e+001]
     [ 5.38018616e-032 -1.29124468e-030  1.54949361e-029  1.00000000e+000
      -5.00000000e+000]
     [ 1.35335283e-001 -5.41341133e-001  1.08268227e+000  1.00000000e+000
       0.00000000e+000]
     [ 1.26641655e-014  2.02626649e-013  1.62101319e-012  1.00000000e+000
       5.00000000e+000]
     [ 4.40853133e-071  1.58707128e-069  2.85672830e-068  1.00000000e+000
       1.00000000e+001]]

    # >>> import matplotlib.pyplot as plt
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret)
    # [...
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret_jac)
    # [...
    # >>> plt.show()
    # >>> plt.close()


    See Also
    --------
    cubefit.lineprofiles.ngauss

    """
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    nterms = a.size

    if nterms < 3:
        raise Exception("gauss() needs at least 3 parameters")

    eps = 1e-100
    # a[2] = np.clip(a[2], -eps, eps)
    if (abs(a[2]) < eps):
        a[2] = np.sign(a[2])*eps

    if (a[2] == 0):
        u1 = 0
    else:
        u1 = np.exp(-0.5*((x-a[1])/a[2])**2)

    # u1 = np.exp(-0.5 * ((x - a[1]) / a[2])**2)

    res = a[0] * u1

    grad = np.zeros(np.shape(x) + (nterms,))
    # grad = np.zeros((*np.shape(x), nterms))
    grad[..., 0] = u1
    grad[..., 1] = res * (x - a[1]) / a[2]**2
    grad[..., 2] = res * (x - a[1])**2 / a[2]**3
    if nterms > 3:
        grad[..., 3] = 1.
    if nterms == 5:
        grad[..., 4] = x

    if nterms > 3:
        res += a[3]

    if nterms == 5:
        res += a[4] * x

    return res, grad


# DONE ajout boolean switch deriv if deriv return res, grad ?
# def gauss(x, *a, deriv):
def old_gauss(x, *a):
    """Compute a Gaussian profile.

    Parameters
    ----------
    x : array_like
        Independent variable, for instance wavelengths for which the
        profile must be computed.
    a : array_like
        Parameters of the Gaussian: (I0, x0, dx [, offset [, slope]]) with:
        I0: peak value
        x0: center
        dx: Gaussian standard deviation
        offset (optional): constant offset
        slope (optional): linear offset

    Returns
    -------
    ydata : array_like
        The values of the Gaussian with paramaters a computed at
        x. Same shape as x.
    jac : array like
        The Jacobian matrix of the model, with shape x.size × a.size
        (if x is a 1D array) or a.size (if x is a scalar).

    Notes
    -----
    Returns a Gaussian:
        I0*exp(-0.5*((x-x0)/dx)**2) [+a[3] [+a[4]*x]]
    Where:
        I0=a[0]
        x0=a[1]
        dx=a[2] (gaussian sigma)

    FHWM=sigma*2*sqrt(2*alog(2)); sum(gauss)=I0*sigma*sqrt(2*pi)

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 1, 0.5, 0.5, 0.1])
    >>> x = np.linspace(-10, 10, 5)
    >>> ret, ret_jac = gauss(x, *a)
    >>> print(f"{ret}")
    [-0.5         0.          0.63533528  1.          1.5       ]
    >>> print(f"{ret_jac}")
    [[ 7.95674389e-106 -3.50096731e-104  7.70212809e-103  1.00000000e+000
      -1.00000000e+001]
     [ 5.38018616e-032 -1.29124468e-030  1.54949361e-029  1.00000000e+000
      -5.00000000e+000]
     [ 1.35335283e-001 -5.41341133e-001  1.08268227e+000  1.00000000e+000
       0.00000000e+000]
     [ 1.26641655e-014  2.02626649e-013  1.62101319e-012  1.00000000e+000
       5.00000000e+000]
     [ 4.40853133e-071  1.58707128e-069  2.85672830e-068  1.00000000e+000
       1.00000000e+001]]

    # >>> import matplotlib.pyplot as plt
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret)
    # [...
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret_jac)
    # [...
    # >>> plt.show()
    # >>> plt.close()


    See Also
    --------
    cubefit.lineprofiles.ngauss

    """
    # ensure a and x are numpy arrays and not some other array_like
    # promote to at least float64
    a = np.asarray(a, dtype=np.float64)
    # a = np.promote_types(a.dtype, np.float64).type(a)
    x = np.asarray(x, dtype=np.float64)
    # x = np.promote_types(x.dtype, np.float64).type(x)

    nterms = a.size

    if nterms < 3:
        raise Exception("gauss() needs at least 3 parameters")

    eps = 1e-100
    if (abs(a[2]) < eps):
        a[2] = np.sign(a[2])*eps

    if (a[2] == 0):
        u1 = 0
    else:
        u1 = np.exp(-0.5*((x-a[1])/a[2])**2)

    res = a[0]*u1

    grad = np.zeros(np.shape(x) + (nterms,))
    grad[..., 0] = u1
    grad[..., 1] = res*(x-a[1])/a[2]**2
    grad[..., 2] = res*(x-a[1])**2/a[2]**3
    if (nterms > 3):
        grad[..., 3] = 1.
    if (nterms == 5):
        grad[..., 4] = x

    if (nterms > 3):
        res = res+a[3]

    if (nterms == 5):
        res = res+a[4]*x

    return res, grad


def ngauss(x, *a):
    """Compute a normalized Gaussian profile.

    Parameters
    ----------
    x : array_like
        Independent variable, for instance wavelengths for which the
        profile must be computed.
    a : array_like
        Parameters of the Gaussian: (Sum, x0, dx [, offset [, slope]]) with:
        Sum: integral of the Gaussian
        x0: center
        dx: Gaussian standard deviation
        offset (optional): constant offset
        slope (optional): linear offset

    Returns
    -------
    ydata : array_like
        The values of the Gaussian with paramaters a computed at
        x. Same shape as x.
    jac : array like
        The Jacobian matrix of the model, with shape x.size × a.size
        (if x is a 1D array) or a.size (if x is a scalar).

    Notes
    -----
    Returns a normalised Gaussian:
     I0*exp(-0.5*((x-x0)/dx)^2) [+a[3] [+a[4]*x]]
    Where:
    Sum=a[0]
    x0=a[1]
    dx=a[2] (gaussian sigma)
    I0=Sum/(sigma*sqrt(2*pi))

    In contrast with gauss(), the first parameter is the integral of
    the Gaussian rather than the peak value.

    FHWM=sigma*2*sqrt(2*alog(2)); sum(gauss)=I0*sigma*sqrt(2*pi)

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 1, 0.5, 0.5, 0.1])
    >>> x = np.linspace(-10, 10, 5)
    >>> ret, ret_jac = ngauss(x, *a)
    >>> print(f"{ret}")
    [-0.5         0.          0.60798193  1.          1.5       ]
    >>> print(f"{ret_jac}")
    [[ 6.34856311e-106 -2.79336777e-104  6.13271196e-103  1.00000000e+000
      -1.00000000e+001]
     [ 4.29276747e-032 -1.03026419e-030  1.22773150e-029  1.00000000e+000
      -5.00000000e+000]
     [ 1.07981933e-001 -4.31927732e-001  6.47891598e-001  1.00000000e+000
       0.00000000e+000]
     [ 1.01045422e-014  1.61672675e-013  1.27317231e-012  1.00000000e+000
       5.00000000e+000]
     [ 3.51749909e-071  1.26629967e-069  2.27230441e-068  1.00000000e+000
       1.00000000e+001]]

    # >>> import matplotlib.pyplot as plt
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret)
    # [...
    # >>> plt.figure()
    # <...
    # >>> plt.plot(x, ret_jac)
    # [...
    # >>> plt.show()
    # >>> plt.close()


    See also
    --------
    cubefit.lineprofiles.gauss
    """

    # ensure a and x are numpy arrays and not some other array_like
    # promote to at least float64
    a = np.asarray(a)
    a = np.promote_types(a.dtype, np.float64).type(a)
    x = np.asarray(x)
    x = np.promote_types(x.dtype, np.float64).type(x)

    sqrt2pim1 = 1/sqrt(2*np.pi)

    eps = 1e-100

    if (abs(a[2]) < eps):
        a[2] = np.sign(a[2])*eps
    if (a[2] == 0):
        raise Exception("BUG: a[2] == 0.")

    sigmam1 = 1./a[2]
    eqwidthm1 = sigmam1*sqrt2pim1
    I0 = a[0]*eqwidthm1
    a[0] = I0

    res, grad = gauss(x, *a)

    grad[..., 2] -= I0*sigmam1*grad[..., 0]
    grad[..., 0] *= eqwidthm1

    return res, grad


def moffat(x, *a):
    """Compute a Moffat profile.

    Parameters
    ----------
    x : array_like
        Independent variable, for instance wavelengths for which the
        profile must be computed.
    a : array_like
        Parameters of the Moffat: (I0, x0, dx [, offset [, slope]]) with:
        I0 : peak value
        x0 : center
        dx : Gaussian standard deviation
        offset (optional) : constant offset
        slope (optional) : linear offset

    Returns
    -------
    ydata : array_like
        The values of the Moffat with paramaters a computed at
        x. Same shape as x.
    jac : array like
        The Jacobian matrix of the model, with shape x.size × a.size
        (if x is a 1D array) or a.size (if x is a scalar).

    Notes
    -----
    Returns a Moffat:
        I=I0*(1+((x-x0)/dx)^2)^-b [+ k0 [+ k1*x]]
    Where:
        I0=a[0]
        x0=a[1] deriv: 2*I0*b*(x-x0)/(dx^2)*(1+((x-x0)/dx)^2)^(-b-1)
        dx=a[2] deriv: 2*I0*b*(x-x0)^2/(dx^3)*(1+((x-x0)/dx)^2)^(-b-1)
        b=a[3]

        and if a is of length 4 or 5:
        k0=a[4]
        k1=a[5]

    Limitation: "b"  should  always  be  positive.  In  order  to  force  it,
    especially in fitting routines, its  abolute value is taken (except at some
    point in the computation of derivates).

    Examples
    --------
    >>> a = np.array([1, 1, 0.5, 0.5, 0.1])
    >>> x = np.linspace(-10, 10, 10)
    >>> ret, ret_jac = moffat(x, *a)
    >>> print(f"{ret}")
    [0.14540766 0.15686984 0.1760503  0.21462411 0.33046638 1.07618706
     0.30952909 0.20910093 0.17357057 0.15547002]

    See Also
    --------
    cubefit.lineprofiles.ngauss, asmoffat
    """
    __moffat_betamax = None
    __moffat_vmax = None
    __moffat_gradmax = None
    a = np.asarray(a)
    if __moffat_gradmax is None:
        __moffat_gradmax = 1e150
    if __moffat_betamax is not None:
        big = __moffat_betamax
    else:
        big = 1e18

    a = np.asarray(a)
    x = np.asarray(x)
    nterms = a.size
    if __moffat_vmax:
        if abs(a[1]) > __moffat_vmax:
            grad = np.zeros((x.size, nterms))
            return np.zeros_like(x)
    small = 1e-80
    if a[2] < small:
        a[2] = small
    u2 = (x - a[1]) / a[2]
    ind = np.where(abs(u2) > __moffat_gradmax)
    if len(ind[0]) > 0:
        u2[ind] = np.sign(u2[ind]) * __moffat_gradmax
        print("*** 1 Warning: MOFFAT caught overflows.")
    u4 = u2 ** 2
    u3 = 1 + u4
    if abs(a[3]) > big:
        u1 = np.zeros_like(x)
        u1b = u1
        ind = np.where(u4 == 0)
        if len(ind[0]) > 0:
            u1[ind] = 1
            u1b[ind] = 1
    else:
        u1 = u3 ** -abs(a[3])
        u1b = u3 ** (-abs(a[3]) - 1)
    res = a[0] * u1
    if nterms > 4:
        res += a[4]
    if nterms == 6:
        res += a[5] * x
    # TODO freestyle
    tb = (x >= a[1])
    grad = np.zeros((x.size, nterms))
    grad[:, 0] = u1
    if np.max(u1b):
        grad[:, 1] = 2 * a[0] * a[3] * u2 / a[2] * u1b
    if np.max(u1b):
        grad[:, 2] = 2 * a[0] * a[3] * u4 / a[2] * u1b
    grad[:, 3] = -a[0] * np.log(u3) * u1
    if nterms > 4:
        #    grad[:, 4] = 0
        # TODO freestyle
        if np.max(u1b):
            grad[:, 4] = 2 * a[0] * (a[3] * u4 / a[4] * u1b * tb)
    if nterms == 6:
        grad[:, 5] = x
    ind1 = np.where(grad > __moffat_gradmax)
    if len(ind1[0]) > 0:
        grad[ind1] = __moffat_gradmax
        print("*** 2 Warning: MOFFAT caught overflows.")
    ind2 = np.where(grad < -__moffat_gradmax)
    if len(ind2[0]) > 0:
        grad[ind2] = -__moffat_gradmax
        print("*** 3 Warning: MOFFAT caught overflows.")
    ind3 = np.where(abs(grad) < 1 / __moffat_gradmax)
    if len(ind3[0]) > 0:
        grad[ind3] = 0
        # print("MOFFAT warning: grad underflows caught, grad is inaccurate.")
    return res, grad


def asmoffat(x, *a):
    """Compute an asymmetrical Moffat profile

    Compute a (1D) asymmetrical Moffat profile:
    I=I0*(1+((x-x0)/dx)^2)^-b [+ k0 [+ k1*x]]

    Where:
    I0 = a[0]
    x0 = a[1]  deriv: 2*I0*b*(x-x0)/(dx^2)*(1+((x-x0)/dx)^2)^(-b-1)
    dx for x<x0 = a[2] deriv: 2*I0*b*(x-x0)^2/(dx^3)*(1+((x-x0)/dx)^2)^(-b-1)
    b for x<x0  = a[3]
    dx for x>=x0 = a[4] deriv: 2*I0*b*(x-x0)^2/(dx^3)*(1+((x-x0)/dx)^2)^(-b-1)
    b for x>=x0 = a[5]

    and if a is of length 7 or 8:
    k0=a[6]
    k1=a[7]

    Returns
    -------
    ydata : array_like
        The values of the asymmetrical Moffat with paramaters a computed at
        x. Same shape as x.
    jac : array like
        The Jacobian matrix of the model, with shape x.size × a.size
        (if x is a 1D array) or a.size (if x is a scalar).

    Examples
    --------
    >>> na = np.array([1, 1, 0.5, 0.5, 0.1, 0.5])
    >>> x=np.arange(-50,50,0.3)
    >>> nx = np.linspace(-10, 10, 10)
    >>> nret, nret_jac = asmoffat(nx, *na)
    ASMOFFAT warning: grad underflows caught, grad is inaccurate.
    >>> print(f"{nret}")
    [0.04540766 0.05686984 0.0760503  0.11462411 0.23046638 0.66896473
     0.04281784 0.02194593 0.01475249 0.01111043]

    See Also
    --------
    moffat
    """
    a = np.asarray(a)
    a = np.promote_types(a.dtype, np.float64).type(a)
    x = np.asarray(x)
    x = np.promote_types(x.dtype, np.float64).type(x)

    __moffat_betamax = np.inf
    __moffat_vmax = np.inf
    __moffat_gradmax = 1e150
    if __moffat_betamax:
        big = __moffat_betamax
    else:
        big = 1e18
    nterms = a.size
    if __moffat_vmax and np.abs(a[1]) > __moffat_vmax:
        grad = np.zeros((x.size, nterms))
        return np.zeros_like(x)
    small = 1e-80
    if a[2] < small:
        a[2] = small
    if a[4] < small:
        a[4] = small
    ta = (x < a[1])
    tb = (x >= a[1])
    u2a = (x - a[1]) / a[2]
    u2b = (x - a[1]) / a[4]
    u4a = u2a ** 2
    u4b = u2b ** 2
    u3a = 1 + u4a
    u3b = 1 + u4b
    # TODO u3 not used ?
    # u3 = u3a * ta + u3b * tb
    if np.abs(a[3]) > big:
        u1a = np.zeros_like(x)
        u1ab = u1a
        ind = np.where(u4a == 0)[0]
        if ind.size > 0:
            u1a[ind] = 1
            u1ab[ind] = 1
    else:
        u1a = u3a ** -np.abs(a[3])
        u1ab = u3a ** (-abs(a[3]) - 1)

    if np.abs(a[5]) > big:
        u1b = np.zeros_like(x)
        u1bb = u1b
        ind = np.where(u4b == 0)[0]
        if ind.size > 0:
            u1b[ind] = 1
            u1bb[ind] = 1
    else:
        u1b = u3b ** -np.abs(a[5])
        u1bb = u3b ** (-np.abs(a[5]) - 1)
    u1 = u1a * ta + u1b * tb
    u1B = u1ab * ta + u1bb * tb
    res = a[0] * u1
    if nterms > 6:
        res += a[6]
    if nterms == 8:
        res += a[7] * x

    grad = np.zeros((x.size, nterms))
    grad[:, 0] = u1
    if np.max(u1B):
        grad[:, 1] = 2 * a[0] * (
            a[3] * u2a / a[2] * u1ab * ta + a[5] * u2b / a[4] * u1bb * tb
            )
    if np.max(u1ab):
        grad[:, 2] = 2 * a[0] * (a[3] * u4a / a[2] * u1ab * ta)
    grad[:, 3] = -a[0] * np.log(u3a)*u1a*ta
    if np.max(u1bb):
        grad[:, 4] = 2*a[0]*(a[5]*u4b/a[4]*u1bb*tb)
    grad[:, 5] = -a[0]*np.log(u3b)*u1b*tb
    # Useless line due to initialisation:
    # if (nterms>6) grad(,7)=0;
    if (nterms == 8):
        grad[:, 7] = x
    # try to avoid overflows in curve_fit.
    ind1 = np.where(grad > __moffat_gradmax)
    if (np.size(ind1) > 0):
        grad[ind1] = __moffat_gradmax
        print("ASMOFFAT warning: grad overflows caught, grad is inaccurate.")

    ind2 = np.where(grad < -__moffat_gradmax)
    if (np.size(ind2) > 0):
        grad[ind2] = -__moffat_gradmax
        print("ASMOFFAT warning: grad overflows caught, grad is inaccurate.")
    # try to avoid underflows in curve_fit.
    ind3 = np.where(np.abs(grad) < 1/__moffat_gradmax)
    if (np.size(ind3) > 0):
        grad[ind3] = 0
        print("ASMOFFAT warning: grad underflows caught, grad is inaccurate.")

    return res, grad


class WrapToCurveFit:
    """Wrap a cubefit profile to a curve_fit objective function

    Parameters
    ----------
    profile : callable
        A curvefit.lineprofiles profile.
    """
    def __init__(self, profile):
        self.profile = profile

    def __call__(self, xdata, *params):
        """self(xdata, *params) -> self.profile(xdata, *params)[0]
        """
        return self.profile(xdata, *params)[0]

    def jac(self, xdata, *params):
        """self.jac(xdata, *params) -> self.profile(xdata, *params)[1]
        """
        return self.profile(xdata, *params)[1]


class WrapFromCurveFit:
    """Wrap a curve_fit objective function as a cubefit profile

    Parameters
    ----------
    f : callable
        A scipy.iptimize.curve_fit objective function.
    jac : callable, optional
        The Jacobian of f. If None, estimated numerically.
    epsilon : array_like, optional
        Passed to numerical_jacobian.

    See Also
    --------
    scipy.optimize.curve_fit
    cubefit.lineprofiles.numerical_jacobian
    """
    def __init__(self, f, jac=None, epsilon=1e-6):
        self.f = f
        self.jac = jac
        self.epsilon = epsilon

    def __call__(self, xdata, *params):
        """ self(xdata, *params) ->
                             (self.f(xdata, *params), self.jac(xdata, *params))
        """
        val = self.f(xdata, *params)
        if self.jac is None:
            jac = numerical_jacobian(self.f, xdata, *params,
                                     epsilon=self.epsilon)
        else:
            jac = self.jac(xdata, *params)
        return val, jac


class WrapFromAstropy:
    """Wrap an astropy.modeling objective function as a cubefit profile

    Uses m.evaluate and m.fit_deriv, falling back to
    numerical_jacobian if the latter is None.

    Parameters
    ----------
    m : astropy.modeling.core.Fittable1DModel
        A  astropy.modeling objective function model class or instance.
    epsilon : float, optional
        Step for numerical_gradient.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.modeling
    >>> g=astropy.modeling.functional_models.Gaussian1D()
    >>> l=astropy.modeling.functional_models.Linear1D()
    >>> gaussl=WrapFromAstropy(g+l)

    This has constructed a cubefit-compliant model as the sum of two
    astropy models. Since (g+l).fit_deriv is None, the Jacobian matrix
    will be estimated numerically. We can now call gaussl the way
    cubefit expects.

    >>> x=np.linspace(-10, 10, 5)
    >>> a=[1. , 1. , 0.5, 0.1, 0.5]
    >>> ydata, jacobian = gaussl(x, *a)
    >>> ydata.shape
    (5,)
    >>> jacobian.shape
    (5, 5)
    >>> ydata
    array([-5.00000000e-01,  5.38018616e-32,  6.35335283e-01,  1.00000000e+00,
            1.50000000e+00])
    >>> jacobian[:,0]
    array([0.00000000e+00, 5.38018616e-32, 1.35335283e-01, 0.00000000e+00,
           0.00000000e+00])

    See Also
    --------
    astropy.modeling
    numerical_jacobian

    """

    def __init__(self, m, epsilon=1e-6):
        self.m = m
        self.epsilon = epsilon

    def __call__(self, xdata, *params):
        """ self(xdata, *params) -> ydata, jacobian
        """
        ydata = self.m.evaluate(xdata, *params)
        if self.m.fit_deriv is None:
            jacobian = numerical_jacobian(self.m.evaluate, xdata, *params,
                                          epsilon=self.epsilon)
        else:
            jacobian = np.transpose(self.m.fit_deriv(xdata, *params))
        return ydata, jacobian


def numerical_jacobian(f, xdata, *params, epsilon=1e-6):
    """Compute Jacobian matrix of a curve_fit model function

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
     xdata : array_like
        The independent variable where the data is measured.
    *a : tuplet
        Parameters of the model.
    epsilon : array_like, optional
        Step for finite-difference estimation of the Jacobian
        matrix. Can be a scalar (same step for all parameters) or same
        size as params.

    Returns
    -------
    jac : array_like
        Estimate of the Jacobian.

    See Also
    --------
    scipy.optimize.curve_fit
    """
    params = np.asarray(params, dtype=np.float64)
    if np.isscalar(xdata):
        xdata = np.float64(xdata)
    else:
        xdata = np.asarray(xdata, dtype=np.float64)
    nterms = params.size
    jac = np.zeros(xdata.shape + (nterms,))

    for k in range(nterms):
        ah = np.copy(params)
        ah[k] += 0.5*epsilon
        yp = f(xdata, *ah)
        ah[k] -= epsilon
        ym = f(xdata, *ah)
        jac[..., k] = (yp-ym)/epsilon

    return jac


if __name__ == '__main__':
    import doctest
    # doctest.testmod()
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
