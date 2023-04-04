#!/usr/bin/env python3

'''Single-line spectral models

Each profile has the following signature:
  ydata, jacobian = profile(xdata, *parameters)

Wrappers are provided to use these profiles with
scipy.optimize.curve_fit (WrapToCurveFit) and to use functions
designed for curve fit with cubefit (WrapFromCurveFit).

'''

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from scipy import optimize

# for plotting
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename


# DONE ajout boolean switch deriv if deriv return res, grad ?
# def gauss(x, *a, deriv):
def gauss(x, *a):
    '''Compute a Gaussian profile.

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

    See Also
    --------
    cubefit.lineprofiles.ngauss
    '''
    # ensure a and x are numpy arrays and not some other array_like
    # promote to at least float64
    a = np.asarray(a)
    a = np.promote_types(a.dtype, np.float64).type(a)
    x = np.asarray(x)
    x = np.promote_types(x.dtype, np.float64).type(x)

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

    # if deriv:
    grad = np.zeros(np.shape(x) + (nterms,))
    grad[..., 0] = u1
    grad[..., 1] = res*(x-a[1])/a[2]**2
    grad[..., 2] = res*(x-a[1])**2/a[2]**3
    if (nterms > 3):
        grad[..., 3] = 1.
    if (nterms == 5):
        grad[..., 4] = x

    # print("inside gauss_jac")
    # print(f"x {x}")
    # print(f"a {a}")

    if (nterms > 3):
        res = res+a[3]

    if (nterms == 5):
        res = res+a[4]*x

    return res, grad


def ngauss(x, *a):
    '''Compute a normalized Gaussian profile.

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

    See also
    --------
    cubefit.lineprofiles.gauss
    '''

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

    # grad = gauss_jac(x, *a)

    grad[..., 2] -= I0*sigmam1*grad[..., 0]
    grad[..., 0] *= eqwidthm1

    return res, grad


class WrapToCurveFit:
    '''Wrap a cubefit profile to a curve_fit objective function

    Parameters
    ----------
    profile : callable
        A curvefit.lineprofiles profile.
    '''
    def __init__(self, profile):
        self.profile=profile

    def __call__(self, xdata, *params):
        '''self(xdata, *params) -> self.profile(xdata, *params)[0]
        '''
        return self.profile(xdata, *params)[0]

    def jac(self, xdata, *params):
        '''self.jac(xdata, *params) -> self.profile(xdata, *params)[1]
        '''
        return self.profile(xdata, *params)[1]

class WrapFromCurveFit:
    '''Wrap a curve_fit objective function as a cubefit profile

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
    '''
    def __init__(self, f, jac=None, epsilon=1e-6):
        self.f=f
        self.jac=jac
        self.epsilon=epsilon

    def __call__(self, xdata, *params):
        ''' self(xdata, *params) -> (self.f(xdata, *params), self.jac(xdata, *params))
        '''
        val = self.f(xdata, *params)
        if self.jac is None:
            jac = numerical_jacobian(self.f, xdata, *params, epsilon=self.epsilon)
        else:
            jac = self.jac(xdata, *params)
        return val, jac

class WrapFromAstropy:
    '''Wrap a astropy.modeling objective function as a cubefit profile

    Parameters
    ----------
    m : astropy.modeling.core.Fittable1DModel
        A  astropy.modeling objective function model class or instance.

    See Also
    --------
    astropy.modeling
    '''

    def __init__(self, m):
        self.m=m

    def __call__(self, xdata, *params):
        ''' self(xdata, *params) -> ydata, jacobian
        '''
        return (self.m.evaluate(xdata, *params),
                np.transpose(self.m.fit_deriv(xdata, *params)))

def numerical_jacobian(f, xdata, *params, epsilon=1e-6):
    '''Compute Jacobian matrix of a curve_fit model function

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
    '''
    params = np.asarray(params, dtype=np.float64)
    if np.isscalar(xdata):
        xdata = np.float64(xdata)
    else:
        xdata = np.asarray(xdata, dtype=np.float64)
    nterms = params.size
    jac = np.zeros(xdata.shape + (nterms,))

    for k in range(nterms):
        ah=np.copy(params)
        ah[k] += 0.5*epsilon
        yp = f(xdata, *ah)
        ah[k] -= epsilon
        ym = f(xdata, *ah)
        jac[:, k]=(yp-ym)/epsilon

    return jac

def test_gauss():
    test_gauss = True
    # test gauss
    if (test_gauss):
        a = np.array([1, 1, 0.5, 0.5, 0.1])
        x = np.linspace(-10, 10, 3000)
        # print(x)
        ret, ret_jac = gauss(x, *a)
        # ret_jac = gauss_jac(x, *a)
        plt.figure()
        plt.xlim(-10, 10)
        plt.plot(x, ret)
        plt.figure()
        plt.plot(x, ret_jac)
        plt.show()

    # test ngauss
    na = np.array([1, 1, 0.5, 0.5, 0.1])
    # x=np.arange(-50,50,0.3)
    # x=3.5
    nx = np.linspace(-10, 10, 3000)
    # print("x ")
    # print(x)
    nret, nret_jac = ngauss(nx, *na)
    # nret_jac = ngauss_jac(nx, *na)
    plt.figure()
    plt.xlim(-10, 10)
    plt.plot(nx, nret)
    plt.figure()
    plt.plot(nx, nret_jac)
    plt.show()

    sigma = 0.02
    y = nret + np.random.standard_normal(nret.size) * sigma

    # TODO add test of the gradient with a optimize.curve_fit
    print("===FIT grad ==========")
    a0 = np.array([1.5, 0.4, 2., 5., 1.5])

    # wrap gauss in a way suitable for curve_fit
    curve_fit_func=WrapToCurveFit(gauss)

    resopt, reqcov = optimize.curve_fit(curve_fit_func, nx, y, p0=a0)
    resopt_jac, reqcov_jac = optimize.curve_fit(curve_fit_func, nx, y, p0=a0,
                                                jac=curve_fit_func.jac)

    model = gauss(nx, *resopt)[0]
    model_jac = gauss(nx, *resopt_jac)[0]
    chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
    chi2_jac = np.sum(((y-model_jac)/sigma)**2)/(y.size-a.size+1)

    print("=======chi2")
    print(f"chi2 reduit {chi2}")
    print(f"chi2_jac reduit {chi2_jac}")
    print(f"a0 {a0}")
    print(f"resopt {resopt}")
    print(f"resopt_jac {resopt_jac}")

    plt.figure()
    # plt.plot(waxis, dop(*a0))
    plt.plot(nx, model, label="model")
    plt.plot(nx, model_jac, label="model_jac")
    plt.plot(nx, y, label="y")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_gauss()
