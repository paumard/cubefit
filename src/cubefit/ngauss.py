#!/usr/bin/env python3

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
    '''
    Returns a gaussian:
        I0*exp(-0.5*((x-x0)/dx)^2) [+a[3] [+a[4]*x]]
    Where:
        I0=a[0]
        x0=a[1]
        dx=a[2] (gaussian sigma)

    Works with lmfit, and can return derivates.
    Notes: FHWM=sigma*2*sqrt(2*alog(2)); sum(gauss)=I0*sigma*sqrt(2*pi)
    SEE ALSO: gauss_fit, asgauss, asgauss_fit
    '''
    a = np.asarray(a, dtype=np.float64)

     # a forcement tableau
    #print("gauss call with parameters")
    #print(f"x is {x}")
    #print(f"a is {a}")
    #print(f"type(a){type(a)}")
    #print(f"a.size is {a.size}")

    # print(f"a0 is {a[0]}")
    # print(f"a1 is {a[1]}")
    # print(f"a2 is {a[2]}")


    if np.isscalar(x):
        x = np.float64(x)
    else:
        x = np.asarray(x, dtype=np.float64)

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
    grad = np.zeros(x.shape + (nterms,))
    grad[:, 0] = u1
    grad[:, 1] = res*(x-a[1])/a[2]**2
    grad[:, 2] = res*(x-a[1])**2/a[2]**3
    if (nterms > 3):
        grad[:, 3] = 1.
    if (nterms == 5):
        grad[:, 4] = x

    # print("inside gauss_jac")
    # print(f"x {x}")
    # print(f"a {a}")

    if (nterms > 3):
        res = res+a[3]

    if (nterms == 5):
        res = res+a[4]*x

    return res, grad


def ngauss(x, *a):
    '''
    Returns a normalised Gaussian:
     I0*exp(-0.5*((x-x0)/dx)^2) [+a[3] [+a[4]*x]]
    Where:
    Sum=a[0]
    x0=a[1]
    dx=a[2] (gaussian sigma)
    I0=Sum/(sigma*sqrt(2*pi))

   Works with lmfit, and can return derivates. In contrast with
   gauss(), the first parameter is the integral of the Gaussian rather
   than the peak value.

   Notes: FHWM=sigma*2*sqrt(2*alog(2)); sum(gauss)=I0*sigma*sqrt(2*pi)

   SEE ALSO: gauss, gauss_fit, asgauss, asgauss_fit
    '''

    a = np.asarray(a)

    if np.isscalar(x):
        x = np.float64(x)
    else:
        x = np.asarray(x, dtype=np.float64)

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

    grad[:, 2] -= I0*sigmam1*grad[:, 0]
    grad[:, 0] *= eqwidthm1

    return res, grad


def curvefit_func(xdata, *params):
    # func dopplerlines_curvefit_func(x, a, &grad, deriv=) {

    """

    """
    #print(f"xdata is {xdata}")
    #print(f"params is {params}")
    return gauss(xdata, *params)[0]


def curvefit_jac(xdata, *params):
    # func dopplerlines_curvefit_func(x, a, &grad, deriv=) {

    """

    """
    #print(f"xdata is {xdata}")
    #print(f"params is {params}")
    return gauss(xdata, *params)[1]


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

    resopt, reqcov = optimize.curve_fit(curvefit_func, nx, y, p0=a0)
    resopt_jac, reqcov_jac = optimize.curve_fit(curvefit_func, nx, y, p0=a0,
                                                jac=curvefit_jac)

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
