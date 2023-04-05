#!/usr/bin/env python3

import numpy as np
# from math import sqrt
import matplotlib.pyplot as plt
from scipy import optimize


# TODO: integrate in lineprofiles.py


def moffat1d(x, *a):
    '''Compute a Moffat profile.

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
        The values of the Gaussian with paramaters a computed at
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

    This function can be used directly with curve_fit.

    Limitation: "b"  should  always  be  positive.  In  order  to  force  it,
    especially in fitting routines, its  abolute value is taken (except at some
    point in the computation of derivates).

    See Also
    --------
    cubefit.lineprofiles.ngauss moffat1d_fit, asmoffat1d, asmoffat1d_fit
    '''
# from chatgpt
    # TODO python flavor ?
    __moffat_betamax = None
    __moffat_vmax = None
    __moffat_gradmax = None
    a = np.asarray(a)
    # print("DBG CALL moffat1d")
    # print(f"a {a}")
    # global __moffat_betamax, __moffat_vmax, __moffat_gradmax
    if __moffat_gradmax is None:
        __moffat_gradmax = 1e150
    if __moffat_betamax is not None:
        big = __moffat_betamax
    else:
        big = 1e18

    a = np.asarray(a)
    x = np.asarray(x)
    # print(f"a asarray {a}")
    # nterms = len(a)
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


def asmoffat1d(x, *a):
    '''Compute an asymmetrical Moffat profile

    Returns a (1D) asymmetrical Moffat profile:
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

    This function can be used directly with curve_fit.

    See Also
    --------
    moffat1d, moffat1d_fit, asmoffat1d_fit
    '''
# from chatgpt
    # print("DBG call asmoffat1d")
    # print(f"a is {a}")
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
    u3 = u3a * ta + u3b * tb
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
    # print(f"a is {a}")
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
    # if deriv:
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


def moffat1d_fit(y, x, w, guess=None, nterms=None, itmax=None):
    '''Fits a moffat (see  moffat1d) profile on a data set using
    curve_fit (see curve_fit).

    The set of data points Y is the only mandatory argument, X defaults
    to indgen(numberof(y)), weights W are optional (see curve_fit).
    MOFFAT1D_FIT tries to guess a set of initial parameters, but you can
    (and should in every  non-trivial case) provide one using the
    GUESS keyword. In  case you don't provide a guess, you should set NTERMS
    to 4 (simple moffat), 5 (adjust constant baseline)
    or 6 (adjust linear baseline). The returned fitted parameters have
    the same format as GUESS, see moffat1d.

    See Also
    --------
    moffat1d, asmoffat1d, asmoffat1d_fit
    '''
# from chatgpt
# import numpy as np
# from curve_fit import minimize, Parameters
#
# def moffat1d(x, a):
#    # Moffat function
#    gamma = a['gamma'].value
#    alpha = a['alpha'].value
#    beta = a['beta'].value
#    return beta / (1.0 + ((x - alpha) / gamma) ** 2) ** beta
#
# def moffat1d_fit(y, x=None, w=None, guess=None, nterms=None, itmax=None):
    if x is None:
        x = np.arange(len(y))
    if guess is None:
        if nterms is None:
            nterms = 4
        if nterms < 4:
            nterms = 4
        if nterms > 6:
            nterms = 6
        guess = np.zeros(nterms)
        if nterms == 5:
            base = np.median(y)
            guess.add('base', base)
        elif nterms == 6:
            n = len(y)
            y1 = np.median(y[:n // 2])
            x1 = np.median(x[:n // 2])
            y2 = np.median(y[-n // 2:])
            x2 = np.median(x[-n // 2:])
            gamma = (x2 - x1) / 2.0
            alpha = (x1 + x2) / 2.0
            beta = 2.5
            base = y1 - moffat1d(alpha, {'gamma': gamma, 'alpha': alpha,
                                 'beta': beta})
            guess.add('gamma', gamma)
            guess.add('alpha', alpha)
            guess.add('beta', beta)
            guess.add('base', base)
        else:
            base = 0.0
        y2 = y - base
        ind0 = np.argmax(np.abs(y2))
        guess.add('alpha', x[ind0])
        guess.add('base', y2[ind0])
        if y2[ind0] == guess['base'].value:
            yy = y2
        else:
            yy = -y2
        ind1 = ind0
        ind2 = ind0
        while ind1 > 0 and yy[ind1] > 0.5 * guess['base'].value:
            ind1 -= 1
        if yy[ind1] < 0.5 * guess['base'].value:
            ind1 += 1
        while ind2 < len(y) - 1 and yy[ind2] > 0.5 * guess['base'].value:
            ind2 += 1
        if yy[ind2] < 0.5 * guess['base'].value:
            ind2 -= 1
        guess.add('gamma', abs(x[ind2] - x[ind1]))
        guess.add('beta', 1.0)

    else:
        nterms = guess.size
    result, req = optimize.curve_fit(moffat1d, x, y, guess, maxfev=itmax)
    return result


def asmoffat1d_fit(y, x, w, guess=None, nterms=None):
    '''Fits an  assymetrical moffat  (see asmoffat1d) profile  on a data
    set using curve_fit  (see  curve_fit).

    The  set  of  data points  Y  is  the only  mandatory argument,
    X defaults to indgen(numberof(y)), weights W are optional (see curve_fit).
    ASMOFFAT1D_FIT tries to guess a set of initial parameters, but you
    can (and  should in every non-trivial case) provide one using
    the GUESS keyword. In  case you don't provide a guess, you should set
    NTERMS to 6 (simple assymmetrical  moffat), 7 (adjust constant baseline)
    or 8 (adjust linear baseline). The  returned fitted parameters have the
    same format as GUESS, see asmoffat1d.

    See Also
    --------
    asmoffat1d, moffat1d, moffat1d_fit
    '''
# import numpy as np
# from scipy.optimize import least_squares, minimize_scalar
#
# def asmoffat1d(x, a):
#    I0, x0, dx, b, alpha, slope, offset = a
#
#    if np.abs(dx) < 1e-80:
#        dx1 = np.sign(np.sign(dx))/np.finfo(dx).eps
#    else:
#        dx1 = 1.0/dx
#
#    X = (x-x0)*dx1
#    R2 = X**2
#
#    u3 = 1+R2
#    u1 = u3**(-np.abs(b))
#    mof = I0*u1 + slope*x + offset
#
#    return mof
#
# def asmoffat1d_fit(y, x=None, w=None, guess=None, nterms=None):
    if x is None:
        x = np.arange(len(y))

    if guess is None:
        if nterms is None:
            nterms = 6
        if nterms < 6:
            nterms = 6
        if nterms > 8:
            nterms = 8

        guess = np.zeros(nterms)
        if nterms == 7:
            base = np.median(y)
            guess[-1] = base
        elif nterms == 8:
            n = len(y)
            y1 = np.median(y[:int(n/2)])
            x1 = np.median(x[:int(n/2)])
            y2 = np.median(y[-int(n/2):])
            x2 = np.median(x[-int(n/2):])
            guess[-2] = (y2-y1)/(x2-x1)
            if guess[-2] != 0:
                guess[-1] = y1 - guess[-2]*x1
            base = guess[-1] + guess[-2]*x
        else:
            base = 0.

        y2 = y - base
        ind0 = np.argmax(np.abs(y2))
        guess[0] = y2[ind0]
        guess[1] = x[ind0]
        yy = np.where(y2[ind0:] == guess[0], y2[ind0:], -y2[ind0:])
        ind1 = ind0
        ind2 = ind0
        while ind1 > 0 and yy[ind1-ind0] > 0.5*guess[0]:
            ind1 -= 1
        if yy[ind1-ind0] < 0.5*guess[0]:
            ind1 += 1
        while ind2 < len(y)-1 and yy[ind2-ind0] > 0.5*guess[0]:
            ind2 += 1
        if yy[ind2-ind0] < 0.5*guess[0]:
            ind2 -= 1
        guess[2] = np.abs(x[ind2] - x[ind1])
        guess[3] = 1.
        guess[4] = np.abs(x[ind2] - x[ind1])
        guess[5] = 1.

    else:
        nterms = len(guess)

    if w is None:
        w = np.ones_like(y)

    result, req = optimize.curve_fit(asmoffat1d, x, y, guess)

    return result, req


def moffat2d(xy, a, grad, deriv=None):
    '''
    /* DOCUMENT moffat2d(xy,a)

    Returns a (2D) Moffat profile:
     I=I0*(1+(X/dx)^2+(Y/dy)^2)^-b

    Where:
     X=(x-x0)*cos(alpha)+(y-y0)*sin(alpha);
     Y=(y-y0)*cos(alpha)-(x-x0)*sin(alpha);
     x=xy(..,1); y=xy(..,2);

    Paramater "a" is a vector with 5 or 7 elements:
        a = [I0, x0, y0, dx=dy, b] (then alpha=0)
     or a = [I0, x0, y0, dx, dy, b, alpha].

    This function can be used directly with curve_fit and provides
    derivatives. Contrary to the similar functions gauss(), moffat1d()
    and gauss2d(), moffat2d() does not offer the possibility to add a
    linear background. See multiprofile.i for compositing several
    curve_fit functions.

    Limitation:  "b"  should  always  be  positive.  In  order  to  force  it,
    especially in fitting routines, its  abolute value is taken (except at some
    point in the computation of derivates).

    astro_util1.i contains two variants of this function: moffat and
    moffatRound. Those two functions do not provide derivatives, take
    alpha in degrees instead of radians, and allow fitting a allow
    fitting a cnstant background, and take a slightly different A
    vector.

   SEE ALSO: moffat1d, gauss2d, moffat, moffatRound
    */
    '''
# from chatgpt
# import numpy as np
#
# def moffat2d(xy, a, deriv=True):
    a = np.array(a, dtype=np.float64)
    npars = a.size

    I0 = a[0]
    x0 = a[1]
    y0 = a[2]
    dx = a[3]
    dy = a[4] if npars >= 6 else dx
    b = a[5] if npars >= 6 else a[4]
    alpha = a[6] if npars >= 7 else 0.0

    small = 1e-80
    dx1 = np.sign(np.sign(dx)) / np.finfo(np.float64).eps if abs(dx) < small else 1.0 / dx
    dy1 = np.sign(np.sign(dy)) / np.finfo(np.float64).eps if abs(dy) < small else 1.0 / dy

    deltax = xy[:, 0] - x0
    deltay = xy[:, 1] - y0
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    X = (deltax * cosa + deltay * sina) * dx1
    Y = (deltay * cosa - deltax * sina) * dy1
    R2 = X**2 + Y**2

    u3 = 1 + R2
    u1 = u3**-np.abs(b)
    u1b = u3**-np.abs(b + 1)
    mof = I0 * u1
    if deriv:
        grad = np.zeros((xy.shape[0], npars), dtype=np.float64)
        grad[:, 0] = u1
        grad[:, 1] = 2.0 * I0 * b * dx1 * X * u1b
        grad[:, 2] = 2.0 * I0 * b * dy1 * Y * u1b
        grad[:, 3] = grad[:, 1] * X
        if npars >= 6:
            grad[:, 4] = grad[:, 2] * Y
        else:
            grad[:, 3] += grad[:, 2] * Y
        grad[:, -1] = -np.log(u3) * mof
        grad[:, 0] = 2.0 * b * I0 * (dx * dy1 - dy * dx1) * X * Y * u1b

        return mof, grad
    else:
        return mof


class WrapToCurveFit:
    '''Wrap a cubefit profile to a curve_fit objective function

    Parameters
    ----------
    profile : callable
        A curvefit.lineprofiles profile.
    '''
    def __init__(self, profile):
        self.profile = profile

    def __call__(self, xdata, *params):
        '''self(xdata, *params) -> self.profile(xdata, *params)[0]
        '''
        return self.profile(xdata, *params)[0]

    def jac(self, xdata, *params):
        '''self.jac(xdata, *params) -> self.profile(xdata, *params)[1]
        '''
        return self.profile(xdata, *params)[1]


def test_moffat():
    plot = 1
    # test moffat1d
    a = np.array([1, 1, 0.5, 0.5, 0.1])
    x = np.linspace(-10, 10, 3000)
    # print(x)
    ret, ret_jac = moffat1d(x, *a)
    # ret_jac = gauss_jac(x, *a)
    if plot:
        plt.figure()
        plt.xlim(-10, 10)
        plt.plot(x, ret)
        plt.figure()
        plt.plot(x, ret_jac)
        plt.show()

    sigma = 0.02
    y = ret + np.random.standard_normal(ret.size) * sigma

    # TODO add test of the gradient with a optimize.curve_fit
    print("===FIT grad ==========")
    a0 = np.array([1.5, 0.4, 1., 0.2, 0.5])

    # wrap moffat1d in a way suitable for curve_fit
    curve_fit_func = WrapToCurveFit(moffat1d)

    resopt, reqcov = optimize.curve_fit(curve_fit_func, x, y, p0=a0)
    print("uhUH")
    resopt_jac, reqcov_jac = optimize.curve_fit(curve_fit_func, x, y, p0=a0,
                                                jac=curve_fit_func.jac)

    print("uh2UH2")
    model = moffat1d(x, *resopt)[0]
    model_jac = moffat1d(x, *resopt_jac)[0]
    chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
    chi2_jac = np.sum(((y-model_jac)/sigma)**2)/(y.size-a.size+1)

    print(f"a initial {a}")
    print("=======chi2")
    print(f"chi2 reduit {chi2}")
    print(f"chi2_jac reduit {chi2_jac}")
    print(f"a0 {a0}")
    print(f"resopt {resopt}")
    print(f"resopt_jac {resopt_jac}")

    if plot:
        plt.figure()
        # plt.plot(waxis, dop(*a0))
        plt.plot(x, model, label="model")
        plt.plot(x, model_jac, label="model_jac")
        plt.plot(x, y, label="y")
        plt.legend()
        plt.show()

    # test asmoffat1d
    na = np.array([1, 1, 0.5, 0.5, 0.1, 0.5])
    # x=np.arange(-50,50,0.3)
    # x=3.5
    nx = np.linspace(-10, 10, 3000)
    # print("x ")
    # print(x)
    nret, nret_jac = asmoffat1d(nx, *na)
    # nret_jac = ngauss_jac(nx, *na)
    if plot:
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
    # a0 = np.array([1.5, 0.4, 2., 5., 1.5, 5.])
    a0 = np.array([1.5, 0.4, 1., 0.2, 0.5 , 0.4])

    # wrap moffat1d in a way suitable for curve_fit
    curve_fit_func = WrapToCurveFit(asmoffat1d)

    resopt, reqcov = optimize.curve_fit(curve_fit_func, nx, y, p0=a0)
    resopt_jac, reqcov_jac = optimize.curve_fit(curve_fit_func, nx, y, p0=a0,
                                                jac=curve_fit_func.jac)

    model = asmoffat1d(nx, *resopt)[0]
    model_jac = asmoffat1d(nx, *resopt_jac)[0]
    chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
    chi2_jac = np.sum(((y-model_jac)/sigma)**2)/(y.size-a.size+1)

    print(f"a initial {a}")
    print("=======chi2")
    print(f"chi2 reduit {chi2}")
    print(f"chi2_jac reduit {chi2_jac}")
    print(f"a0 {a0}")
    print(f"resopt {resopt}")
    print(f"resopt_jac {resopt_jac}")

    if plot:
        plt.figure()
        # plt.plot(waxis, dop(*a0))
        plt.plot(nx, model, label="model")
        plt.plot(nx, model_jac, label="model_jac")
        plt.plot(nx, y, label="y")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    test_moffat()
