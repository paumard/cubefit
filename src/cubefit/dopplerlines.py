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

import numpy as np

# First try a relative import. This will work when profiles,
# dopplerlines and cubefit are submodules of a common module.  This
# will fail when calling one of the submodules is called as a script,
# so fall back to a simple import to cope with that use case.
try:
    from cubefit.profiles import gauss, ngauss
except ImportError:
    from profiles import gauss, ngauss


class DopplerLines():
    """
     A class for fitting one or several Doppler shifted lines over a
     spectrum. The line will bear the same Doppler shift and the same
     width (in km/s). The base profile is Gaussian by default but can
     be different (Moffat works as well).

    Attributes
    ----------
     profile : callable
        the base profile. Default: gauss. Profile must be an
        curvefit-compatible function of the form:
            func profile(waxis, *a)
        where WAXIS is the member below and A is of the form:
              [I0, LAMBDA0, DLAMBDA, MORE...]
        I0 is the line intensity, LAMBDA0 the central wavelength
        of a given line, DLAMBDA the line width, and MORE are 0 or
        more parameters which will be equal for all the lines.
     waxis : array_like
        the wavelength axis for which the model will be computed.
     lines : array_like
        the wavelengths of the spectral lines, in the same unit as WAXIS.
     light_speed : float64
        speed of light in the unit in which velocities are
        to be experessed. Defaults to 299792.458 (i.e. velocities
        are in km/s). For the sake of optimisation, light_speed is
        not stored as such in a data member. The corresponding data
        member is c_1=1./light_speed.
     relative : line number relative to which the other's flux is expressed

    Methods
    -------
     eval()
        evaluate the model

     curvefit_func()
        an curvefit-compatible wrapper around eval

    See Also
    --------
     curvefit, dopplerlines.eval, dopplerlines.curvefit_func
    """

    def __init__(self, lines, profile=None, profile_jac=None,
                 light_speed=None, relative=None):

        if profile is None:
            self.profile = ngauss
        else:
            self.profile = profile

        if light_speed is None:
            self.light_speed = 299792.458  # default to km/s

        self.c_1 = 1/self.light_speed

        # declare lines np.zeros? et affectation
        # self.lines=np.asarray(lines)
        # np.array(y_0, ndmin=1, copy=False)
        # force ndim a 1 !!!
        self.lines = np.array(lines, ndmin=1, copy=False)
        self.nlines = self.lines.size

        # relative : index of the reference wavelength in the array
        # TODO write test for special case
        if ((relative is not None)
           and (relative > lines.size or relative <= 0)):
            raise Exception("relative should be > 0 and <= numberof(lines)")
        else:
            self.relative = relative

    def __call__(self, xdata, *params):
        """Evaluate model

        Parameters
        ----------
        xdata : array_like
        params : array_like

        Examples
        --------
        >>> sigma = 0.5
        >>> lines = 2.166120
        >>> waxis = np.linspace(2.15, 2.175, 100)
        >>> dop = DopplerLines(lines)
        >>> a = np.array([1.2, 25., 100.])
        >>> y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma

        See Also
        --------
        dopplerlines, oxy
        """
        vel = params[self.nlines]
        dvel = params[self.nlines+1]
        more = np.array([])

        if len(params) > self.nlines + 2:
            more = np.array(params[self.nlines+2:])

        lambda1 = self.lines * (1 + vel * self.c_1)
        # lambda1b = np.multiply((1 + vel * self.c_1), self.lines)

        widths = lambda1 * dvel * self.c_1

        model = np.zeros(xdata.shape)
        grad = np.zeros((len(params), *xdata.shape), dtype=float)

        aa = np.zeros((3 + more.size))

        for k in range(self.nlines):
            # aa[0] = params[0][k] if np.size(params[0]) > 1 else params[k]
            aa[0] = params[0, k] if np.size(params[0]) > 1 else params[k]
            aa[1] = lambda1[k]
            aa[2] = widths[k]
            if more.size > 0:
                aa[3:] = more

            if self.relative is not None and k != self.relative:
                aa[1] *= params[self.relative]

            acc_model, t_agrad = self.profile(xdata, *aa)
            model += acc_model
            agrad = np.transpose(t_agrad, [1, 0])

            if self.relative is not None:
                if k == self.relative:
                    grad[k, :] += agrad[0, :]
                else:
                    grad[self.relative, :] += params[k] * agrad[0, :]
                    grad[k, :] = params[self.relative] * agrad[0, :]
            else:
                grad[k, :] = agrad[0, :]

            grad[self.nlines, :] += np.asarray(self.lines)[k] * self.c_1 * agrad[1, :]
            grad[self.nlines+1, :] += lambda1[k] * self.c_1 * agrad[2, :]

            if more.size > 0:
                grad[self.nlines+2:, :] += agrad[3:, :]

        return model, np.transpose(grad, [1, 0])


    # In this optimized version, I made the following changes:

    # Removed unnecessary comments and print statements.
    # Removed redundant assignment of lambda1 using np.zeros.
    # Simplified the computation of lambda1 and lambda1b using array multiplication.
    # Avoided using np.append to append elements to more by directly assigning the sliced array.
    # Removed duplicate transpose operation on agrad.
    # Simplified the condition checks by removing unnecessary parentheses.
    # Replaced len(params) with a pre-calculated variable self.nlines for better performance.


    def old_call(self, xdata, *params):
        # def __call__(self, xdata, *params):
        """Evaluate model

        Parameters
        ----------
        xdata : array_like
        params : array_like

        Examples
        --------
        >>> sigma = 0.5
        >>> lines = 2.166120
        >>> waxis = np.linspace(2.15, 2.175, 100)
        >>> dop = DopplerLines(lines)
        >>> a = np.array([1.2, 25., 100.])
        >>> y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma

        See Also
        --------
        dopplerlines, oxy
        """

        vel = params[self.nlines]
        dvel = params[self.nlines+1]
        more = np.array([])

        if (len(params) > self.nlines+2):
            more = np.append(more, params[self.nlines+2:])

        # TODO correct with dimsmin ?
        lambda1 = np.zeros(self.lines.size)

        # np.asarray*float => return float
        lambda1b = np.zeros(self.lines.size)

        # print(f"lamda1.type {type(lambda1)}")
        # print(f"lambda1.shape {lambda1.shape}")
        # print(f"lambda1.size {lambda1.size}")
        # print(f"lambda1 {lambda1[0]}")

        lambda1 = self.lines * (1 + vel * self.c_1)
        lambda1 = np.multiply(np.asarray((1 + vel * self.c_1)),
                              self.lines, out=lambda1b)
        # print(f"lamda1.type {type(lambda1)}")
        # print(f"lambda1.shape {lambda1.shape}")
        # print(f"lambda1.size {lambda1.size}")
        # print(f"lambda1 {lambda1[0]}")

        widths = np.asarray(lambda1 * dvel * self.c_1)

        model = np.zeros(xdata.shape)

        grad = np.zeros((len(params), *xdata.shape), dtype=float)

        aa = np.zeros((3+more.size))

        for k in range(self.nlines):
            if np.size(params[0]) > 1:
                # aa[0] = params[0][k]
                aa[0] = params[0, k]
            else:
                aa[0] = params[k]
            aa[1] = lambda1[k]
            aa[2] = widths[k]
            if more.size > 0:
                aa[3:] = more

            if ((self.relative is not None) and (k != self.relative)):
                aa[1] *= params[self.relative]

            acc_model, t_agrad = self.profile(xdata, *aa)
            model += acc_model
            agrad = np.transpose(t_agrad, [1, 0])
            agrad = np.transpose(t_agrad, [1, 0])

            if (self.relative is not None):
                if (k == self.relative):
                    grad[k, :] += agrad[0, :]
                else:
                    grad[self.relative, :] += params[k] * agrad[0, :]
                    grad[k, :] = params[self.relative] * agrad[0, :]

            else:
                grad[k, :] = agrad[0, :]

            grad[self.nlines, :] += (np.asarray(self.lines)[k]
                                     * self.c_1 * agrad[1, :])

            grad[self.nlines+1, :] += lambda1[k] * self.c_1 * agrad[2, :]

            if (more.size > 0):
                grad[self.nlines+2:, :] += agrad[3:, :]

        return model, np.transpose(grad, [1, 0])

    def curvefit_func(self, xdata, *params):
        """curvefit-compatible wrapper eval

        Examples
        --------
        >>> sigma = 20.5
        >>> lines = 2.166120
        >>> waxis = np.linspace(2.15, 2.175, 100)
        >>> dop = DopplerLines(lines)
        >>> a = np.array([1.2, 25., 100.])
        >>> y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
        >>> a0 = np.array([1., 0., 50.])
        >>> resopt, _ = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0)
        >>> model = dop(waxis, *resopt)[0]
        >>> chi = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)

        # >>> print(f"resopt {resopt}")
        # >>> print(f"chi {chi}")

        See Also
        --------
        curvefit, dopplerlines, dopplerlines.eval

        """
        return self(xdata, *params)[0]

    def curvefit_jac(self, xdata, *params):
        """curvefit-compatible wrapper around eval jacobian function

        Examples
        --------
        >>> sigma = 20.5
        >>> lines = 2.166120
        >>> waxis = np.linspace(2.15, 2.175, 100)
        >>> dop = DopplerLines(lines)
        >>> a = np.array([1.2, 25., 100.])
        >>> y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
        >>> a0 = np.array([1., 0., 50.])
        >>> resopt, _ = optimize.curve_fit(dop.curvefit_func, waxis,
        ...         y, p0=a0, jac=dop.curvefit_jac)
        >>> model = dop(waxis, *resopt)[0]
        >>> chi = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)

        # >>> print(f"resopt {resopt}")
        # >>> print(f"chi {chi}")

        See Also
        --------
        curvefit, dopplerlines, dopplerlines.eval
        """
        return self(xdata, *params)[1]


if __name__ == '__main__':
    import doctest
    # doctest.testmod()
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
