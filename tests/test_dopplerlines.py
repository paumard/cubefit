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

import os
from scipy import optimize
import unittest
import numpy as np
from .common import Test1DModel
from cubefit.dopplerlines import DopplerLines
from cubefit.lineprofiles import ngauss

DEBUG = os.environ.get("TEST_DOPPLERLINES_DEBUG")

if DEBUG:
    from matplotlib import pyplot as plt


class TestDopplerlines(Test1DModel):
    '''UnitTest class to test gauss function
    '''
    def __init__(self, methodName='runTest'):
        self.waxis = np.linspace(2.15, 2.175, 100)
        self.lineobj1 = DopplerLines(2.166120)
        self.lineobj2 = DopplerLines((2.166120, 2.155))
        Test1DModel.__init__(self, methodName)

    def test_dopplerlines_jacobian(self):
        self.check_jacobian(self.lineobj1, self.waxis, 1., 0., 50.,
                            reltol=1e-2, diftol=1e-9)
        self.check_jacobian(self.lineobj2, self.waxis, 1.2, 0.5, 25., 100.,
                            reltol=1e-2)
        waxis = np.linspace(2.15, 2.175, 433)
        lineobj = DopplerLines(2.166120, profile=ngauss)
        xreal_1d = np.array([1.2, 0.5, 25., 100])
        self.check_jacobian(lineobj, waxis, *xreal_1d, epsilon=1e-6,
                            reltol=1e-2, diftol=2e-8)

    def test_dopplerlines(self):
        """
        test_dopplerlines
            x = span(2.15, 2.175, 100);
            obj = DopplerLines(new, waxis = x, lines=2.166120);
            y = obj(eval, [1.2, 25., 100.]) + random_n(100) * sigma;
            plg, y, x;
            a = [1., 0., 50.];
            curvefit, obj.curvefit_func, obj, a, y, deriv=1;
            write, format="Model parameters: [%e, %e, %e]\n", a(1), a(2), a(3);
            model = obj(eval, a);
            plg, color = "red", model, x;
        """
        if DEBUG:
            print("testing dopplerlines module")

        # instanciate a random number generator with fixed seed
        # warning: changing the seed may affect the success of certain
        # tests below
        rng = np.random.default_rng(3)

        sigma = 20.5

        # first test
        if DEBUG:
            print("# first test")
        lines = 2.166120
        waxis = np.linspace(2.15, 2.175, 100)
        dop = DopplerLines(lines)
        if DEBUG:
            print("after init")
        a = np.array([1.2, 25., 100.])
        y = dop(waxis, *a)[0] + rng.standard_normal(100) * sigma

        if DEBUG:
            print(f"---- y {y}")
            print(f"---- y[0] {y[0]}")

        if DEBUG:
            print("=============")
            print("===FIT  1==========")

        a0 = np.array([1., 0., 50.])

        resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                            p0=a0, jac=dop.curvefit_jac)
        resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                              p0=a0)

        model = dop(waxis, *resopt)[0]
        model2 = dop(waxis, *resopt2)[0]
        chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
        chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

        # Raise error if the two fits are not close enough
        for factor in resopt/resopt2:
            self.assertAlmostEqual(factor, 1.)
        self.assertAlmostEqual(chi2, chi22)

        if DEBUG:
            print(f"model parameters: {a}")
            print(f"initial guess {a0}")
            print(f"curvefit result {resopt}")
            print(f"curvefit result with jacobian {resopt2}")
            print("=======chi2")
            print(f"chi2 reduit {chi2}")
            print(f"chi22 reduit {chi22}")

            plt.figure()
            plt.plot(waxis, model)
            plt.plot(waxis, model2)
            plt.plot(waxis, y)
            plt.show()

        # second test two lines
        if DEBUG:
            print("# second test two lines")
        lines = (2.166120, 2.155)
        waxis = np.linspace(2.15, 2.175, 100)
        dop = DopplerLines(lines)
        a = np.array([1.2, 0.5, 25., 100.])
        y = dop(waxis, *a)[0] + rng.standard_normal(100) * sigma

        if DEBUG:
            print("===FIT 2==========")
        a0 = np.array([1., 0.3, 50., 50.])
        resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                            p0=a0, jac=dop.curvefit_jac)
        resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                              p0=a0)

        model = dop(waxis, *resopt)[0]
        model2 = dop(waxis, *resopt2)[0]
        chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
        chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

        # Raise error if the two fits are not close enough
        for factor in resopt/resopt2:
            self.assertAlmostEqual(factor, 1., places=5)
        self.assertAlmostEqual(chi2, chi22)

        if DEBUG:
            print(f"model parameters: {a}")
            print(f"initial guess {a0}")
            print(f"curvefit result {resopt}")
            print(f"curvefit result with jacobian {resopt2}")
            print("=======chi2")
            print(f"chi2 reduit {chi2}")
            print(f"chi22 reduit {chi22}")

            plt.figure()
            # plt.plot(waxis, dop(*a0))
            plt.plot(waxis, model, label="model")
            plt.plot(waxis, model2, label="model2")
            plt.plot(waxis, y, label="y")
            plt.legend()
            plt.show()

        # third test two lines and more parameter
        if DEBUG:
            print("# third test two lines and more parameter")
        lines = (2.166120, 2.155)
        waxis = np.linspace(2.15, 2.175, 100)
        dop = DopplerLines(lines)
        a = np.array([1.2, 0.5, 25., 100., 1.])
        y = dop(waxis, *a)[0] + rng.standard_normal(100) * sigma
        if DEBUG:
            print(f"y==={y}")
            print("===FIT 2 + cst==========")
        a0 = np.array([1., 0.4, 50., 50, 1.5])
        resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                              p0=a0)
        resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y,
                                            p0=a0, jac=dop.curvefit_jac)

        model = dop(waxis, *resopt)[0]
        model2 = dop(waxis, *resopt2)[0]
        chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
        chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

        # Raise error if the two fits are not close enough
        for factor in resopt/resopt2:
            self.assertAlmostEqual(factor, 1., places=5)
        self.assertAlmostEqual(chi2, chi22)

        if DEBUG:
            print(f"model parameters: {a}")
            print(f"initial guess {a0}")
            print(f"curvefit result {resopt}")
            print(f"curvefit result with jacobian {resopt2}")
            print("=======chi2")
            print(f"chi2 reduit {chi2}")
            print(f"chi22 reduit {chi22}")

            plt.figure()
            # plt.plot(waxis, dop(*a0))
            plt.plot(waxis, model, label="model")
            plt.plot(waxis, model2, label="model2")
            plt.plot(waxis, y, label="y")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    unittest.main()
