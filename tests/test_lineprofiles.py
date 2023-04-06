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
import unittest
import numpy as np
from .common import Test1DModel
from cubefit.lineprofiles import gauss, ngauss, WrapToCurveFit, \
 WrapFromCurveFit, WrapFromAstropy

DEBUG = os.environ.get("TEST_LINEPROFILES_DEBUG")


class TestGauss(Test1DModel):
    '''UnitTest class to test gauss function
    '''
    def test_gauss(self):
        # Test Gaussian for a few known values
        params = (1., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2))
        params = (2., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 2)
        self.assertAlmostEqual(gauss(1., *params)[0], 2*np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], 2*np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], 2*np.exp(-2))
        params = (1., 1., 1.)
        self.assertAlmostEqual(gauss(1., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(3., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-1., *params)[0], np.exp(-2))
        params = (1., 0., 1., 2)
        self.assertAlmostEqual(gauss(0., *params)[0], 1+2)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5)+2)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2)+2)
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2)+2)
        params = (1., 0., 1., 2, 2)
        self.assertAlmostEqual(gauss(0., *params)[0], 1+2)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5)+2+2)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2)+2+4)
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2)+2-4)
        params = (1., 0., 2.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(4., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-4., *params)[0], np.exp(-2))

    def test_gauss_jacobian(self):
        a = [1., 1., 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(gauss, x, *a)


class TestAstropyGauss(Test1DModel):
    '''UnitTest class to test astropy.modeling Gaussian function
    '''

    def __init__(self, *args, **kwargs):
        super(TestAstropyGauss, self).__init__(*args, **kwargs)
        import astropy.modeling
        self.gauss = WrapFromAstropy(
            astropy.modeling.functional_models.Gaussian1D)
        gauss1d = astropy.modeling.functional_models.Gaussian1D()
        linear1d = astropy.modeling.functional_models.Linear1D()
        self.gaussl = WrapFromAstropy(gauss1d+linear1d)

    def test_gauss(self):
        gauss = self.gauss
        gaussl = self.gaussl
        # Test Gaussian for a few known values
        params = (1., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2))
        params = (2., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 2)
        self.assertAlmostEqual(gauss(1., *params)[0], 2*np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], 2*np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], 2*np.exp(-2))
        params = (1., 1., 1.)
        self.assertAlmostEqual(gauss(1., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(3., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-1., *params)[0], np.exp(-2))
        params = (1., 0., 1., 0, 2)
        self.assertAlmostEqual(gaussl(0., *params)[0], 1+2)
        self.assertAlmostEqual(gaussl(1., *params)[0], np.exp(-0.5)+2)
        self.assertAlmostEqual(gaussl(2., *params)[0], np.exp(-2)+2)
        self.assertAlmostEqual(gaussl(-2., *params)[0], np.exp(-2)+2)
        params = (1., 0., 1., 2, 2)
        self.assertAlmostEqual(gaussl(0., *params)[0], 1+2)
        self.assertAlmostEqual(gaussl(1., *params)[0], np.exp(-0.5)+2+2)
        self.assertAlmostEqual(gaussl(2., *params)[0], np.exp(-2)+2+4)
        self.assertAlmostEqual(gaussl(-2., *params)[0], np.exp(-2)+2-4)
        params = (1., 0., 2.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(4., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-4., *params)[0], np.exp(-2))

    def test_gauss_jacobian(self):
        a = [1., 1., 0.5]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss, x, *a)
        # Check that the Jacobian is still fine if fit_deriv is None
        self.assertEqual(self.gaussl.m.fit_deriv, None)
        a = [1., 1., 0.5, 0.1, 0.5]
        self.check_jacobian(self.gaussl, x, *a)


class TestNGauss(Test1DModel):
    '''UnitTest class to test ngauss function
    '''
    def test_ngauss(self):
        # Test Gaussian for a few known values
        params = (1., 0., 1.)
        xdata = np.linspace(-10, 10, 2001)
        ydata = ngauss(xdata, *params)[0]
        self.assertAlmostEqual(ydata.sum()*1e-2, 1.)
        params = (2., 3., 4)
        xdata = np.linspace(-40, 40, 8001)
        ydata = ngauss(xdata, *params)[0]
        self.assertAlmostEqual(ydata.sum()*1e-2, 2.)
        self.assertEqual(ngauss(xdata[4000], *params)[0], ydata[4000])

    def test_ngauss_jacobian(self):
        a = [1., 1., 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(ngauss, x, *a)


class TestCurveFitWrappers(Test1DModel):
    def test_wrap(self):
        origfunc = gauss
        cvfitfunc = WrapToCurveFit(origfunc)
        cbfitfunc_num = WrapFromCurveFit(cvfitfunc)
        cbfitfunc_ana = WrapFromCurveFit(cvfitfunc, cvfitfunc.jac)
        a = [1., 1., 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        origval, origjac = origfunc(x, *a)
        cvval, cvjac = cvfitfunc(x, *a), cvfitfunc.jac(x, *a)
        cbnval, cbnjac = cbfitfunc_num(x, *a)
        cbaval, cbajac = cbfitfunc_ana(x, *a)
        for k in range(np.size(origval)):
            self.assertEqual(origval[k], cvval[k])
            self.assertEqual(origval[k], cbnval[k])
            self.assertEqual(origval[k], cbaval[k])
            for param in range(np.size(a)):
                self.assertEqual(origjac[k, param], cvjac[k, param])
                self.assertAlmostEqual(origjac[k, param], cbnjac[k, param])
                self.assertEqual(origjac[k, param], cbajac[k, param])


if __name__ == '__main__':
    unittest.main()
