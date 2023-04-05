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

import unittest
import numpy as np
import matplotlib.pyplot as plt
from cubefit.lineprofiles import numerical_jacobian, WrapToCurveFit, WrapFromCurveFit

class Test1DModel(unittest.TestCase):

    def check_jacobian(self, f, x, *a, epsilon=1e-6, reltol=1e-3, diftol=None, diflim=None):
        '''Check the Jacobian of f where f is a callable with syntax
         ymodel, dyda = f(xdata, a)
        xdata: independent variable
        a: parameters
        ymodel: dependant variable
        dyda: Jacobian matrix, d(ymodel)/d(a)'''
        if diflim is None:
            diflim=epsilon;
        if diftol is None:
            diftol=diflim*reltol;
        a = np.asarray(a, dtype=np.float64)
        if np.isscalar(x):
            x = np.float64(x)
        else:
            x = np.asarray(x, dtype=np.float64)
        nterms = a.size
        jac = np.zeros(x.shape + (nterms,))

        y0, jac0=f(x, *a)
        jac = numerical_jacobian(lambda x, *a: f(x, *a)[0], x, *a)

        absval=0.5*np.abs(jac+jac0)
        difval=np.abs(jac-jac0)
        cond=absval>diflim
        if np.any(cond):
            maxrel=np.max(difval[cond]/absval[cond])
            self.assertTrue(maxrel < reltol, f"Jacobian is not within relative tolerance (max: {maxrel}, reltol: {reltol}, diflim: {diflim})")
        cond=absval<=diflim
        if np.any(cond):
            maxdif=np.max(difval[cond])
            self.assertTrue(maxdif < diftol, f"Jacobian is not within absolute tolerance (max: {maxdif}, diftol: {diftol}, diflim: {diflim})")
