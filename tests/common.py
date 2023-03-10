import unittest
import numpy as np
import matplotlib.pyplot as plt

def numerical_jacobian(f, x, *a, epsilon=1e-6):
    '''Jacobian = numerical_jacobian(f, x, *a, [epsilon])

    Numerically compute Jacobian matrix of function f where f has
    following calling sequence:

      ydata = f(xdata, parameters)

    the Jacobian matrix is [ d ydata[i] / d parameters[j] ].

    '''
    a = np.asarray(a, dtype=np.float64)
    if np.isscalar(x):
        x = np.float64(x)
    else:
        x = np.asarray(x, dtype=np.float64)
    nterms = a.size
    jac = np.zeros(x.shape + (nterms,))

    for k in range(nterms):
        ah=np.copy(a)
        ah[k] += 0.5*epsilon
        yp = f(x, *ah)
        ah[k] -= epsilon
        ym = f(x, *ah)
        jac[:, k]=(yp-ym)/epsilon

    return jac

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
