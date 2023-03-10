from .common import *
from cubefit.dopplerlines import DopplerLines
from cubefit import CubeFit
from cubefit.ngauss import ngauss

class TestCubefit(unittest.TestCase):
    '''UnitTest class to test gauss function
    '''
    def check_gradient(self, f, x, epsilon=1e-6, reltol=1e-3, diftol=None, diflim=None):
        if diflim is None:
            diflim=epsilon;
        if diftol is None:
            diftol=diflim*reltol;
        d = x.shape
        g = np.zeros(d)
        f0, g0 = f(x)
        for k in range(d[2]):
            for j in range(d[1]):
                for i in range(d[0]):
                    temp = np.copy(x)
                    temp[i, j, k] += 0.5*epsilon
                    fp, gp=f(temp)
                    temp[i, j, k] -= epsilon
                    fm, gm=f(temp)
                    # this is (f(x+h/2)-f(x-h/2))/h
                    g[i, j, k] = (fp-fm)/epsilon
        absval=0.5*np.abs(g+g0)
        difval=np.abs(g-g0)
        cond=absval>diflim
        if np.any(cond):
            maxrel=np.max(difval[cond]/absval[cond])
            self.assertTrue(maxrel < reltol, f"Gradient is not within relative tolerance (max: {maxrel}, reltol: {reltol}, diflim: {diflim})")
        cond=absval<=diflim
        if np.any(cond):
            maxdif=np.max(difval[cond])
            self.assertTrue(maxdif < diftol, f"Gradient is not within absolute tolerance (max: {maxdif}, diftol: {diftol}, diflim: {diflim})")


    def test_cubefit_gradient(self):
        nx = 5
        ny = 5
        nz = 433

        cube_zeros = np.zeros((nx, ny, nz))
        weight = np.ones((nx, ny, nz))

        xreal_1d = np.array([1.2, 0.5, 25., 100])
        nterms = len(xreal_1d)
        xreal = np.zeros((nx, ny, xreal_1d.size))
        for k in range(nterms):
            xreal[:, :, k]=xreal_1d[k]
        waxis = np.linspace(2.15, 2.175, nz)
        lineobj = DopplerLines(2.166120, waxis, profile=ngauss)
        fitobj = CubeFit(cube_zeros, lineobj, waxis, weight)

        self.check_gradient(fitobj.eval, xreal, epsilon=1e-10)

if __name__ == '__main__':
   unittest.main()
