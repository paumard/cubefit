from .common import *
from cubefit.lineprofiles import gauss, ngauss

class TestGauss(Test1DModel):
    '''UnitTest class to test gauss function
    '''
    def test_gauss(self):
        # Test Gaussian for a few known values
        params=(1., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2))
        params=(2., 0., 1.)
        self.assertAlmostEqual(gauss(0., *params)[0], 2)
        self.assertAlmostEqual(gauss(1., *params)[0], 2*np.exp(-0.5))
        self.assertAlmostEqual(gauss(2., *params)[0], 2*np.exp(-2))
        self.assertAlmostEqual(gauss(-2., *params)[0], 2*np.exp(-2))
        params=(1., 1., 1.)
        self.assertAlmostEqual(gauss(1., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(3., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-1., *params)[0], np.exp(-2))
        params=(1., 0., 1., 2)
        self.assertAlmostEqual(gauss(0., *params)[0], 1+2)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5)+2)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2)+2)
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2)+2)
        params=(1., 0., 1., 2, 2)
        self.assertAlmostEqual(gauss(0., *params)[0], 1+2)
        self.assertAlmostEqual(gauss(1., *params)[0], np.exp(-0.5)+2+2)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-2)+2+4)
        self.assertAlmostEqual(gauss(-2., *params)[0], np.exp(-2)+2-4)
        params=(1., 0., 2.)
        self.assertAlmostEqual(gauss(0., *params)[0], 1)
        self.assertAlmostEqual(gauss(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(gauss(4., *params)[0], np.exp(-2))
        self.assertAlmostEqual(gauss(-4., *params)[0], np.exp(-2))

    def test_gauss_jacobian(self):
        a = [1. , 1. , 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(gauss, x, *a)

class TestNGauss(Test1DModel):
    '''UnitTest class to test ngauss function
    '''
    def test_ngauss(self):
        # Test Gaussian for a few known values
        params=(1., 0., 1.)
        xdata=np.linspace(-10, 10, 2001)
        ydata=ngauss(xdata, *params)[0]
        self.assertAlmostEqual(ydata.sum()*1e-2, 1.)
        params=(2., 3., 4)
        xdata=np.linspace(-40, 40, 8001)
        ydata=ngauss(xdata, *params)[0]
        self.assertAlmostEqual(ydata.sum()*1e-2, 2.)
        self.assertEqual(ngauss(xdata[4000], *params)[0], ydata[4000])

    def test_ngauss_jacobian(self):
        a = [1. , 1. , 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(ngauss, x, *a)

if __name__ == '__main__':
   unittest.main()
