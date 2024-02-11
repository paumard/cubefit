import os
from .common import *
from cubefit.lineprofiles import gauss, \
    WrapToCurveFit, WrapFromCurveFit, WrapFromAstropy
from cubefit.multiprofile import MultiProfile

DEBUG=os.environ.get("TEST_MULTIPROFILE_DEBUG")
if DEBUG:
    from matplotlib import pyplot as plt

class TestMultiProfile(Test1DModel):
    '''UnitTest class to test MultiProfile class
    '''
    def __init__(self, *args, **kwargs):
        super(TestMultiProfile, self).__init__(*args, **kwargs)
        self.gauss3_1=MultiProfile(gauss, 3)
        self.gauss4_1=MultiProfile(gauss, 4)
        self.gauss5_1=MultiProfile(gauss, 5)
        self.gauss3_3=MultiProfile(gauss, 3, 3)
        self.gauss3_3_equals=MultiProfile(gauss, 3, 3, equals=((2,),) )
        self.gauss3x3_1=MultiProfile((gauss,gauss,gauss), 3)
        self.gauss3_1_5_1=MultiProfile((gauss, gauss), (3, 5))
        
    def test_gaussN_1(self):
        # Test Gaussian for a few known values
        params=(1., 0., 1.)
        self.assertAlmostEqual(self.gauss3_1(0., *params)[0], 1)
        self.assertAlmostEqual(self.gauss3_1(0., *params)[0], 1)
        self.assertAlmostEqual(self.gauss3_1(1., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(self.gauss3_1(2., *params)[0], np.exp(-2))
        self.assertAlmostEqual(self.gauss3_1(-2., *params)[0], np.exp(-2))
        params=(2., 0., 1.)
        self.assertAlmostEqual(self.gauss3_1(0., *params)[0], 2)
        self.assertAlmostEqual(self.gauss3_1(1., *params)[0], 2*np.exp(-0.5))
        self.assertAlmostEqual(self.gauss3_1(2., *params)[0], 2*np.exp(-2))
        self.assertAlmostEqual(self.gauss3_1(-2., *params)[0], 2*np.exp(-2))
        params=(1., 1., 1.)
        self.assertAlmostEqual(self.gauss3_1(1., *params)[0], 1)
        self.assertAlmostEqual(self.gauss3_1(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(self.gauss3_1(3., *params)[0], np.exp(-2))
        self.assertAlmostEqual(self.gauss3_1(-1., *params)[0], np.exp(-2))
        params=(1., 0., 1., 2)
        with self.assertRaises(AssertionError):
            self.gauss3_1(0., *params)
        self.assertAlmostEqual(self.gauss4_1(0., *params)[0], 1+2)
        self.assertAlmostEqual(self.gauss4_1(1., *params)[0], np.exp(-0.5)+2)
        self.assertAlmostEqual(self.gauss4_1(2., *params)[0], np.exp(-2)+2)
        self.assertAlmostEqual(self.gauss4_1(-2., *params)[0], np.exp(-2)+2)
        params=(1., 0., 1., 2, 2)
        self.assertAlmostEqual(self.gauss5_1(0., *params)[0], 1+2)
        self.assertAlmostEqual(self.gauss5_1(1., *params)[0], np.exp(-0.5)+2+2)
        self.assertAlmostEqual(self.gauss5_1(2., *params)[0], np.exp(-2)+2+4)
        self.assertAlmostEqual(self.gauss5_1(-2., *params)[0], np.exp(-2)+2-4)
        params=(1., 0., 2.)
        self.assertAlmostEqual(self.gauss3_1(0., *params)[0], 1)
        self.assertAlmostEqual(self.gauss3_1(2., *params)[0], np.exp(-0.5))
        self.assertAlmostEqual(self.gauss3_1(4., *params)[0], np.exp(-2))
        self.assertAlmostEqual(self.gauss3_1(-4., *params)[0], np.exp(-2))

    def test_gauss5_1_jacobian(self):
        a = [1. , 1. , 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss5_1, x, *a)

    def test_gauss3_3(self):
        # Test Gaussian for a few known values
        params1=(1., 0., 1.)
        params2=(2., 0., 1.)
        params3=(1., 1., 1.)
        params=params1+params2+params3
        for x in (0., 1., 2., -2.):
            self.assertAlmostEqual(self.gauss3_3(x, *params)[0],
                                   gauss(x, *params1)[0]+
                                   gauss(x, *params2)[0]+
                                   gauss(x, *params3)[0])

    def test_gauss3_3_jacobian(self):
        params1=(1., 0., 1.)
        params2=(2., 0., 1.)
        params3=(1., 1., 1.)
        params=params1+params2+params3
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss3_3, x, *params)

    def test_gauss3_3_equals(self):
        # Test Gaussian for a few known values
        params1=(1., 0., 1.)
        params2=(2., 0.)
        params3=(1., 1.)
        params=params1+params2+params3
        for x in (0., 1., 2., -2.):
            self.assertAlmostEqual(self.gauss3_3_equals(x, *params)[0],
                                   gauss(x, *params1)[0]+
                                   gauss(x, *(params2+(params1[2],)) )[0]+
                                   gauss(x, *(params3+(params1[2],)) )[0])

    def test_gauss3_3_equals_jacobian(self):
        params1=(1., 0., 1.)
        params2=(2., 0.)
        params3=(1., 1.)
        params=params1+params2+params3
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss3_3_equals, x, *params)

    def test_gauss3x3_1(self):
        # Test Gaussian for a few known values
        params1=(1., 0., 1.)
        params2=(2., 0., 1.)
        params3=(1., 1., 1.)
        params=params1+params2+params3
        for x in (0., 1., 2., -2.):
            self.assertAlmostEqual(self.gauss3x3_1(x, *params)[0],
                                   gauss(x, *params1)[0]+
                                   gauss(x, *params2)[0]+
                                   gauss(x, *params3)[0])

    def test_gauss3x3_1_jacobian(self):
        params1=(1., 0., 1.)
        params2=(2., 0., 1.)
        params3=(1., 1., 1.)
        params=params1+params2+params3
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss3x3_1, x, *params)

    def test_gauss3_1_5_1(self):
        # Test Gaussian for a few known values
        params1=(1., 1., 1.)
        params2=(1., 0., 1., 2, 2)
        params=params1+params2
        for x in (0., 1., 2., -2.):
            self.assertAlmostEqual(self.gauss3_1_5_1(x, *params)[0],
                                   gauss(x, *params1)[0]+
                                   gauss(x, *params2)[0])

    def test_gauss3_1_5_1jacobian(self):
        params1=(1., 1., 1.)
        params2=(1., 0., 1., 2, 2)
        params=params1+params2
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(self.gauss3_1_5_1, x, *params, epsilon=1e-5)

if __name__ == '__main__':
   unittest.main()
