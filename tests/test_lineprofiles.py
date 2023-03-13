from .common import *
from cubefit.lineprofiles import gauss, ngauss

class TestGauss(Test1DModel):
    '''UnitTest class to test gauss function
    '''
    def test_gauss_jacobian(self):
        a = [1. , 1. , 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(gauss, x, *a)

class TestNGauss(Test1DModel):
    '''UnitTest class to test ngauss function
    '''
    def test_ngauss_jacobian(self):
        a = [1. , 1. , 0.5, 0.5, 0.1]
        x = np.linspace(-10, 10, 3000)
        self.check_jacobian(ngauss, x, *a)

if __name__ == '__main__':
   unittest.main()
