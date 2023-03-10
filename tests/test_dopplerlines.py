from .common import *
from cubefit.dopplerlines import DopplerLines
from cubefit.ngauss import ngauss

class TestDopplerlines(Test1DModel):
    '''UnitTest class to test gauss function
    '''
    def __init__(self, methodName='runTest'):
        self.waxis = np.linspace(2.15, 2.175, 100)
        self.lineobj1 = DopplerLines(2.166120, self.waxis)
        self.lineobj2 = DopplerLines((2.166120, 2.155), self.waxis)
        Test1DModel.__init__(self, methodName)
        
    def test_dopplerlines_jacobian(self):
        self.check_jacobian(self.lineobj1, self.waxis, 1., 0., 50., reltol=1e-2, diftol=1e-9)
        self.check_jacobian(self.lineobj2, self.waxis, 1.2, 0.5, 25., 100., reltol=1e-2)
        waxis = np.linspace(2.15, 2.175, 433)
        lineobj = DopplerLines(2.166120, waxis, profile=ngauss)
        xreal_1d = np.array([1.2, 0.5, 25., 100])
        self.check_jacobian(lineobj, waxis, *xreal_1d, reltol=1e-2)
if __name__ == '__main__':
   unittest.main()
