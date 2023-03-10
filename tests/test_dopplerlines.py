from scipy import optimize
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
        self.check_jacobian(lineobj, waxis, *xreal_1d, epsilon=1e-6, reltol=1e-2, diftol=2e-8)

    def test_dopplerlines(self, dbg=False):
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
         if dbg:
             print("testing dopplerlines module")

         sigma = 20.5

         # first test
         if dbg:
             print("# first test")
         lines = 2.166120
         waxis = np.linspace(2.15, 2.175, 100)
         dop = DopplerLines(lines, waxis)
         if dbg:
             print("after init")
         a = np.array([1.2, 25., 100.])
         y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
         # y=dop(*a) + np.random.standard_normal(100) * sigma
         # print("ok will change")

         # print(f"y.shape {y.shape}")
         if dbg:
             print(f"---- y {y}")
             print(f"---- y[0] {y[0]}")
         # print(f"*y {y}")
         # plt.figure()
         # plt.plot(waxis,y)
         # plt.show()

         if dbg:
             print("=============")
             print("===FIT  1==========")

         a0 = np.array([1., 0., 50.])

         # optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(- inf, inf), method=None, jac=None, **kwargs)

         resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0, jac=dop.curvefit_jac)
         resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0)

         model = dop(waxis, *resopt)[0]
         model2 = dop(waxis, *resopt2)[0]
         chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
         chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

         if dbg:
             print(f"=======chi2")
             print(f"chi2 reduit {chi2}")
             print(f"chi22 reduit {chi22}")
             print(f"a0 {a0}")
             print(f"resopt {resopt}")
             print(f"resopt2 {resopt2}")

             plt.figure()
             # plt.plot(waxis, dop(*a0))
             plt.plot(waxis, model)
             plt.plot(waxis, model2)
             plt.plot(waxis, y)
             plt.show()
             # jac = dop.curfit_jac(waxis, *a)

         # second test two lines
         if dbg:
             print("# second test two lines")
         lines = (2.166120, 2.155)
         waxis = np.linspace(2.15, 2.175, 100)
         dop = DopplerLines(lines, waxis)
         a = np.array([1.2, 0.5, 25., 100.])
         # y=dop(*a) + np.random.standard_normal(100) * sigma
         y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma

         if dbg:
             print("===FIT 2==========")
         a0 = np.array([1., 0.3, 50., 50.])
         resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0, jac=dop.curvefit_jac)
         resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0)

         model = dop(waxis, *resopt)[0]
         model2 = dop(waxis, *resopt2)[0]
         chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
         chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

         if dbg:
             print(f"=======chi2")
             print(f"chi2 reduit {chi2}")
             print(f"chi22 reduit {chi22}")
             print(f"a0 {a0}")
             print(f"resopt {resopt}")
             print(f"resopt2 {resopt2}")

             plt.figure()
             # plt.plot(waxis, dop(*a0))
             plt.plot(waxis, model, label="model")
             plt.plot(waxis, model2, label="model2")
             plt.plot(waxis, y, label="y")
             plt.legend()
             plt.show()
        # third test two lines and more parameter
         if dbg:
             print("# third test two lines and more parameter")
         lines = (2.166120, 2.155)
         waxis = np.linspace(2.15, 2.175, 100)
         dop = DopplerLines(lines, waxis)
         a = np.array([1.2, 0.5, 25., 100., 1.])
         # y=dop(*a) + np.random.standard_normal(100) * sigma
         y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
         if dbg:
             print(f"y==={y}")
             print("===FIT 2 + cst==========")
         a0 = np.array([1., 0.4, 50., 50, 1.5])
         # resopt,reqcov=optimize.curve_fit(dop.curvefit_func,waxis,y,p0=a0, jac=dop.jacobian)
         resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0)
         resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0, jac=dop.curvefit_jac)

         model = dop(waxis, *resopt)[0]
         model2 = dop(waxis, *resopt2)[0]
         chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
         chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

         if dbg:
             print(f"=======chi2")
             print(f"chi2 reduit {chi2}")
             print(f"chi22 reduit {chi22}")
             print(f"a0 {a0}")
             print(f"resopt {resopt}")
             print(f"resopt2 {resopt2}")

             plt.figure()
             # plt.plot(waxis, dop(*a0))
             plt.plot(waxis, model, label="model")
             plt.plot(waxis, model2, label="model2")
             plt.plot(waxis, y, label="y")
             plt.legend()
             plt.show()

if __name__ == '__main__':
   unittest.main()
