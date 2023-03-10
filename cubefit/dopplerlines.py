import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from ngauss import ngauss

# TODO: pythonify docstrings

# for plotting
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename


class dopplerlines():
    """
     A class for fitting one or several Doppler shifted lines over a
     spectrum. The line will bear the same Doppler shift and the same
     width (in km/s). The base profile is Gaussian by default but can
     be different (Moffat works as well).

    MEMBERS
     The NEW method accepts the following keywords. The data members
     of the class are the same as those keywords, except for
     light_speed (which see).

     profile= the base profile. Default: gauss. Profile must be an
            curvefit-compatible function of the form:
              func profile(waxis, a, &grad, deriv=) {}
            where WAXIS is the member below and A is of the form:
              [I0, LAMBDA0, DLAMBDA, MORE...]
            I0 is the line intensity, LAMBDA0 the central wavelength
            of a given line, DLAMBDA the line width, and MORE are 0 or
            more parameters which will be equal for all the lines.
     waxis= the wavelength axis for which the model will be computed.
     lines= the wavelengths of the spectral lines, in the same unit as WAXIS.
     light_speed= speed of light in the unit in which velocities are
            to be experessed. Defaults to 299792.458 (i.e. velocities
            are in km/s). For the sake of optimisation, light_speed is
            not stored as such in a data member. The corresponding data
            member is c_1=1./light_speed.
     relative= line number relative to which the other's flux is expressed

    METHODS
     eval   evaluate the model

    FUNCTIONS
     curvefit_func an curvefit-compatible wrapper around eval

    EXAMPLE
     x = span(2.15, 2.175, 100);
     obj = dopplerlines(new, waxis = x, lines=2.166120);
     y = obj(eval, [1.2, 25., 100.]) + random_n(100) * sigma;
     plg, y, x;
     a = [1., 0., 50.];
     curvefit, obj.curvefit_func, obj, a, y, deriv=1;
     write, format="Model parameters: [%e, %e, %e]\n", a(1), a(2), a(3);
     model = obj(eval, a);
     plg, color = "red", model, x;

    SEE ALSO:
     curvefit, dopplerlines.eval, dopplerlines.curvefit_func
    """

    def __init__(self, lines, waxis, profile=None, profile_jac=None,
                 light_speed=None, relative=None):

        if profile is None:
            self.profile = ngauss
        else:
            self.profile = profile

        if light_speed is None:
            self.light_speed = 299792.458  # default to km/s

        self.c_1 = 1/self.light_speed

        # declare lines np.zeros? et affectation
        # self.lines=np.asarray(lines)
        # np.array(y_0, ndmin=1, copy=False)
        # force ndim a 1 !!!
        self.lines = np.array(lines, ndmin=1, copy=False)
        self.nlines = self.lines.size

        self.waxis = np.asarray(waxis)

        # relative : index of the reference wavelength in the array
        # indice de la raie reference dans le tableau
        # TODO write test for special case
        if ((relative is not None) and (relative > lines.size or relative <= 0)):
            raise Exception("relative should be > 0 and <= numberof(lines)")
        else:
            self.relative = relative

        self.dbg = False
        if self.dbg:
            print(f"dop_init withi:")
            print(f"self.lines  {self.lines}")
            print(f"self.nlines  {self.nlines}")
            print(f"self.waxis {self.waxis}")

    def __call__(self,xdata ,*params):
        """
        model = obj(eval, a, grad, deriv=deriv)
        Evaluate model. OBJ is an instance of the DOPPLERLINES class.

        SEE ALSO: dopplerlines, oxy
        """
        if self.dbg:
            print("DBG CALL doppler line eval start __call__")
            print(f"type {type(self.nlines)}")
            print(f"self.nlines {self.nlines}")
            print(f"type(xdata) {type(xdata)}")
            print(f"call with xdata {xdata}")
            print(f"type(params) {type(params)}")
            print(f"call with params {params}")
            print(f"call with params.size {len(params)}")
            # print(f"call with a.size {a.size}")
            print(f"params {params}")
            # print(f"a[0] {a[0]}")
            # print(f"a[0][0] {a[0][0]}")

            # check indices correct
            # vel = a[nlines]
            print(f"self.nlines {self.nlines}")

        # TODO check modif a cause d ajout xdata
        # TODO X2 check modif a cause d ajout xdata
        vel = params[self.nlines]
        # vel = params[0]

        dvel = params[self.nlines+1]
        # dvel = params[1]
        # dvel = a[2]

        more = np.array([])
        # if (a.size > nlines+2):
        # if (a.size > nlines+2):
        if (len(params) > self.nlines+2):
            more = np.append(more, params[self.nlines+2:])

        if self.dbg:
            print(f"vel {vel}")
            print(f"dvel {dvel}")
            print(f"more {more}")

            print(f"self.lines {self.lines}")
            print(f"vel {vel}")
            print(f"c_1 {self.c_1}")

            print(f"self.lines.type {type(self.lines)}")
            print(f"vel.type {type(vel)}")

        # TODO correct with dimsmin ?
        lambda1 = np.zeros(self.lines.size)

        # np.asarray*float => return float
        lambda1b = np.zeros(self.lines.size)

        # print(f"lamda1.type {type(lambda1)}")
        # print(f"lambda1.shape {lambda1.shape}")
        # print(f"lambda1.size {lambda1.size}")
        # print(f"lambda1 {lambda1[0]}")

        lambda1 = self.lines * (1 + vel * self.c_1)
        lambda1 = np.multiply(np.asarray((1 + vel * self.c_1)),
                              self.lines, out=lambda1b)
        # print(f"lamda1.type {type(lambda1)}")
        # print(f"lambda1.shape {lambda1.shape}")
        # print(f"lambda1.size {lambda1.size}")
        # print(f"lambda1 {lambda1[0]}")

        widths = np.asarray(lambda1 * dvel * self.c_1)

        # print(f"widths {widths}")
        # print(f"widths0 {widths[0]}")
        # print f"{:.2e}".format(widths)
        # print(np.format_float_positional(widths,precision=6))

        model = np.zeros(self.waxis.shape)

        # print(f"model {model}")

        # TODO need to declare grad even if not deriv
        # if (deriv):
        # grad = np.zeros(self.waxis.shape, a.size)
        # grad = np.zeros((a.size, self.waxis.size), dtype=float)
        # print(f"dim grad {len(params)} {self.waxis.size}")
        # print(f"waxis shape is {self.waxis.shape}")
        # grad = np.zeros((len(a),self.waxis.size), dtype=float)
        grad = np.zeros((len(params), *self.waxis.shape), dtype=float)
        # print(f"grad {grad}")

        if self.dbg:
            print(f"params[0] {params[0]} lambda1[0] {lambda1[0]}")
            print(f"widths[0] {widths[0]} more {more}")

        # print(f"{a.size} {lambda1.size} {widths.size}
        # {more.size} {self.lines.size}")
            print(f"{len(params)} {lambda1.size} {widths.size}")
            print(f"{more.size} {self.lines.size}")

        aa = np.zeros((3+more.size))
        # aa = np.column_stack((a, lambda1, widths, more))
        # TODO voir code yorick bizarre
        for k in range(self.nlines):
            # print(f"k ---------{k}")
            # print(f"aa {aa}")
            # print(f"params[0] {params[0]}")
            if params[0].size > 1:
                aa[0] = params[0][k]
            else:
                aa[0] = params[k]
            aa[1] = lambda1[k]
            aa[2] = widths[k]
            if more.size > 0:
                aa[3:] = more

            # aa[k] = np.column_stack((a[k], lambda1, widths, more))
            # aa = np.concatenate(np.concatenate(a,lambda1),np.concatenate(widths,more))

            # print(f"Xaa {aa}")

            if ((self.relative is not None) and (k != self.relative)):
                aa[1] *= params[self.relative]

            # two func profile + profile_jac
            # model += self.profile(self.waxis, aa, agrad, deriv=deriv)
            # print(f"Yaa {aa}")
            # print(" ~~~ before gauss~~~")
            acc_model, t_agrad = self.profile(self.waxis, *aa)
            model += acc_model
            agrad = np.transpose(t_agrad, [1, 0])
            # TODO do we need deriv ?
            # if deriv:
                # t_agrad = self.profile_jac(self.waxis, *aa)
            agrad = np.transpose(t_agrad, [1, 0])
            # print(f"{agrad.shape}")
            # print(f"agrad  {agrad}")
            # plt.plot(self.waxis,agrad[0])
            # plt.show()

            if (self.relative is not None):
                # print("self.relative is not None")
                if (k == self.relative):
                    # print("k==self.relative")
                    grad[k, :] += agrad[0, :]
                    # print(f"grad  {grad}")
                else:
                    # print("k!=self.relative")
                    grad[self.relative, :] += params[k] * agrad[0, :]
                    # print(f"grad  {grad}")
                    grad[k, :] = params[self.relative] * agrad[0, :]
                    # print(f"grad  {grad}")

            else:
                # print(f"self.relative is None")
                # grad[:, k] = agrad[:, 1];
                grad[k, :] = agrad[0, :]
                # print(f"grad  {grad}")
                # plt.plot(self.waxis,grad[0])
                # plt.show()

            # print(f"grad shape {grad.shape}  lines shape {self.lines.shape}   agrad shape {agrad.shape}")
            # print(f"{grad[self.nlines, :]}")
            # print(f"{self.lines[0]}")
            # print(f"{np.asarray(self.lines)[0]}")
            grad[self.nlines, :] += np.asarray(self.lines)[k] * self.c_1 * agrad[1, :]

            # plt.plot(self.waxis,grad[nlines])
            # plt.show()

            # print(f"grad  {grad} (vel)")
            grad[self.nlines+1, :] += lambda1[k] * self.c_1 * agrad[2, :]
            # print(f"grad  {grad} (sigma)")

            # plt.plot(self.waxis,grad[nlines+1])
            # plt.show()

            # NOTE diff syntax pour le array(..
            # yorick
            # grad(.., nlines+3:) += agrad(.., 4:);
            # numpy
            # grad[nlines+2:,:] += agrad[3:,:]

            # if (more is not None):
            if (more.size > 0):
                # grad[:, nlines+2:] += agrad[:, 3:]
                grad[self.nlines+2:, :] += agrad[3:, :]
                # print(f"grad  {grad} (more)")
        # return np.transpose(grad, [1, 0])

            # print(f"model {model}")

        # print(f"model {model}")
        # print(f"model.shape {model.shape}")
        # plt.figure()
        # plt.plot(self.waxis,model)
        # plt.show()
        # print(f"type model {type(model)}")

        # print(f"type np.transpose(grad, [1, 0]) {type(np.transpose(grad, [1, 0]))}")

        return model, np.transpose(grad, [1, 0])

    def curvefit_func(self, xdata, *params):
        """
        model = dopplerlines.curvefit_func(self,xdata,*params)

        curvefit-compatible wrapper around:
        model = obj(eval, a, grad, deriv=deriv)
        where OBJ is an instance of the DOPPLERLINES OXY class.

        SEE ALSO:
        curvefit, dopplerlines, dopplerlines.eval, oxy.

        """
        if self.dbg:
            print("DBG CALL curvefit_func")
            print(f"xdata is {xdata}")
            print(f"params is {params}")
            print(f"self.waxis is {self.waxis}")
        # print(f"x is {x}")
        # if not (xdata==self.waxis).all():
        #    print(f"xdata {xdata}")
        #    raise Exception("Bad input")
        # print(f"xdata.shape is {xdata}")

        return self(xdata, *params)[0]

    def curvefit_jac(self, xdata, *params):
        """
        model = dopplerlines.curvefit_jac(self, xdata, *params)

        curvefit-compatible wrapper around:
        model = obj(eval, a, grad, deriv=deriv)
        where OBJ is an instance of the DOPPLERLINES OXY class.

        SEE ALSO:
        curvefit, dopplerlines, dopplerlines.eval, oxy.
        """
        if self.dbg:
            print(f"xdata is {xdata}")
            print(f"self.waxis is {self.waxis}")
        if not (xdata == self.waxis).all():
            raise Exception("Bad input")
        return self(xdata, *params)[1]


def test():
    """
    EXAMPLE
        x = span(2.15, 2.175, 100);
        obj = dopplerlines(new, waxis = x, lines=2.166120);
        y = obj(eval, [1.2, 25., 100.]) + random_n(100) * sigma;
        plg, y, x;
        a = [1., 0., 50.];
        curvefit, obj.curvefit_func, obj, a, y, deriv=1;
        write, format="Model parameters: [%e, %e, %e]\n", a(1), a(2), a(3);
        model = obj(eval, a);
        plg, color = "red", model, x;
    """
    print("testing dopplerlines module")

    sigma = 20.5

    # first test
    print("# first test")
    lines = 2.166120
    waxis = np.linspace(2.15, 2.175, 100)
    dop = dopplerlines(lines, waxis)
    print("after init")
    a = np.array([1.2, 25., 100.])
    y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
    # y=dop(*a) + np.random.standard_normal(100) * sigma
    # print("ok will change")

    # print(f"y.shape {y.shape}")
    print(f"---- y {y}")
    print(f"---- y[0] {y[0]}")
    # print(f"*y {y}")
    # plt.figure()
    # plt.plot(waxis,y)
    # plt.show()

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
    print("# second test two lines")
    lines = (2.166120, 2.155)
    waxis = np.linspace(2.15, 2.175, 100)
    dop = dopplerlines(lines, waxis)
    a = np.array([1.2, 0.5, 25., 100.])
    # y=dop(*a) + np.random.standard_normal(100) * sigma
    y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma

    print("===FIT 2==========")
    a0 = np.array([1., 0.3, 50., 50.])
    resopt, reqcov = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0, jac=dop.curvefit_jac)
    resopt2, reqcov2 = optimize.curve_fit(dop.curvefit_func, waxis, y, p0=a0)

    model = dop(waxis, *resopt)[0]
    model2 = dop(waxis, *resopt2)[0]
    chi2 = np.sum(((y-model)/sigma)**2)/(y.size-a.size+1)
    chi22 = np.sum(((y-model2)/sigma)**2)/(y.size-a.size+1)

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
    print("# third test two lines and more parameter")
    lines = (2.166120, 2.155)
    waxis = np.linspace(2.15, 2.175, 100)
    dop = dopplerlines(lines, waxis)
    a = np.array([1.2, 0.5, 25., 100., 1.])
    # y=dop(*a) + np.random.standard_normal(100) * sigma
    y = dop(waxis, *a)[0] + np.random.standard_normal(100) * sigma
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
    test()
