import sys
from .common import *
from cubefit.dopplerlines import DopplerLines
from cubefit.cubemodel import CubeModel
from cubefit.ngauss import gauss, ngauss

def write_fits(cube, cname):
    fit_hdu = fits.PrimaryHDU(np.transpose(cube,[2,0,1]))
    #fit_hdu = fits.PrimaryHDU(cube)
    fit_hdul = fits.HDUList([fit_hdu])
    fit_hdul.writeto(cname, overwrite=True)


def open_fits(cube, cname):
    fit_hdul = fits.open(cname)
    fit_hdul.info()
    fit_hdr = fit_hdul[0].header
    list(fit_hdr.keys())

    cube = fit_hdul[0].data
    print("open_fits hdul0.data--")
    print(type(cube))
    print(cube.ndim)
    print(cube.shape)



def print_array_info(array):
    print(f'{array=}'.split('=')[0])
    print(f"array is a ")
    print(type(array))
    print(array.shape)
    print(array.ndim)
    print("---")




def print_array_slice(myarray):
    size=min(myarray.shape)//2
    print(f"myarray size-1 {myarray[size-1,size-1,:]}")
    print(f"myarray size {myarray[size,size,:]}")
    print(f"myarray size+1 {myarray[size+1,size+1,:]}")


def plot_array_slice(myarray):
    #print(myarray.shape)
    size=min(myarray.shape)//2
    plt.figure()
    plt.imshow(myarray[size, size, :], cmap='gray')
    plt.colorbar()
    plt.show()

def add_noise(cube, sigma=0.02):
    '''Add noise to data.

    This method always uses the same pseudo-random sequence as it is
    intended for use in a reproduceable test suite. Don't use it for
    Monte-Carlo simulations or such.

    '''
    # instanciate a random number generator with fixed seed
    # warning: changing the seed may affect the success of certain tests below
    rng = np.random.default_rng(3)
    psigma = sigma
    tmp_cube = np.copy(cube)
    tmp_cube = cube + rng.standard_normal(cube.shape) * psigma
    return tmp_cube


class TestCubemodel(unittest.TestCase):
    '''UnitTest class to test gauss function
    '''
    def check_gradient(self, f, x, epsilon=1e-6, reltol=1e-3, diftol=None, diflim=None):
        if diflim is None:
            diflim=np.min(epsilon);
        if diftol is None:
            diftol=diflim*reltol;
        d = x.shape
        if np.isscalar(epsilon):
            epsilon=np.ones(d[2])*epsilon
        g = np.zeros(d)
        f0, g0 = f(x)
        for k in range(d[2]):
            for j in range(d[1]):
                for i in range(d[0]):
                    temp = np.copy(x)
                    temp[i, j, k] += 0.5*epsilon[k]
                    fp, gp=f(temp)
                    temp[i, j, k] -= epsilon[k]
                    fm, gm=f(temp)
                    # this is (f(x+h/2)-f(x-h/2))/h
                    g[i, j, k] = (fp-fm)/epsilon[k]
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


    def test_cubemodel_eval_no_data_raises(self):
        model=CubeModel()
        x="whatever"
        with self.assertRaises(ValueError, msg="CubeModel.eval() should raise a Value Error when data is None"):
            model.eval(x)

    def test_cubemodel_gradient(self):
        nx, ny, nz = 5, 5, 433

        xreal_1d = np.array([1.2, 0.5, 25., 100])
        nterms = len(xreal_1d)
        xreal = np.zeros((nx, ny, xreal_1d.size))
        for k in range(nterms):
            xreal[:, :, k]=xreal_1d[k]

        waxis = np.linspace(2.15e-6, 2.175e-6, nz)

        profile = DopplerLines(2.166120e-6, waxis, profile=gauss)
        model = CubeModel(profile=profile, profile_xdata=waxis)
        cube_real = model.model(xreal)
        model.data = cube_real

        xtest_1d = np.array([1.0, 0.6, 50., 120])
        nterms = len(xreal_1d)
        xtest = np.zeros((nx, ny, xtest_1d.size))
        for k in range(nterms):
            xtest[:, :, k]=xtest_1d[k]

        self.check_gradient(model.eval, xtest, epsilon=[1e-2, 1e3, 1., 1.], diftol=1e-2)

    def test_cubemodel_gauss(self, dbg=False):
        # debug model
        if dbg:
            print("starting cubemodel test gauss....")

        # create a cube
        nx, ny, nz = 5, 5, 433
        nparms = 5

        # choose parameter for model

        xreal_1d = np.array([1,1, 0.5, 0.5,0.1])
        nterms = xreal_1d.size
        profile = gauss
        profile_xdata = np.linspace(-10,10,nz)
        sigma=0.02

        rng = np.random.default_rng(3)

        xreal = np.zeros((nx,ny,nterms))
        for i in range(nx):
            for j in range(ny):
                xreal[i,j,:] = xreal_1d

        if dbg:
            print(f" xreal[4,4,:]{xreal[4,4,:]}")

        # create model object
        if dbg:
            print("creating fit obj")
        model = CubeModel(profile=profile, profile_xdata=profile_xdata)

        # create "true" cube using model
        if dbg:
            print("create model cube...")
        cube_real = model.model(xreal)

        # add noise to get mock observational data
        data = add_noise(cube_real, sigma)

        # write fits cube
        if dbg:
            write_fits(cube_real, "mycube_model_gauss.fits")
            write_fits(data, "mycube_gauss.fits")
            print("check cube")
            print(f"{cube_real[4,4,:]}")
            print(f"============#debug eval")

        # Attach data to model object
        model.data=data

        # create initial guess
        x0 = xreal

        # tiré de l exemple
        # besoin fonction objectif (param 1)  gradient (param 2)
        # (res_x, fx, gx, status) = vmlmb(lambda x: (f(1, x), f(2, x)), x0,
        #            mem=x0.size , blmvm=False, fmin=0, verb=1, output=sys.stdout)

        (res_x, fx, gx, status) = model.fit(x0)

        # build best model cube
        model_cube=model.model(res_x)
        # compute reduced chi2
        chi2=np.sum(((data-model_cube)/sigma)**2)/(data.size-res_x.size)
        # raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)

        # write_fits(res_x, "mycube_res_x_gauss.fits")
        # write_fits(fx, "mycube_res_fx_dop.fits")
        # write_fits(gx, "mycube_res_gx_dop.fits")
        if dbg:
            print(f"fx {fx}")
            print(f"gx {gx}")
            print(f"status {status}")

    def test_cubemodel_fit(self, dbg=False):
        """
        test_cubemodel_fit
        implementation of the fit function test
        """
        # debug model
        if dbg:
            print("starting cubemodel test_fit....")
        # create a cube
        nx = 5
        ny = 5
        nz = 433

        if dbg:
            print("create model ...")

        cube_doppler = np.ndarray((nx, ny, nz))
        cube_noise_doppler = np.ndarray((nx, ny, nz))
        weight = np.ones((nx, ny, nz))
        if dbg:
            print("test doppler ...")
        # nlines = 1
        lines = np.array([2.166120])
        doppler_param = np.array([1.2, 0.5, 25., 100])
        doppler_xdata = np.linspace(2.15, 2.175, nz)

        if dbg:
            print("creating line obj")
        lineobj_doppler = DopplerLines(lines, doppler_xdata, profile=ngauss)

        sigma = 0.2

        model_param_doppler = np.zeros((nx, ny, doppler_param.size))

        # TODO find a more pythonic expression
        for i in range(doppler_param.size):
            model_param_doppler[:, :, i] = doppler_param[i]

        # TODO choose return tuple or not
        fcn_doppler = lineobj_doppler.__call__
        fcn_x_doppler = doppler_xdata

        if dbg:
            print("creating fit obj")
        # create fit obj
        fitobj_doppler_model = CubeModel(cube_doppler, fcn_doppler,
                                       fcn_x_doppler, weight)
        # model
        if dbg:
            print("compute model ...")
        cube_model_doppler = fitobj_doppler_model.model(model_param_doppler)
        # add noise
        cube_noise_doppler = add_noise(cube_model_doppler, sigma)

        if dbg:
            write_fits(cube_model_doppler, "mycube_model_doppler_fit.fits")
            write_fits(cube_noise_doppler, "mycube_doppler_fit.fits")

        #print("check fits")
        #open_fits(cube_model_doppler, "mycube_model_doppler_fit.fits")
        # print("diff noise - model")
        # print(f"{cube_noise_doppler[49,49,:] - cube_model_doppler[49,49,:]}")

        if dbg:
            print("check cube")
            print(f"{cube_model_doppler[4,4,:]}")

        # debug view
        # fitobj_doppler.view(cube_model_doppler)

        # debug eval
        if dbg:
            print(f"============#debug eval")

        # TODO nz=433
        cube_empty = np.ndarray((nx, ny, nz))

        # on recree un obj cubemodel
        fitobj_eval_doppler = CubeModel(cube_noise_doppler,
                                      fcn_doppler, fcn_x_doppler, weight)

        doppler_param_test = np.array([1.1, 1., 25., 100])
        #doppler_param_test = np.array([1.2, 0.5, 25., 100])
        model_param_doppler_test = np.zeros((nx, ny, doppler_param_test.size))

        # TODO find a more pythonic expression
        for i in range(doppler_param_test.size):
            model_param_doppler_test[:, :, i] = doppler_param_test[i]


        # calcul du point de depart x0 les cartes de parametres initiales
        # x0 = cube_noise_doppler
        x0_test = model_param_doppler_test

        if dbg:
            print(f"x0_test {x0_test}")
            print(f"calling fit function")

        (res_x, fx, gx, status) = fitobj_eval_doppler.fit(x0_test, fmin=0.)

        # build best model cube
        cube_model=fitobj_eval_doppler.model(res_x)
        # compute reduced chi2
        chi2=np.sum(((cube_noise_doppler-cube_model)/sigma)**2)/(cube_noise_doppler.size-res_x.size)
        # raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)


        if dbg:
            print("# Iter.   Time (ms)   Eval. Reject.    Obj. Func.      Grad.  Step")
            print("res_x for test_fit")
            print(f"{res_x}")
            print(f"fx {fx}")
            print(f"gx {gx}")
            print(f"status {status}")

    def test_cubemodel(self, dbg=False):
        """
        test_cubemodel
        #fitobj = cubemodel(new, cube = cubl(,,indlines), weight = noise,
        #                 fcn = lineobj.lmfit_func, fcn_x= lineobj,
        #                 scale=scale, delta=delta, regularisation=cubemodel.l1l2,
        #                 pscale=pscale, poffset=poffset, ptweak=myvlsr2vobs)
            'cube', 'weight', 'fcn', 'fcn_x',
            'scale', 'delta', 'pscale', 'poffset', and 'ptweak'
        """
        # scaling parameters
        # scale = np.array([1., 1., 1])
        # print(f"scale {scale}")
        # delta   =             [ 1.   ,   1. ,   1.   ]
        # delta = 1./scale
        # print(f"delta {delta}")
        # poffset
        # poffset = np.array([0., 0, 0])
        # print(f"poffset {poffset}")
        # pscale
        # pscale = np.array([5e-5, 100., 40.])

        # debug model
        if dbg:
            print("starting cubemodel test....")
        # create a cube
        nx = 5
        ny = 5
        nz = 433

        if dbg:
            print("create model ...")
        weight = np.ones((nx, ny, nz))

        cube_doppler = np.ndarray((nx, ny, nz))
        cube_noise_doppler = np.ndarray((nx, ny, nz))

        if dbg:
            print("test doppler ...")
        # choose parameter for model
        # nlines = 1
        lines = np.array([2.166120])
        # lines=np.array([2.166120,2.155])

        doppler_param = np.array([1.2, 0.5, 25., 100])
        doppler_xdata = np.linspace(2.15, 2.175, nz)

        if dbg:
            print("creating line obj")
        lineobj_doppler = DopplerLines(lines, doppler_xdata, profile=ngauss)

        if dbg:
            print("after doppler init")
        sigma = 0.5
        if dbg:
            print("test doppler call")
        # instanciate a random number generator with fixed seed
        # warning: changing the seed may affect the success of certain tests below
        rng = np.random.default_rng(3)

        lineobj_doppler(doppler_xdata, *doppler_param)[0] \
            + rng.standard_normal(nz) * sigma

        model_param_doppler = np.zeros((nx, ny, doppler_param.size))

        # print(f"doppler_param is {doppler_param}")

        # TODO find a more pythonic expression
        for i in range(doppler_param.size):
            model_param_doppler[:, :, i] = doppler_param[i]

        # print("creating line obj")
        # lineobj_doppler = DopplerLines(lines=lines, waxis=doppler_xdata,
        #                               profile=ngauss)

        # TODO choose return tuple or not
        # fcn_doppler = lineobj_doppler.curvefit_func
        fcn_doppler = lineobj_doppler.__call__
        # fcn_x=a0
        # important
        fcn_x_doppler = doppler_xdata

        if dbg:
            print("creating fit obj")
        # create fit obj
        fitobj_doppler_model = CubeModel(cube_doppler, fcn_doppler,
                                       fcn_x_doppler, weight)
        # print(f"cube shape {cube.shape}")
        # print(f"cube_noise shape {cube_noise.shape}")

        # test more parameters
        # fitobj=cubemodel(cube,cube_noise,fcn,fcn_x,scale, delta,
        #            pscale,poffset, ptweak,regularisation=None,decorrelate=None)

        # model
        # cube_model=fitobj.model(cube,a0)
        if dbg:
            print("compute model ...")
        cube_model_doppler = fitobj_doppler_model.model(model_param_doppler)

        # add noise

        cube_noise_doppler = add_noise(cube_model_doppler, sigma)

        # print_array_slice(cube_noise_doppler)
        # print_array_slice(cube_noise_gauss)

        if dbg:
            write_fits(cube_model_doppler, "mycube_model_doppler.fits")
            write_fits(cube_noise_doppler, "mycube_doppler.fits")

        #print("check fits")
        #open_fits(cube_model_doppler, "mycube_model_doppler.fits")
        # print("diff noise - model")
        # print(f"{cube_noise_doppler[49,49,:] - cube_model_doppler[49,49,:]}")

        if dbg:
            print("check cube")
            print(f"{cube_model_doppler[4,4,:]}")

        # debug view
        # fitobj_doppler.view(cube_model_doppler)

        # debug eval
        if dbg:
            print(f"============#debug eval")

        # TODO nz=433
        cube_empty = np.ndarray((nx, ny, nz))

        # on recree un obj cubemodel
        fitobj_eval_doppler = CubeModel(cube_noise_doppler,
                                      fcn_doppler, fcn_x_doppler, weight)


        # tiré de l exemple
        # f = eval(f"fitobj_eval.eval")
        # x0 = f(0, n=n, factor=factor)

        # calcul du point de depart x0 les cartes de parametres initiales
        # x0 = cube_noise_doppler
        x0 = model_param_doppler


        if dbg:
            print(f"calling vmlmb")
        #def vmlmb_printer(output, iters, evals, rejects, t, x, fx, gx, pgnorm, alpha, fg):
        # print("# Iter.   Time (ms)    Eval. Reject.       Obj. Func.         Grad.       Step")
        # ---------------------------------------------------------------------------------
        # besoin fonction objectif (param 1)  gradient (param 2)
        # (res_x, fx, gx, status) = vmlmb(lambda x: (f(1, x), f(2, x)), x0,
        #         mem=x0.size , blmvm=False, fmin=0, verb=1, output=sys.stdout)
        print()
        (res_x, fx, gx, status) = fitobj_eval_doppler.fit(x0)

        # build best model cube
        cube_model=fitobj_eval_doppler.model(res_x)
        # compute reduced chi2
        chi2=np.sum(((cube_noise_doppler-cube_model)/sigma)**2)/(cube_noise_doppler.size-res_x.size)
        # raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)

        # print("# Iter.   Time (ms)    Eval. Reject.       Obj. Func.         Grad.       Step")
        if dbg:
            print("res_x for dop")
            print(f"{res_x}")
        #write_fits(res_x, "mycube_res_x_dop.fits")
        #write_fits(fx, "mycube_res_fx_dop.fits")
        #write_fits(gx, "mycube_res_gx_dop.fits")
        if dbg:
            print(f"fx {fx}")
            print(f"gx {gx}")
            print(f"status {status}")

        # def test(probs = range(1, 19),  n=None, factor=None, fmin=0,
        #    mem="max", verb=1, blmvm=False, output=sys.stdout, **kwds)
        # test(verb=1, gtol=0, xtol=0, ftol=0, fmin=0, mem="max")

        # from test_fit-Brg4
        # xl1 = fitobj.fit( xl, verb=1, xmin=xmin, xmax=xmax);


if __name__ == '__main__':
   unittest.main()
