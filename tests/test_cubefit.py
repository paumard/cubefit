import sys
from .common import *
from cubefit.dopplerlines import DopplerLines
from cubefit.cubefit import CubeFit, vmlmb
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

        self.check_gradient(fitobj.eval, xreal, epsilon=1e-4)

    def test_cubefit_gauss(self, dbg=False):
        # debug model
        if dbg:
            print("starting cubefit test gauss....")

        # create a cube
        nx = 5
        ny = 5
        nz = 433
        nparms = 5

        weight=np.ones((nx,ny,nz))

        cube_gauss = np.ndarray((nx, ny, nz))
        cube_noise_gauss = np.ndarray((nx, ny, nz))

        # choose parameter for model

        gauss_param = np.array([1,1, 0.5, 0.5,0.1])
        #np.repeat(gauss_param,433,axis=0)
        #np.tile(gauss_param,(433,1))
        #gauss_param = np.full((433,5),[1,1, 0.5, 0.5,0.1])
        # gauss_xdata = np.linspace(-10,10,3000)
        gauss_xdata = np.linspace(-10,10,nz)

        sigma=0.02

        if dbg:
            print("test gauss call")
        y = gauss(gauss_xdata,*gauss_param)[0]
        # instanciate a random number generator with fixed seed
        # warning: changing the seed may affect the success of certain tests below
        rng = np.random.default_rng(3)
        y_noise = y + rng.standard_normal(y.size) * sigma

        #plt.figure()
        #plt.plot(gauss_xdata, y)
        #plt.show()


        model_param_gauss = np.zeros((nx,ny,nparms))

        for i in range(nx):
            for j in range(ny):
                model_param_gauss[i,j,:] = gauss_param

        if dbg:
            print(f" model_param_gauss[4,4,:]{model_param_gauss[4,4,:]}")

        fcn_gauss = gauss
        fcn_x_gauss = gauss_xdata

        # cube_gauss[...,0:nz] = gauss_param
        # np.c_[cube_gauss,gauss_param]

        if dbg:
            print("creating line obj")
        lineobj_gauss=DopplerLines(lines=gauss_param[0], waxis=gauss_xdata,
                                   profile=ngauss)

        # create fit obj
        if dbg:
            print("creating fit obj")
        fitobj_gauss = CubeFit(cube_gauss, fcn_gauss, fcn_x_gauss, weight)

        # model
        # cube_model=fitobj.model(cube,a0)
        if dbg:
            print("create model ...")
        cube_model_gauss = fitobj_gauss.model(model_param_gauss)

        # add noise
        cube_noise_gauss = add_noise(cube_model_gauss, sigma)

        # write fits cube
        if dbg:
            write_fits(cube_model_gauss, "mycube_model_gauss.fits")
            write_fits(cube_noise_gauss, "mycube_gauss.fits")

        # print("check fits")
        # open_fits(cube_model_gauss, "mycube_model_gauss.fits")

        if dbg:
            print("check cube")
            print(f"{cube_model_gauss[4,4,:]}")

        # debug eval
        if dbg:
            print(f"============#debug eval")

        # TODO nz=433
        # cube_empty = np.ndarray((nx, ny, nz))

        # on recree un obj cubefit
        fitobj_eval_gauss = CubeFit(cube_noise_gauss,
                                    fcn_gauss, fcn_x_gauss, weight)

        # calcul du point de depart x0 les cartes de parametres initiales
        x0 = model_param_gauss
        # x0 = gauss_param

        # tiré de l exemple
        # besoin fonction objectif (param 1)  gradient (param 2)
        # (res_x, fx, gx, status) = vmlmb(lambda x: (f(1, x), f(2, x)), x0,
        #            mem=x0.size , blmvm=False, fmin=0, verb=1, output=sys.stdout)

        (res_x, fx, gx, status) = vmlmb(fitobj_eval_gauss.eval, x0, mem=x0.size,
                                        blmvm=False, fmin=0, verb=1,
                                        output=sys.stdout)

        # write_fits(res_x, "mycube_res_x_gauss.fits")
        # write_fits(fx, "mycube_res_fx_dop.fits")
        # write_fits(gx, "mycube_res_gx_dop.fits")
        if dbg:
            print(f"fx {fx}")
            print(f"gx {gx}")
            print(f"status {status}")

    def test_cubefit_fit(self, dbg=False):
        """
        implementation of the fit function test
        """
        # debug model
        if dbg:
            print("starting cubefit test_fit....")
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
        fitobj_doppler_model = CubeFit(cube_doppler, fcn_doppler,
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

        # on recree un obj cubefit
        fitobj_eval_doppler = CubeFit(cube_noise_doppler,
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

        (res_x, fx, gx, status) = fitobj_eval_doppler.fit( x0_test)

        if dbg:
            print("# Iter.   Time (ms)   Eval. Reject.    Obj. Func.      Grad.  Step")
            print("res_x for test_fit")
            print(f"{res_x}")
            print(f"fx {fx}")
            print(f"gx {gx}")
            print(f"status {status}")

    def test_cubefit(self, dbg=False):
        """
        EXAMPLE
        #fitobj = cubefit(new, cube = cubl(,,indlines), weight = noise,
        #                 fcn = lineobj.lmfit_func, fcn_x= lineobj,
        #                 scale=scale, delta=delta, regularisation=cubefit.l1l2,
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
            print("starting cubefit test....")
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
        fitobj_doppler_model = CubeFit(cube_doppler, fcn_doppler,
                                       fcn_x_doppler, weight)
        # print(f"cube shape {cube.shape}")
        # print(f"cube_noise shape {cube_noise.shape}")

        # test more parameters
        # fitobj=cubefit(cube,cube_noise,fcn,fcn_x,scale, delta,
        #            pscale,poffset, ptweak,regularisation=None,decorrelate=None)

        # model
        # cube_model=fitobj.model(cube,a0)
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

        # on recree un obj cubefit
        fitobj_eval_doppler = CubeFit(cube_noise_doppler,
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
        (res_x, fx, gx, status) = vmlmb(fitobj_eval_doppler.eval, x0, mem=x0.size,
                                    blmvm=False, fmin=0,
                                    verb=1, output=sys.stdout)

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
