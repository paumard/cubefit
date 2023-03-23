# import sys
import os
from .common import *
from cubefit.dopplerlines import DopplerLines
from cubefit.cubemodel import CubeModel , markov
from cubefit.lineprofiles import gauss, ngauss

DEBUG = os.environ.get("TEST_CUBEMODEL_DEBUG")
if DEBUG:
    from matplotlib import pyplot as plt


def write_fits(cube, cname):
    fit_hdu = fits.PrimaryHDU(np.transpose(cube, [2, 0, 1]))
    # fit_hdu = fits.PrimaryHDU(cube)
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
    def check_gradient(self, f, x, epsilon=1e-6, reltol=1e-3, diftol=None, diflim=None, **kwargs):
        if diflim is None:
            diflim=np.min(epsilon);
        if diftol is None:
            diftol=diflim*reltol;
        d = x.shape
        if np.isscalar(epsilon):
            epsilon=np.ones(d[2])*epsilon
        g = np.zeros(d)
        f0, g0 = f(x, **kwargs)
        for k in range(d[2]):
            for j in range(d[1]):
                for i in range(d[0]):
                    temp = np.copy(x)
                    temp[i, j, k] += 0.5*epsilon[k]
                    fp, gp=f(temp, **kwargs)
                    temp[i, j, k] -= epsilon[k]
                    fm, gm=f(temp, **kwargs)
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
        '''Check that CubeModel.eval raises an error when self.data is not set
        '''
        model=CubeModel()
        x="whatever"
        with self.assertRaises(ValueError, msg="CubeModel.eval() should raise a Value Error when data is None"):
            model.eval(x)

    def test_cubemodel_gradient(self):
        '''Check that CubeModel.eval evaluates the gradient correctly
        '''
        nx, ny, nz = 5, 5, 433

        xreal_1d = [1.2, 0.5, 25., 100]
        xreal = np.full((nx, ny, len(xreal_1d)), xreal_1d)

        waxis = np.linspace(2.15e-6, 2.175e-6, nz)

        profile = DopplerLines(2.166120e-6, profile=gauss)
        model = CubeModel(profile=profile, profile_xdata=waxis)
        cube_real = model.model(xreal)
        model.data = cube_real

        xtest_1d = np.array([1.0, 0.6, 50., 120])
        xtest = np.full((nx, ny, len(xtest_1d)), xtest_1d)

        self.check_gradient(model.eval, xtest, epsilon=[1e-2, 1e3, 1., 1.], diftol=1e-2)

    def helper_cubemodel_fit(self, model, shape, xreal, xtest, sigma, **fit_kwargs):
        """Helper for CubeModel.fit tests

        This does the following:
        1- use model.model() to build a cube of dimensions given by shape
        according to parameters xreal;

        2- add noise with standard deviation sigma to obtain mock
        observational data and attach them as model.data;

        3- run fitres = model.fit(xtest);

        4- compute reduced chi2 of fitted model.

        Parameters
        ----------
        model : cubefit.cubemodel.CubeModel
            The model to test.
        shape : tuple
            Shape of the data cube to generate.
        xreal : array_like
            Model parameters to produce "true" cube, can have one or
            three dimensions. If 3D, used as-is. If 1D, reproduced for
            each pixel of the field.
        xtest : array_like
            Initial guess to fit model, can have one or three
            dimensions. If 3D, used as-is. If 1D, reproduced for each
            pixel of the field.
        sigma
            Standard deviation of noise to add to "true" cube to get
            mock observational data.
        **fit_kwargs : dict, optional
            Any additional keyword argument is passed to model.fit().

        Returns
        -------
        real number
            reduced chi2 of fit
        tuple
            return value of model.fit()

        Raises
        ------
        ValueError
            when an inconsistency is detected between the value of
            parameter shape and the shape of parameters xreal and
            xtest

        """
        # get data dimensions
        nx, ny, nz = shape

        # Check whether a vector or stack was provided for xreal.
        # If a vector, build the stack.
        # If a stack, check dimensions.
        if len(np.shape(xreal)) == 1:
            nterms = len(xreal)
            xreal_1d = xreal
            xreal = np.full((nx, ny, nterms), xreal_1d)
        elif len(np.shape(xreal)) == 3:
            nterms = np.shape(xreal)[2]
            if np.shape(xreal)[0] != nx or np.shape(xreal)[1] != ny:
                raise ValueError("inconsistency between np.shape(xreal) "
                                 "and shape")
        else:
            raise ValueError("xreal has wrong shape")

        # Same for xtest
        if len(np.shape(xtest)) == 1:
            nterms = len(xtest)
            xtest_1d = np.asarray(xtest)
            xtest = np.full((nx, ny, nterms), xtest_1d)
        elif len(np.shape(xtest)) == 3:
            nterms = np.shape(xtest)[2]
            if np.shape(xtest)[0] != nx or np.shape(xtest)[1] != ny:
                raise ValueError("inconsistency between xtest.shape and shape")
        else:
            raise ValueError("xtest has wrong shape")

        # create "true" cube using model
        cube_real = model.model(xreal)

        # add noise to get mock observational data
        data = add_noise(cube_real, sigma)

        # Attach data to model object
        model.data=data

        # perform fit
        fitres = model.fit(xtest, **fit_kwargs)

        # build best model cube
        res_x=fitres[0]
        model_cube=model.model(res_x)
        # compute reduced chi2
        chi2=np.sum(((data-model_cube)/sigma)**2)/(data.size-res_x.size)

        # return chi2 and fit results
        return (chi2, fitres)

    def test_cubemodel_fit_gauss(self):
        '''Check that CubeModel.fit succeeds (flavor: gauss)
        '''
        # Shape of data cube (nx, ny, nz)
        nx, ny, nz = (5, 5, 433)
        # Model we want to test
        model = CubeModel(profile=gauss, profile_xdata=np.linspace(-10,10,nz), regularization=None)
        # Parameters for "true" cube. Can be 1D or 3D.
        xreal_1d = (1,1, 0.5, 0.5,0.1)
        # Initial guess for fit. Can be 1D or 3D.
        xtest_1d = xreal_1d
        # Sigma of errors to add to "true" cube to get "observational" data
        sigma=0.02

        # Call helper
        chi2, testres = self.helper_cubemodel_fit(model, (nx, ny, nz), xreal_1d, xtest_1d, sigma)

        # At this stage, perform some verifications on chi2 and/testres.
        # For instance: raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)

    def test_cubemodel_fit_dopplerlines(self):
        '''Check that CubeModel.fit succeeds (flavor: dopplerlines)
        '''
        # Shape of data cube (nx, ny, nz)
        nx, ny, nz = 5, 4, 433

        # Model we want to test
        profile = DopplerLines(2.166120, profile=ngauss)
        profile_xdata = np.linspace(2.15, 2.175, nz)
        model = CubeModel(profile=profile, profile_xdata=profile_xdata, regularization=None)

        # Parameters for "true" cube. Can be 1D or 3D.
        xreal_1d=(1.2, 0.5, 25., 100)

        # Initial guess for fit. Can be 1D or 3D.
        xtest_1d = (1.1, 1., 25., 100)

        # Sigma of errors to add to "true" cube to get "observational" data
        sigma = 0.2

        # Call helper
        chi2, testres = self.helper_cubemodel_fit(model, (nx, ny, nz), xreal_1d, xtest_1d, sigma, fmin=0.)

        # At this stage, perform some verifications on chi2 and/testres.
        # raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)

    def test_cubemodel_fit_dopplerlines2(self):
        '''Check that CubeModel.fit succeeds (flavor: dopplerlines2)
        '''
        # Shape of data cube (nx, ny, nz)
        nx, ny, nz = 5, 4, 433

        # Model we want to test
        profile = DopplerLines(2.166120, profile=ngauss)
        profile_xdata = np.linspace(2.15, 2.175, nz)
        model = CubeModel(profile=profile, profile_xdata=profile_xdata, regularization=None)

        # Parameters for "true" cube. Can be 1D or 3D.
        xreal_1d=(1.2, 0.5, 25., 100)

        # Initial guess for fit. Can be 1D or 3D.
        xtest_1d = xreal_1d

        # Sigma of errors to add to "true" cube to get "observational" data
        sigma = 0.5

        # Call helper
        chi2, testres = self.helper_cubemodel_fit(model, (nx, ny, nz), xreal_1d, xtest_1d, sigma)

        # At this stage, perform some verifications on chi2 and/testres.
        # raise error is chi2 not close to 1
        self.assertAlmostEqual(chi2, 1, places=1)

    def test_cubemodel_pscale_poffset_ptweak(self):
        '''Check that CubeFit.eval() and model() use pscale, poffset and ptweak
        '''
        nx, ny, nz = 5, 5, 101

        xreal_1d = np.array([1.2, 2.1701e-6, 1.9e-10])
        xtest_1d = np.array([1.1, 2.1703e-6, 2.5e-10])
        xreal = np.full((nx, ny, len(xreal_1d)), xreal_1d)
        xtest = np.full((nx, ny, len(xtest_1d)), xtest_1d)

        waxis = np.linspace(2.16e-6, 2.18e-6, nz)

        poffset=np.asarray([1., 2.17e-6, 2e-10])
        pscale =np.asarray([1e-1, 1e-10, 1e-11])
        # Also test that auto-broadcasting works like explicit
        xreal_normed=(xreal-poffset)/pscale
        xtest_normed=(xtest-poffset[np.newaxis, np.newaxis, :])/pscale[np.newaxis, np.newaxis, :]

        profile = gauss
        model = CubeModel(profile=profile, profile_xdata=waxis, regularization=None)
        model.data = model.model(xreal)
        val = model.criterion(xtest)


        # Check that model() behaves correctly with pscale and poffset not set
        self.assertEqual((np.max(np.abs(model.data-model.model(xreal, noscale=True)))), 0)
        self.assertEqual((np.max(np.abs(model.data-model.model(xreal, noscale=False)))), 0)

        # Set pscale and poffset
        model.pscale  = pscale
        model.poffset = poffset

        # Check that eval() behaves correctly with pscale and poffset set
        self.assertEqual(model.criterion(xreal, noscale=True), 0,
                          "Criterion for xreal_normed should be 0")
        self.assertEqual(model.criterion(xreal_normed), 0,
                          "Criterion for xreal_normed should be 0")
        self.assertEqual(model.criterion(xtest, noscale=True), val,
                          f"Criterion for xtest_normed should be {val}")
        self.assertEqual(model.criterion(xtest_normed), val,
                          f"Criterion for xtest_normed should be {val}")

        self.check_gradient(model.eval, xtest_normed)

        # Check that model() behaves correctly with pscale and poffset set
        self.assertEqual((np.max(np.abs(model.data-model.model(xreal, noscale=True)))), 0)
        self.assertEqual((np.max(np.abs(model.data-model.model(xreal_normed, noscale=False)))), 0)

        # Create a non-trivial ptweak function
        def ptweak(params):
            # Derivatives for all unchanged parameters and parameters
            # changed by a constant amount is 1. So let's initialize
            # all derivatives to 1.
            derivs=np.ones(params.shape)
            params[:, : ,0] += 1.
            params[:, :, 2] *= 1.05
            # Only derivs[:, :, 2] is not 1 in this example.
            derivs[:, :, 2] = 1.05
            return params, derivs

        model.ptweak=ptweak

        # Modify input accordingly
        xintrinsic=xtest.copy()
        xintrinsic[:, :, 0] -= 1
        xintrinsic[:, :, 2] /= 1.05

        # ptweak may modify xx in place
        xx=xintrinsic.copy()
        xx, derivs = ptweak(xx)
        self.assertAlmostEqual(np.max(np.abs(xx-xtest)), 0,
                               msg="Something's wrong with ptweak")

        xintrinsic_normed=model.normalize_parameters(xintrinsic)
        self.check_gradient(model.eval, xintrinsic_normed)

        model.poffset=None
        model.pscale=None
        self.check_gradient(model.eval, xintrinsic, noscale=True,
                            epsilon=[1e-6, 1e-12, 1e-12])

        xintrinsic=xreal.copy()
        xintrinsic[:, :, 0] -= 1
        xintrinsic[:, :, 2] /= 1.05
        self.assertAlmostEqual((np.max(np.abs(model.data
                                              -model.model(xintrinsic,
                                                           noscale=True)))), 0)

    def test_cubemodel_regularization(self):
        '''Check that CubeModel.regularization succeeds
        (flavor: noreg, markov, l1l2)
        '''
        print(f"  Temporary debug of test regularization")
        nx = ny = 10
        # test for uniform image
        # create a uniform image (2D array)
        img_uniform = np.full((nx,ny),50)
        crit, grad = markov(img_uniform)
        print(f"crit uni {crit}")
        print(f"grad {grad}")
        # adding a constant should not change result
        img_uniform += 50
        crit, grad = markov(img_uniform)
        print(f"crit uni const {crit}")
        print(f"grad {grad}")
        # test for spiked image
        img_spike = np.full((nx,ny),50)
        img_spike[4:6,4:6] = 100
        crit, grad = markov(img_spike)
        print(f"crit spike {crit}")
        print(f"grad {grad}")
        # test for random image
        img_rand = np.random.normal(0, 1, (nx,ny))
        crit, grad = markov(img_rand)
        print(f"crit rand {crit}")
        print(f"grad {grad}")
        img_bicolor = np.zeros((nx,ny))
        img_bicolor[5:9] = 1
        crit, grad = markov(img_bicolor)
        print(f"crit bicol {crit}")
        print(f"grad {grad}")


if __name__ == '__main__':
    unittest.main()
