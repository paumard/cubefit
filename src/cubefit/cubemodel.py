#!/usr/bin/env python3
#    Copyright (C) 2023  Thibaut Paumard <thibaut.paumard@obspm.fr>
#            Julien Brulé
#            Damien Gratadour
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits  # pour fits

# First try a relative import. This will work when lineprofiles,
# dopplerlines and cubefit are submodules of a common module.  This
# will fail when calling one of the submodules is called as a script,
# so fall back to a simple import to cope with that use case.
try:
    from cubefit.lineprofiles import gauss, ngauss
except ImportError:
    from lineprofiles import gauss, ngauss
try:
    from cubefit.dopplerlines import DopplerLines
except ImportError:
    from dopplerlines import DopplerLines

# Look hard for vmlmb
try:
    from VMLMB.python.optm import vmlmb, reason as vmlmb_reason
except ImportError:
    try:
        from vmlmb.optm import vmlmb, reason as vmlmb_reason
    except ImportError:
        from optm import vmlmb, reason as vmlmb_reason

# Regularization functions


def markov(x, scale=None, delta=None):
    """Compute a markov regularization

    cubemodel.markov(x,scale=,delta=])

    delta^2 . sum [|dx|/delta -ln(1+|dx|/delta) + |dy|/delta -ln(1+|dy|/delta)]
    where |dx| (m,n) = (o(m,n)-o(m-1,n))/scale is the x component of the
    object gradient normalized by scale.

    Parameters
    ----------
    x : array like where to apply the markov regularization
    scale : Scale factor applied to do (default value is 1)
    delta : Threshold on the gradient for switching between linear and
    quadratic behavour. When delta tends to infinity, the criterion
    becomes purely quadratic.

    AUTHOR: Damien Gratadour, borrowed from Yoda.
    """
    # TODO scale et delta already in self
    if (scale is None):
        scale = 1.
    if (delta is None):
        delta = 1.

    dx = (x - np.roll(x, 1, axis=1)) / (delta * scale)
    dy = (x - np.roll(x, 1, axis=0)) / (delta * scale)

    amap = np.abs(dx) - np.log(1. + np.abs(dx)) + \
        np.abs(dy) - np.log(1. + np.abs(dy))

    # TODO integrate returnmap with self.dbg
    # if (returnmap):
    #    return (delta**2) * amap

    crit = (delta**2) * np.sum(amap)

    dx /= (1. + abs(dx))
    dy /= (1. + abs(dy))

    # TODO reimplement the true gradient
    grad_obj = (dx - np.roll(dx, -1, axis=1)
                + dy - np.roll(dy, -1, axis=0)) * (delta / scale)

    return crit, grad_obj


def l1l2(x, scale=None, delta=None):
    """Compute a l1l2 regularization

    cubefit.l1l2(x[,scale=,delta=])
    delta^2 . sum [|dx|/delta -ln(1+|dx|/delta)+ |dy|/delta -ln(1+|dy|/delta)]
    where |dx| (m,n) = [o(m,n)-o(m-1,n)]/scale
    is the x component of the  object gradient normalized by scale.

    Parameters
    ----------
    x : array like
    scale : Scale factor applied to Do (default value is 1)
    delta : Threshold on the gradient for switching between linear
    and quadratic behavour. When delta tends to infinity,
    the criterion becomes purely quadratic.

    AUTHOR: Damien Gratadour, borrowed from Yoda.
    """
    # TODO scale et delta already in self
    if (scale is None):
        scale = 1.
    if (delta is None):
        delta = 1.

    dx = (x - np.roll(x, 1, axis=1)) / (delta * scale)
    dy = (x - np.roll(x, 1, axis=0)) / (delta * scale)

    r = np.sqrt(dx**2 + dy**2)

    amap = r - np.log(1.+r)

    # TODO integrate returnmap with self.dbg
    # if (returnmap):
    #    return (delta**2) * amap

    # if (returnmap):
    #    return (delta**2) * amap

    crit = (delta**2) * np.sum(amap)

    dx /= (1. + r)
    dy /= (1. + r)

    # TODO reimplement the true gradient
    # TODO dx-roll is that ok (x - roll) en place de object-roll
    # ou est ce object - roll ?
    grad_obj = (dx - np.roll(dx, -1, axis=1)
                + dy - np.roll(dy, -1, axis=0)) * (delta / scale)

    return crit, grad_obj


class CubeModel:
    """Spectro-imaging data model

    The CubeModel class is designed for spectral fitting with spatial
    regularization in a spectro-imaging context.

    The 3D model is based on a 1D model (called profile) and 2D
    parameter maps. The 2D maps are regularised (using by default an
    L1L2 regularization).

    The estimator is a compound of a chi^2 (based on the 1D model), a
    regularization term (based of the 2D regularisation of the various
    2D parameter maps) and an optional decorrelation term (based on
    the crosscorrelation of specific pairs of paramter maps).

    Attributes
    ----------
    data : array_like, optional
        The data to fit, a NY x NX x NW array. Mandatory to call
        methods eval() and fit(). The spectra are in the W dimension.
    weight : array_like, optional
        An array with the same dimensions as data, giving the relative
        weights of the data points for fitting. Set cell to 0. for
        points to ignore altogether. For Gaussian noise, this should
        be set to data standard deviation. Defaults to None, in which
        case it is uniformly initialized to 1. the first time eval ()
        is called.
    profile : callable
        1D model function with signature
           profile(xdata, *params) -> ydata, Jacobian
    profile_xdata : array_like
        whatever profile accepts as its first positional
        argument. Often a wavelength axis, sometimes a complex object,
        possibly None.
    regularization: callable, optional
        The regularization function to use, by default
        cubefit.cubemodel.markov. Any function with the same prototype
        as l1l2 or markov is suitable. Set regularization to None
        remove regularization entirely.
    decorrelate : UNIMPLEMENTED
        pairs of map ID for which the cross-correlation should ba
        minimal together with weight, e.g. decorrelate = [1, 2, 0.4]
        if the maps for parameters 1 and 2 should not be correlated,
        with a weight of 0.4 relative to the other terms of the
        estimator.
    pscale : {None, array_like}
        A vector with the same size as the params argument of the
        profile function. If not None, the parameters will be
        automatically devided and multiplied by this amount so the
        fitting engine only manipulates rescaled parameters. This
        helps some fitting engines to behave properly.
    poffset : {None, array_like}
        A vector with the same size as the params argument of the
        profile function. If not None, the parameters will be
        automatically devided and multiplied by this amount so the
        fitting engine only manipulates rescaled parameters. This
        helps some fitting engines to behave properly.
    ptweak : callable
        A function to add instrumental signature to the parameters,
        with signature
            ptweak(params) -> params, derivatives
        Should modify the parameters and return the modified
        parameters and their derivatives as two arrays of the same
        shape. The params array may be modified in place. At the
        moment, these modifications of the parameters should not
        introduce correlations (because only the derivative of each
        parameter is returned, not a full Jacobian matrix). Possible
        usage includes adding a per-pixel velocity offset or modify
        linewidth to mimick convolution by a spectral PSF.
    scale
        UNDOCUMENTED
    delta
        UNDOCUMENTED
    dbg : bool
        Whether to activate debugging output.
    """
    def __init__(self, data=None, profile=None, profile_xdata=None,
                 weight=None, scale=None, delta=None, pscale=None,
                 poffset=None, ptweak=None, regularization=markov,
                 decorrelate=None, view_data=None, view_more=None,
                 framedelay=3):

        self.regularization = regularization
        self.profile = profile
        self.profile_xdata = profile_xdata
        self.data = data
        self.weight = weight
        self.scale = scale
        self.delta = delta
        self.pscale = pscale
        self.poffset = poffset
        self.ptweak = ptweak
        self.derivatives = None
        self.decorrelate = decorrelate
        self.framedelay = framedelay
        # private data
        # to pass additional information from eval to printer
        self._eval_data = dict()
        self._printer_data = dict()
        # to store data between printer calls
        if view_data is None:
            view_data = {"figsize": (7, 7)}
        self.view_data = view_data
        self.view_more = view_more

        # debug option
        self.dbg = False
        # TODO ajout dict debug, monitor pour returnmaps voir plus
        self.dbg_data = {}

    def view(self, x, noscale=None):
        """ View a CUBEFIT patameter maps.

        Used internally for visualisation during a fit (see fit),
        but can also be called direclty (in this case, NOSCALE should
        probably be set to 1).

        Parameters
        ----------
        x : array_like
        a stack of parameter maps
        noscale : boolean
        noscale=1 to ignore pscale and poffset. NOSCALE should almost
        always be used when VIEW is used manually (in this case,
        x has not been rescaled consistently with pscale and poffset).
        """

        if not noscale:
            x = self.denormalize_parameters(x)

        nterms = np.shape(x)[2]

        # Prepare plot window
        if ("fig" not in self.view_data or self.view_data["fig"] is None):
            self.view_data["fig"] = \
                plt.figure(figsize=self.view_data["figsize"])
            self.view_data["fig"].show()
            self.view_data["axes"] = []
        if ("axes" not in self.view_data
                or len(self.view_data["axes"]) != nterms):
            nx = int(np.sqrt(nterms))
            if nx*nx == nterms:
                ny = nx
            else:
                ny = nx+1
                nx = ny
            self.view_data["axes"] = \
                [self.view_data["fig"].add_subplot(ny, nx, p)
                 for p in range(1, nterms+1)]

        for k in range(nterms):
            self.view_data["axes"][k].clear()
            if "imshow_kwds" in self.view_data:
                kwds = self.view_data["imshow_kwds"][k]
            else:
                kwds = dict()
            self.view_data["axes"][k].imshow(x[:, :, k], **kwds)
        if self.view_more is not None:
            self.view_more(self.view_data["fig"], self.view_data["axes"])
        self.view_data["fig"].canvas.draw()

    def model(self, params, noscale=False):
        """Create a model cube from a set of parameter maps.

        Like for the VIEW method, NOSCALE should almost always be set to 1.

        Parameters
        ----------
        params : array_like
        a parameter maps array first dimension number of profile function
        parameters  ny nx

        See Also
        --------
        cubefit, view
        """
        # /!\ WARNING: this function is for visualisation,
        # it is "inlined" in eval for optimization

        params_dim = params.shape

        if noscale:
            xs = params
        else:
            xs = self.denormalize_parameters(params)

        # TODO derivatives never used in model?
        if self.ptweak is not None:
            self.derivatives = self.ptweak(xs)

        nx = params_dim[0]
        ny = params_dim[1]
        nz = self.profile(self.profile_xdata, *xs[0, 0, :])[0].size

        y = np.zeros((nx, ny, nz))

        for i in range(nx):
            for j in range(ny):
                y[i, j, :] = self.profile(self.profile_xdata, *xs[i, j, :])[0]

        return y

    def denormalize_parameters(self, x):
        '''De-apply pscale and poffset from normalized parameters x

            denormalize paramaters from -1,1 boundaries to physical unit

            See Also
            --------
            normalize_parameters
        '''
        # Note: self.pscale and poffset are broadcast automatically
        if (self.pscale is None):
            xs = x.copy()
        else:
            xs = x * self.pscale

        if (self.poffset is not None):
            xs += self.poffset

        return xs

    def normalize_parameters(self, x):
        '''Apply pscale and poffset to parameters x
           Parameters
           ----------
           x : aray_like
           the array of parameters to be normalized
        '''
        # Note: self.pscale and poffset are broadcast automatically
        x_norm = np.copy(x)
        if self.poffset is not None:
            x_norm -= self.poffset
        if self.pscale is not None:
            x_norm /= self.pscale

        return x_norm

    def eval(self, x, noscale=False, returnmaps=None):
        """Evaluates fitting criterion and its gradient

        Parameters
        ----------
        x : array_like
            3D array of parameters, with shape NY×NX×NP where NY and
            NX are the same as the two first dimensions of self.data
            and NP is the number of parameters for self.profile.
        noscale : bool, optional
            Whether to scale parameters according to poffset and
            pscale. Always False during a fit, but should generally be
            set to True when calling the method manually. Note that
            noscale only affects how x is interpreted: the computation
            of the gradient still takes pscale and poffset into
            account. This way the user can get the value of the
            gradient that is seen by the fitting routine. If in doubt,
            rescale x using self.normalize_parameters and set noscale
            to False.
        returnmaps: bool, UNIMPLEMENTED
            If returnmaps is True, returns a stack of maps for each
            component of the criterion (chi2 term last) instead of the
            integral. For debugging purposes.

        Returns
        -------
        res : float
            The value of the criterion, the sum of a weighted chi2 and
            a regularization term..
        grad : array_like
            The gradient of the criterion.

        Raises
        ------
        ValueError
            If self.data is not set or if ptweak derivatives have
            wrong shape.

        Notes
        -----
        To remove regularization altogether, set self.regularisation
        to None.

        Side effect: if data is not known but weight is None, set
        weight to a cube of ones.

        """
        if self.dbg:
            print("DBG CALL eval func with x")
            print(f"{x}")

        # eval() requires self.data to be set
        if self.data is None:
            raise ValueError("eval() requires self.data to be set")

        # weights default to 1
        if self.weight is None:
            self.weight = np.ones(self.data.shape)

        d = x.shape
        nx = d[0]
        ny = d[1]
        res = 0.
        gx = np.zeros(d)

        if (noscale):
            x = self.normalize_parameters(x)

        xs = self.denormalize_parameters(x)

        if self.dbg:
            print(f"shape xs {xs.shape}")

        if (self.ptweak is not None):
            xs, self.derivatives = self.ptweak(xs)
            if (self.derivatives is not None
                and (self.derivatives.size != xs.size
                     or (
                       np.asarray(self.derivatives.shape) != xs.shape).any())):
                raise ValueError("ptweak derivatives should be None "
                                 "or same size as parameter array")
            if self.dbg:
                print("DBG xs after ptweak:")
                print(xs)
        else:
            self.derivatives = None

        if (self.dbg):
            self.dbg_data["maps"] = np.zeros((nx, ny, d[2]+1))

        for i in range(nx):
            for j in range(ny):
                if (any(self.weight[i, j, :])):
                    spectrum = self.data[i, j, :]
                    model, model_jacobian = self.profile(self.profile_xdata,
                                                         *xs[i, j, :])
                    grad = model_jacobian
                    if (self.pscale is not None):
                        for k in range(self.profile_xdata.size):
                            grad[k, :] *= self.pscale

                    if (self.derivatives is not None):
                        grad *= self.derivatives[i, j, np.newaxis, :]

                    atom = (model - spectrum) * self.weight[i, j, :]

                    if (self.dbg):
                        self.dbg_data["maps"][i, j, 0] = sum(atom**2)

                    res += sum(atom**2)
                    gx[i, j, :] += 2. * np.sum(
                                        (grad * atom[:, np.newaxis] *
                                         self.weight[i, j, :][:, np.newaxis]),
                                               axis=0)
        self._eval_data["chi2"] = res

        # create a bigger map to avoid side effect
        xbig = np.zeros((d[0]*2, d[1]*2, d[2]))
        xbig[:d[0], :d[1], :] = x
        xbig[:d[0], d[1]:, :] = np.flip(x, 1)
        xbig[d[0]:, :, :] = xbig[d[0]:0:-1, :, :]

        self._eval_data["regul"] = np.zeros(d[2])
        for k in range(d[2]):
            # TODO pass dict to regularization function
            if self.regularization is not None:
                if self.scale is not None and self.delta is not None:
                    tmp, g = self.regularization(xbig[:, :, k],
                                                 self.scale[k], self.delta[k])
                else:
                    tmp, g = self.regularization(xbig[:, :, k])

                tmp = tmp / 4.

            # TODO returnmaps
                if (self.dbg):
                    self.dbg_data["maps"][:, :, k] = tmp[:d[0], :d[1]]
                    print(f"g after regul {g}")
                    print(f"g is a {g.shape}")

                self._eval_data["regul"][k] = tmp
                res += tmp
                gx[:, :, k] += g[0:d[0]:+1, 0:d[1]:+1]

        # TODO a voir a la fin faible priorite
        if (self.decorrelate is not None):
            dd = self.decorrelate.shape
            if (len(dd.shape) != 2):
                # TODO reform
                self.decorrelate = np.reshape(self.decorrelate,
                                              (2, self.decorrelate.size, 1))
                dd = self.decorrelate.shape

            npairs = dd.shape[2]
            for pair in range(npairs):
                # TODO long
                # i1 = long(decorrelate[1, pair])
                # TODO change
                i1 = np.floor(self.decorrelate[1, pair])
                i2 = np.floor(self.decorrelate[2, pair])
                w = self.decorrelate[2, pair]
                if (dd.shape[1] >= 4):
                    pow = self.decorrelate[4, pair]
                else:
                    pow = 2
                xy = x[:, :, [i1, i2]]
                # TODO
                correl = CubeModel.corr(xy, grad, deriv=1)
                res += w * correl**pow
                gx[:, :, [i1, i2]] += w*pow*correl**(pow-1)*grad

        #  out:
        return res, gx

    def printer(self, output, iters, evals, rejects,
                t, x, fx, gx, pgnorm, alpha, fg):
        """Printer for `optm.vmlmb`."""

        nterms = np.shape(x)[2]

        # First call with iters < 1 to initialize
        if iters < 1:
            # Initialize tprev
            self._printer_data["tprev"] = -self.framedelay-1

            # Print header line
            row = "# Iter.   Time (ms)    Eval. Reject.       Obj. Func.           Chi2   "
            lin = "# ----------------------------------------------------------------------"
            for k in range(nterms):
                row += f"    Regul[{k}]"
                lin += "------------"
            row += "       Grad.       Step"
            lin += "-----------------------"
            print(row, file=output)
            print(lin, file=output)

        # print info on one row
        row = f"{iters:7d} {t*1e3:11.3f} {evals:7d} {rejects:7d} {fx:23.15e} {self._eval_data['chi2']:11.3e} "
        for val in self._eval_data['regul']:
            row += f"{val:11.3e} "
        row += f"{pgnorm:11.3e} {alpha:11.3e}"
        print(row, file=output)

        # Plot x
        if (self.framedelay >= 0
                and t - self._printer_data["tprev"] > self.framedelay):
            self.view(x)
            self._printer_data["tprev"] = t

    def criterion(self, x, noscale=False):
        '''return the criterion of the eval function

            self.eval(x, noscale=noscale)[0]
        '''
        return self.eval(x, noscale=noscale)[0]

    def fit(self, x,
            lower=None, upper=None,
            verb=1, printer=None, output=sys.stdout,
            **vmlmb_kwargs):
        """Fit model to self.data.

        Tis is a wrapper around the fitter (vmlmb) which normalizes
        parameters according to self.pscale and self.poffset, sets up
        a pretty printer etc.

        Parameters
        ----------
        x : array_like
            Initial guess for the stack of parameters to fit (see
            eval). Rescaled according to self.pscale and self.poffset
            prior to calling vmlmb and scaled back to physical values
            in self.eval(). The result is also scaled back before
            being returned.
        lower, upper : array_like, optional
            Passed to vmlmb after rescaling according to self.poffset
            and self.pscale.
        printer : callable
            Passed to vmlmb, if None, default to self.printer.
        verb : int
            Passed to vmlmb.
        output :
            Passed to vmlmb.
        **vmlmb_kwargs : dict, optional
            Extra arguments are passed to vmlmb.

        Returns
        -------
        array_like
            Best-fit parameters.
        float
            Value of the objective function eval at the found optimum.
        array_like
            Gradient of the objective function eval at the found optimum.
        int
            Status code indicating the reason of the termination of
            the vmlmb algorithm. See optm.reason.

        See Also
        --------
        CubeModel.eval : The objective function.
        optm.vmlmb : The fitter.
        optm.reason : Reasons corresponding to status codes.

        Notes
        -----
        Accept all vmlmb keywords (vmlmb_kwargs).
        lower and upper need special treatment (for rescaling).
        There's a bug in optm which requires verb > 0.
        output is used in fit, we need it.
        Set printer to self.printer by default, but not in the signature.
        """

        # This can't be done in signature
        if printer is None:
            printer = self.printer

        # Normalize paramaters according to user expectation as fitter
        # may behave better and it may be easier to tune
        # hyperparameters if all fitted quantities are or order 1.
        x = self.normalize_parameters(x)
        if (lower is not None):
            lower = self.normalize_parameters(lower)
        if (upper is not None):
            upper = self.normalize_parameters(upper)

        # scale and delta are the hyperparameters for
        # self.regularization. Ensure they have the right shape.
        nterms = x.shape[2]
        if (self.scale is None):
            self.scale = np.ones(nterms)
        else:
            if (self.scale.size == 1):
                self.scale = np.full(nterms, self.scale)

        if (self.delta is None):
            self.delta = np.ones(nterms)
        else:
            if (self.delta.size == 1):
                self.delta = np.full(nterms, self.delta)

        # Perform the actual fit.
        (result, fx, gx, status) = vmlmb(self.eval, x,
                                         lower=lower, upper=upper,
                                         verb=verb,
                                         printer=printer,
                                         output=output,
                                         **vmlmb_kwargs)
        if verb > 0:
            print(f"# Termination: {vmlmb_reason(status)}", file=output)
            if self.framedelay >= 0:
                self.view(result)

        # If we normalized x, we need to denormalize the results.
        result = self.denormalize_parameters(result)

        return result, fx, gx, status

    def op_printer(self, output, iter, eval, cpu,
                   fx, gnorm, steplen, x, extra):
        # TODO printer
        # extra, printer, output, iter, eval, cpu, fx, gnorm, steplen, x
        print("TODO op_printer")

    @staticmethod
    def op_f(self, x, *gx, extra):
        """
        wrapper de wrapper...
        """
        return extra(eval, x, gx)

    def gmin_f(self, x, a):
        return x(eval, a)

    def gmin_viewer(self, x, a):
        # TODO
        # x, view, a
        print("TODO gmin_viewer")

    def op_viewer(x, extra):
        """
        /* DOCUMENT cubefit.op_viewer, x, obj

        A "static function" wrapper around the cubefit.view method,
        which can be used as the VIEWER parameter of the OP_MNB routine
        from optimpack. Equivalent to:

            obj, view, x

        SEE ALSO:
                cubefit, op_mnb, view
        */
        """
        # TODO
        # extra,view, x
        print("TODO op_viewer")


def corr(xy, *grad, deriv=None):
    """
    DOCUMENT correlation = cubefit.corr(xy [, grad, deriv=1])
    Return the cross-correlation of XY(..,1) and XY(..,2)
    """
    x = xy[:, 1]
    y = xy[:2]

    d = x.shape
    n = x.size
    sx = sum(x)
    sy = sum(y)
    u = n*sum(x*y) - (sx*sy)
    a = n*sum(x**2) - sx**2
    b = n*sum(y**2) - sy**2
    v = np.sqrt(a*b)
    if (v):
        res = u/v
    else:
        res = 1.

    # res = v ? u/v : 1.

    if (deriv):
        if (v):
            gx = (n*y - sy - u*(n*x - sx)/a) / v
            gy = (n*x - sx - u*(n*y - sy)/b) / v
            grad = [gx, gy]

        else:
            grad = np.array([1., d, 2])

    return res, grad


def test_gauss():
    # debug model
    print("starting cubefit test gauss....")

    # create a cube
    nx = 5
    ny = 5
    nz = 433
    nparms = 5

    weight = np.ones((nx, ny, nz))

    cube_gauss = np.ndarray((nx, ny, nz))
    cube_noise_gauss = np.ndarray((nx, ny, nz))

    # choose parameter for model

    gauss_param = np.array([1, 1, 0.5, 0.5, 0.1])
    gauss_xdata = np.linspace(-10, 10, nz)

    sigma = 0.02

    print("test gauss call")
    _ = gauss(gauss_xdata, *gauss_param)[0]
    # y_noise = y + np.random.standard_normal(y.size) * sigma

    # plt.figure()
    # plt.plot(gauss_xdata, y)
    # plt.plot(gauss_xdata, y_noise)
    # plt.show()

    model_param_gauss = np.full((nx, ny, nparms), gauss_param)

    print(f" model_param_gauss[4,4,:]{model_param_gauss[4,4,:]}")

    fcn_gauss = gauss
    fcn_x_gauss = gauss_xdata

    # cube_gauss[...,0:nz] = gauss_param
    # np.c_[cube_gauss,gauss_param]

    print("creating line obj")
    DopplerLines(lines=gauss_param[0], profile=ngauss)

    # create fit obj
    print("creating fit obj")
    fitobj_gauss = CubeModel(cube_gauss, fcn_gauss, fcn_x_gauss, weight)

    # model
    # cube_model=fitobj.model(cube,a0)
    print("create model ...")
    cube_model_gauss = fitobj_gauss.model(model_param_gauss)

    # add noise
    cube_noise_gauss = add_noise(cube_model_gauss, sigma)

    # write fits cube
    write_fits(cube_model_gauss, "mycube_model_gauss.fits")
    write_fits(cube_noise_gauss, "mycube_gauss.fits")

    print("check fits")
    open_fits(cube_model_gauss, "mycube_model_gauss.fits")

    print("check cube")
    print(f"{cube_model_gauss[4,4,:]}")

    # on recree un obj cubefit
    fitobj_eval_gauss = CubeModel(cube_noise_gauss,
                                  fcn_gauss, fcn_x_gauss, weight)

    # calcul du point de depart x0 les cartes de parametres initiales
    x0 = model_param_gauss
    # x0 = gauss_param

    (res_x, fx, gx, status) = vmlmb(fitobj_eval_gauss.eval, x0, mem=x0.size,
                                    blmvm=False, fmin=0, verb=1,
                                    output=sys.stdout)

    print(f"fx {fx}")
    print(f"gx {gx}")
    print(f"status {status}")


def test():
    """
    EXAMPLE
    #fitobj = cubefit(new, cube = cubl(,,indlines), weight = noise,
    #                 fcn = lineobj.lmfit_func, fcn_x= lineobj,
    #                 scale=scale, delta=delta, regularization=cubefit.l1l2,
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
    print("starting cubefit test....")
    # create a cube
    nx = 5
    ny = 5
    nz = 433

    print("create model ...")
    weight = np.ones((nx, ny, nz))

    cube_doppler = np.ndarray((nx, ny, nz))
    cube_noise_doppler = np.ndarray((nx, ny, nz))

    print("test doppler ...")
    # choose parameter for model
    # nlines = 1
    lines = np.array([2.166120])
    # lines=np.array([2.166120,2.155])

    doppler_param = np.array([1.2, 0.5, 25., 100])
    doppler_xdata = np.linspace(2.15, 2.175, nz)

    print("creating line obj")
    lineobj_doppler = DopplerLines(lines, profile=ngauss)

    print("after doppler init")
    sigma = 0.5
    print("test doppler call")
    lineobj_doppler(doppler_xdata, *doppler_param)[0] \
        + np.random.standard_normal(nz) * sigma

    model_param_doppler = np.full((nx, ny, doppler_param.size),
                                  doppler_param)

    # print("creating line obj")
    # lineobj_doppler = DopplerLines(lines=lines, profile=ngauss)

    # TODO choose return tuple or not
    # fcn_doppler = lineobj_doppler.curvefit_func
    fcn_doppler = lineobj_doppler.__call__
    # fcn_x=a0
    # important
    fcn_x_doppler = doppler_xdata

    print("creating fit obj")
    # create fit obj
    fitobj_doppler_model = CubeModel(cube_doppler, fcn_doppler,
                                     fcn_x_doppler, weight)
    # print(f"cube shape {cube.shape}")
    # print(f"cube_noise shape {cube_noise.shape}")

    # test more parameters
    # fitobj=cubefit(cube,cube_noise,fcn,fcn_x,scale, delta,
    #            pscale,poffset, ptweak,regularization=None,decorrelate=None)

    # model
    # cube_model=fitobj.model(cube,a0)
    print("compute model ...")
    cube_model_doppler = fitobj_doppler_model.model(model_param_doppler)

    # add noise

    cube_noise_doppler = add_noise(cube_model_doppler, sigma)

    # print_array_slice(cube_noise_doppler)
    # print_array_slice(cube_noise_gauss)

    write_fits(cube_model_doppler, "mycube_model_doppler.fits")
    write_fits(cube_noise_doppler, "mycube_doppler.fits")

    print("check fits")
    open_fits(cube_model_doppler, "mycube_model_doppler.fits")
    # print("diff noise - model")
    # print(f"{cube_noise_doppler[49,49,:] - cube_model_doppler[49,49,:]}")

    print("check cube")
    print(f"{cube_model_doppler[4,4,:]}")

    # debug view
    # fitobj_doppler.view(cube_model_doppler)

    # debug eval
    print(f"============#debug eval")

    # TODO nz=433
    # cube_empty = np.ndarray((nx, ny, nz))

    # on recree un obj cubefit
    fitobj_eval_doppler = CubeModel(cube_noise_doppler,
                                    fcn_doppler, fcn_x_doppler, weight)

    # tiré de l exemple
    # f = eval(f"fitobj_eval.eval")
    # x0 = f(0, n=n, factor=factor)

    # calcul du point de depart x0 les cartes de parametres initiales
    # x0 = cube_noise_doppler
    x0 = model_param_doppler


    print(f"calling vmlmb")
    # def vmlmb_printer(output, iters, evals, rejects, t, x, fx, gx,
    # pgnorm, alpha, fg):
    print("# Iter.   Time (ms)    Eval. Reject.       Obj. Func.         Grad.       Step")
    # ---------------------------------------------------------------------------------
    # besoin fonction objectif (param 1)  gradient (param 2)
    # (res_x, fx, gx, status) = vmlmb(lambda x: (f(1, x), f(2, x)), x0,
    #         mem=x0.size , blmvm=False, fmin=0, verb=1, output=sys.stdout)
    (res_x, fx, gx, status) = vmlmb(fitobj_eval_doppler.eval, x0, mem=x0.size,
                                    blmvm=False, fmin=0,
                                    verb=1, output=sys.stdout)

    print("# Iter.   Time (ms)    Eval. Reject.       Obj. Func.         Grad.       Step")
    print("res_x for dop")
    print(f"{res_x}")
    # write_fits(res_x, "mycube_res_x_dop.fits")
    # write_fits(fx, "mycube_res_fx_dop.fits")
    # write_fits(gx, "mycube_res_gx_dop.fits")
    print(f"fx {fx}")
    print(f"gx {gx}")
    print(f"status {status}")

    # def test(probs = range(1, 19),  n=None, factor=None, fmin=0,
    #    mem="max", verb=1, blmvm=False, output=sys.stdout, **kwds)
    # test(verb=1, gtol=0, xtol=0, ftol=0, fmin=0, mem="max")

    # from test_fit-Brg4
    # xl1 = fitobj.fit( xl, verb=1, lower=lower, upper=upper);


def test_fit():
    """
    implementation of the fit function test
    """
    # debug model
    print("starting cubefit test_fit....")
    # create a cube
    nx = 5
    ny = 5
    nz = 433

    print("create model ...")

    cube_doppler = np.ndarray((nx, ny, nz))
    cube_noise_doppler = np.ndarray((nx, ny, nz))
    weight = np.ones((nx, ny, nz))
    print("test doppler ...")
    # nlines = 1
    lines = np.array([2.166120])
    doppler_param = np.array([1.2, 0.5, 25., 100])
    doppler_xdata = np.linspace(2.15, 2.175, nz)

    print("creating line obj")
    lineobj_doppler = DopplerLines(lines, profile=ngauss)

    sigma = 0.2

    model_param_doppler = np.full((nx, ny, doppler_param.size),
                                  doppler_param)

    # TODO choose return tuple or not
    fcn_doppler = lineobj_doppler.__call__
    fcn_x_doppler = doppler_xdata

    print("creating fit obj")
    # create fit obj
    fitobj_doppler_model = CubeModel(cube_doppler, fcn_doppler,
                                     fcn_x_doppler, weight)
    # model
    print("compute model ...")
    cube_model_doppler = fitobj_doppler_model.model(model_param_doppler)
    # add noise
    cube_noise_doppler = add_noise(cube_model_doppler, sigma)

    write_fits(cube_model_doppler, "mycube_model_doppler_fit.fits")
    write_fits(cube_noise_doppler, "mycube_doppler_fit.fits")

    print("check fits")
    open_fits(cube_model_doppler, "mycube_model_doppler_fit.fits")
    # print("diff noise - model")
    # print(f"{cube_noise_doppler[49,49,:] - cube_model_doppler[49,49,:]}")

    print("check cube")
    print(f"{cube_model_doppler[4,4,:]}")

    # debug view
    # fitobj_doppler.view(cube_model_doppler)

    # debug eval
    print(f"============#debug eval")

    # TODO nz=433
    # cube_empty = np.ndarray((nx, ny, nz))

    # on recree un obj cubefit
    fitobj_eval_doppler = CubeModel(cube_noise_doppler,
                                    fcn_doppler, fcn_x_doppler, weight,
                                    regularization=l1l2)

    doppler_param_test = np.array([1.1, 1., 25., 100])
    # doppler_param_test = np.array([1.2, 0.5, 25., 100])
    model_param_doppler_test = np.full((nx, ny, doppler_param_test.size),
                                       doppler_param_test)

    # calcul du point de depart x0 les cartes de parametres initiales
    # x0 = cube_noise_doppler
    x0_test = model_param_doppler_test

    print(f"x0_test {x0_test}")
    print(f"calling fit function")

    (res_x, fx, gx, status) = fitobj_eval_doppler.fit(x0_test)

    print("# Iter.   Time (ms)   Eval. Reject.    Obj. Func.      Grad.  Step")
    print("res_x for test_fit")
    print(f"{res_x}")
    print(f"fx {fx}")
    print(f"gx {gx}")
    print(f"status {status}")


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
    size = min(myarray.shape)//2
    print(f"myarray size-1 {myarray[size-1,size-1,:]}")
    print(f"myarray size {myarray[size,size,:]}")
    print(f"myarray size+1 {myarray[size+1,size+1,:]}")


def plot_array_slice(myarray):
    # print(myarray.shape)
    size = min(myarray.shape)//2
    plt.figure()
    plt.imshow(myarray[size, size, :], cmap='gray')
    plt.colorbar()
    plt.show()


def add_noise(cube, sigma=0.02):
    psigma = sigma
    tmp_cube = np.copy(cube)
    tmp_cube = cube + np.random.standard_normal(cube.shape) * psigma
    return tmp_cube


def numerical_gradient3(f, x, epsilon=1e-6):
    '''Compute gradient numerically
    '''
    d = x.shape
    g = np.zeros(d)
    f0 = f(x)
    for k in range(d[2]):
        for j in range(d[2]):
            for i in range(d[2]):
                temp = np.copy(x)
                temp[i, j, k] += epsilon
                g[i, j, k] = (f(temp)-f0)/epsilon
    return g


class RegularizationWithNumericalGradient:
    '''Wrapper around a regularization function to provide numerical gradient

    Meant for debugging or prototyping. Gradient should be estimated
    analitycally whenever possible.

    '''

    def __init__(self, func, epsilon=1e-6):
        self.func = func
        self.epsilon = epsilon

    def __call__(self, x, scale=None, delta=None):

        # ensure x is an array and we will not modify the input array
        x = np.copy(x)

        shape = x.shape

        f0 = self.func(x, scale, delta)
        if type(f0) is tuple:
            f0 = f0[0]

        g = np.zeros(shape)

        for j in range(shape[1]):
            for i in range(shape[0]):
                val = x[j, i]
                x[j, i] = val + 0.5*self.epsilon
                fplus = self.func(x, scale, delta)
                if type(fplus) is tuple:
                    fplus = fplus[0]
                x[j, i] = val - 0.5*self.epsilon
                fminus = self.func(x, scale, delta)
                if type(fminus) is tuple:
                    fminus = fminus[0]
                x[j, i] = val
                g[j, i] = (fplus-fminus)/self.epsilon

        return f0, g


if __name__ == '__main__':
    testpourgauss = 0
    if testpourgauss:
        test_gauss()
    else:
        test_fit()
