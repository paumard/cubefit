
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename

from astropy.io import fits  # pour fits

# First try a relative import. This will work when ngauss,
# dopplerlines and cubefit are submodules of a common module.  This
# will fail when calling one of the submodules is called as a script,
# so fall back to a simple import to cope with that use case.
try:
    from cubefit.ngauss import gauss, ngauss
except ImportError:
    from ngauss import gauss, ngauss
try:
    from cubefit.dopplerlines import DopplerLines
except ImportError:
    from dopplerlines import DopplerLines

import sys

# Look hard for vmlmb
try:
    from VMLMB.python.optm import vmlmb
except ImportError:
    try:
        from vmlmb.optm import vmlmb
    except ImportError:
        from optm import vmlmb


# include "multiprofile.i"
# include "OptimPack1.i"
# include "cubeview.i"
# include "util_fr.i"

# python3
# from inspect import currentframe, getframeinfo
# frameinfo = getframeinfo(currentframe())
# print(frameinfo.filename, frameinfo.lineno)


"""
/*
   AUTHOR: Thibaut Paumard
           Damien Gratadour (l1l2, markov)
 */
"""
"""
// First save previous values of temporary function names
scratch = save(scratch,
               // methods
               new, view, eval, printer,
               fit, model,
               // static functions
               l1l2, markov, corr,
               op_f, op_viewer, op_printer
               )
#fitobj = CubeModel(new, cube = cubl(,,indlines), weight = noise,
#                 fcn = lineobj.lmfit_func, fcn_x= lineobj,
#                 scale=scale, delta=delta, regularisation=cubefit.l1l2,
#                 pscale=pscale, poffset=poffset, ptweak=myvlsr2vobs)

"""


class CubeModel:
    """
    CubeModel class

    CubeModel is an OXY class for designed for spectral fitting with
    spatial regularisation in a spectro-imaging context.

    The 3D model is based on a 1D model and 2D parameter maps. The 2D
           maps are regularised (using by default an L1L2 regularisation).

           The estimator is a compound of a chi^2 (based on the 1D model), a
           regularisation term (based of the 2D regularisation of the various
           2D parameter maps) and an optional decorrelation term (based on the
           crosscorrelation of specific pairs of paramter maps).

       MEMBERS
         data:   the data to fit, a NX x NY x NZ array. The spectra are in
                 the Z dimension.
         weight: optional (defaults to nil). An array with the same
                 dimensions as DATA, giving the relative weights of the
                 data points for fitting. Set to 0. for points to ignore
                 altogether. For Gaussian noise, this should be set to
                 data root-mean-square (beware: this is the square root of
                 what lmfit expects).
         fcn:    an LMFIT compatible 1D model function:
                  model_1d = fcn(fcn_x, parameters, grad, deriv=1)
                 model_1d must be an array(double, NZ).
         fcn_x:  whatever FCN accepts as its first positional
                 argument. Often a wavelength axis, sometimes a complex
                 object, possibly nil.
         regularisation: the regularisation function to use, by default
                 cubemodel.l1l2. Any function with the same prototype as
                 l1l2 or markov is suitable. Set regularisation to nil
                 remove regularisation entirely. Default: cubemodel.markov.
         decorrelate: pairs of map ID for which the cross-correlation
                 should ba minimal together with weight, e.g. decorrelate
                 = [1, 2, 0.4] if the maps for parameters 1 and 2 should
                 not be correlated, with a weight of 0.4 relative to the
                 other terms of the estimator.
         scale, delta, pscale, poffset, ptweak: should be documented.

    METHODS
         These methods can be called as
             fitobj, METHOD, [parameters]
         or  result = fitobj(METHOD, [parameters]).

         The NEW method is a bit peculiar as it is normally called from
         the template CUBEFIT object itself:
             fitobj = CubeModel(new, )
         It can however be called as
             fitobj2 = fitobj(new, )
    in which case the (potentially overriden) methods and static
    virtual functions are copied from FITOBJ to FITOBJ2. The members
    are not copied though (this may change in the future).

    The methods are all inherently "virtual" and can be overriden
    with
        fitobj, METHOD=new_implementation

    new:   create an instance of the class. This methods accepts a
           keyword for each of the cubefit members listed above.
    view:  visualise parameter maps
    eval:  compute the criterion (the function to minimise)
    model: compute a model cube from parameter maps (for visualisation)
    printer: print various things (in particular the criterion)
    fit:   performs the actual fit, using op_mnb()

    STATIC VIRTUAL FUNCTIONS
    Those functions are not "methods" as they don't "use" any
    member. They can be accessed as fitobj.FUNCTION() and can be
    overriden with fitobj, FUNCTION=newfunc.

    op_f:  wrapper around EVAL, suitable as the F argument of op_mnb()
    op_viewer: wrapper around view, suitable as VIEWER for op_mnb()
    op_printer: wrapper around printer, suitable as PRINTER for op_mnb()

    STATIC FUNCTIONS
    Those functions are in the cubefit object but not copied by NEW
    in the FIOBJ object. They can be called with: cubefit.FUNCTION().

    l1l2:   L1L2 regularisation method. e.g.:
              fitobj, regularisation=cubefit.l1l2
    markov: Markov regularisation method. e.g.:
              fitobj, regularisation=cubefit.markov

    SYNOPSIS
    fitobj = CubeModel(new, [members=members])
    x = array(double, NX, NY, NPARMS) // initial guess
    fitobj, fit, x
    fitobj, view, x

    SEE ALSO: oxy, METHOD, FUNCTION, op_mnb, lmfit
    */
    """
    # TODO important apres premiere phase , regularisation et ptweak
    #
    # def __init__(self,cube,weight,fcn=None,fcn_x=None, \
    #            regularisation=None,decorrelate=None, \
    #            scale=None, delta=None, pscale=None, \
    #            poffset=None, ptweak=None):
    def __init__(self, data=None, profile=None, profile_xdata=None, weight=None,
                 scale=None, delta=None, pscale=None, poffset=None,
                 ptweak=None, regularisation=None, decorrelate=None):
        if (regularisation is not None):
            self.regularisation = regularisation
        else:
            self.regularisation = markov
            # TODO regularisation should default to markov

        # function/methods should be the dopplerlin eval func
        self.profile = profile
        self.profile_xdata = profile_xdata
        # data
        self.data = data
        self.weight = weight
        self.scale = scale
        self.delta = delta
        self.pscale = pscale
        self.poffset = poffset
        self.ptweak = ptweak
        self.decorrelate = decorrelate

        # debug option
        self.dbg = False
        # TODO ajout dict debug, monitor pour returnmaps voir plus
        self.dbg_data = {}

    def view(self, x, noscale=None):
        """
        /* DOCUMENT fitobj, view, x
        View CUBEFIT patameter maps. Used internally for visualisation
        during a fit (see fit), but can also be called direclty (in
        this case, NOSCALE should probably be set to 1).

        OBJECT
        FITOBJ    an instance of thecubefit class, created with cibefit.new
        PARAMETERS
        X         a stack of parameter maps
        KEYWORDS
        noscale=1 to ignore pscale and poffset. NOSCALE should almost
        always be used when VIEW is used manually (in this case,
        X has not been rescaled consistently with pscale and poffset).

        SEE ALSO cubefit, new
        */
        """

        d = x.shape

        if (noscale is not None):
            psc = np.ones(d[2])
            pof = np.zeros(d[2])
        else:
            if (self.pscale is None):
                psc = np.ones(d[2])
            else:
                if (self.pscale.size == 1):
                    psc = np.full(d[2], self.pscale)
                else:
                    psc = self.pscale

            if (self.poffset is None):
                pof = np.zeros(d[2])
            else:
                if (self.poffset.size == 1):
                    pof = np.full(d[2], self.poffset)
                else:
                    pof = self.poffset

        #astropy_mpl_style['axes.grid'] = False
        #plt.style.use(astropy_mpl_style)
        #plt.figure()
        #for k in range(d[2]):
            # plt.imshow(noise[:,:,22], cmap='viridis')
            # plt.imshow(noise[:,:,22], cmap='gray')
            # plt.show()
        #    plt.imshow(x[:,:,k], cmap='viridis')
        #    plt.plot(x[:, :, k])

        #plt.colorbar()
        #plt.show()
        # voir mappable vmin vmax
        # colorbar, min(x(,,k))*psc(k)+pof(k),max(x(,,k))*psc(k)+pof(k)
        #plt.pause(1)

    def model(self, params, noscale=None):
        """
        DOCUMENT
        model_cube = fitobj(model, x)
        Create a model cube from a set of parameter maps. Like for the
        VIEW method, NOSCALE should almost always be set to 1.

        parameter maps array first dimension number of profile function parameters
                            ny
                            nx

        SEE ALSO: cubefit, view

        """
    # /!\ WARNING: this function is for visualisation, it is "inlined" in eval
    #  for optimization
        if self.dbg:
            print("DBG CALL model")
            print(f"with params[4,4,:] {params[4,4,:]}")
            print(f"with params shape {params.shape}")

        params_dim = params.shape
        # d = len(x)
   #     print(f"params_dim {params_dim}")

        # TODO for debug purpose dont scale
        noscale = 1

        # if (noscale is not None):
        if (noscale):
            psc = np.ones(params_dim[2])
            pof = np.zeros(params_dim[2])
            if self.dbg:
                print(f"psc {psc}")
                print(f"pof {pof}")

        else:
            psc = self.pscale
            pof = self.poffset
            if self.dbg:
                print(f"noscale {noscale}")
                print(f"psc {psc}")
                print(f"pof {pof}")

        if (psc is None):
            if self.dbg:
                print("psc is None")
            xs = params
        else:
            # xs = x * psc(-,-,)
            xs = params * psc
            #xs = params * psc[np.newaxis, np.newaxis, :]
            if self.dbg:
                print(f"xs {xs}")

        if (pof is not None):
            #xs += pof(-,-,);
            xs += pof[np.newaxis, np.newaxis, :]
            #xs += np.reshape(pof, (1, 1, pof.shape[0]))

        if (self.ptweak is not None):
            # TODO use_method, ptweak, xs, derivatives
            # xs et derivatives

            xs, derivatives = self.ptweak(xs)
            print("use_method ptweak is not None")

        nx = params_dim[0]
        #print(f"nx {nx}")

        ny = params_dim[1]
        #print(f"ny {ny}")

        #print(f"fcn_x {self.profile_xdata}")

        #print(f"xs[0,0,:] {xs[0,0,:]}")
        #TODO choose return tuple
        if self.dbg:
            print("dbgla")
        nz = self.profile(self.profile_xdata, *xs[0, 0, :])[0].size
        #nz = self.profile(self.profile_xdata, *xs[0, 0, :]).size
        #TODO wrong nz for doppler

        # y = np.array(double, nx, ny, nz)
        y = np.zeros((nx, ny, nz))
        #y = np.zeros((nz, nx, ny))

        for i in range(nx):
            for j in range(ny):
                y[i,j,:] = self.profile(self.profile_xdata, *xs[i,j,:])[0]
                #y[:=,i,j] = self.profile(self.profile_xdata, *xs[i,j,:])[0]
                # y[:, i, j] = self.profile(xs[:, i, j], *self.profile_xdata)
        #print(f"y {y[4,4,:]}")
        #print(f"y.shape {y.shape}")
        #print(f"y[50,50,:] {y[50,50,:]}")
        #print(f"y[100,100,:] {y[100,100,:]}")
        #print("dbg end model func")
        return y

    # TODO check for unapply and remove the apply inside ?
    ## denormalize paramaters from -1,1 boundaries to physical unit
    # see normalize comment
    def denormalize_parameters(self, x):
        """
        apply offset and scale and returnmaps for checking ?
        ie noscale = 1 returnmaps = 1
        """

        if self.dbg:
            print("DBG CALL denormalize parameters with x")
            print("apply_offset_scale with x")
            print(f"{x}")
            #print(f"x[49,49,:]{x[49,49,]}")
            #print(f"x[50,50,:]{x[50,50,]}")
            #print(f"x[51,51,:]{x[51,51,]}")

            print(f"with pscale {self.pscale}")
            print(f"with poffset {self.poffset}")

            print(f"with scale {self.scale}")
            print(f"with delta {self.delta}")


        # d = x.shape
        # nx = d[0]
        # ny = d[1]
        # res = 0.

        if (self.pscale is None):
            # which x ?
            xs = x
        else:
            if self.dbg:
                print(f"shape pscale {self.pscale.shape}")
                print(f"pscale {self.pscale}")
                print(f"pscale[0] {self.pscale[0]}")
                print(f"shape x {x.shape}")
                print(f"x[0,0,:] {x[0,0,:]}")
                print(f"x[50,50,:] {x[50,50,:]}")
                # print(f"shape xs {xs.shape}")
            #TODO faire une boucle
            # ValueError: operands could not be broadcast together
            # with shapes (433,206,197) (1,1,3)
            # is xs = x * pscale(-,-,) in yorick
            #xs = x * np.reshape(self.pscale,(1,1,self.pscale.shape[0]))
            # xs = x * self.pscale[np.newaxis,np.newaxis,:]
            # xs = x[1:,:,:] * self.pscale[:, np.newaxis,np.newaxis]
            xs = x * self.pscale[np.newaxis,np.newaxis,:]

            #or k in range(x[]):
            #    xs[k,:] *= self.pscale


        if (self.poffset is not None):
            # xs += np.reshape(self.poffset,(1,1,self.poffset.shape[0]))
            xs += self.poffset[np.newaxis,np.newaxis,:]

        return xs


        # TODO returnmaps function
#
#         returnmaps = 1
#         if (returnmaps):
#             maps = np.zeros((nx, ny, d.shape[3]+1))
#         # //  tot=cube(1,1,)*0.
#         for i in range(nx):
#             for j in range(ny):
#                 if (any(self.weight[i, j, ])):
#                     spectrum = self.data[i, j, :]
#                     # TODO adapt with jacobian function
#                     model = self.profile(self.profile_xdata, xs[i, j, :], deriv=1)
#                     model_jacobian = self.profile_jacobian(self.profile_xdata, xs[i, j, :])
#                     grad = model_jacobian
#                     if (self.pscale is not None):
#                         grad *= self.pscale[:, ]
#                     if (derivatives is not None):
#                         grad *= derivatives[i, j, :, ]
#                     atom = (model - spectrum) * self.weight[i, j, ]
#                     # //        tot+=atom
#                     if (returnmaps):
#                         maps[i, j, 0] = sum(atom**2)
#                     # returnmaps force to 1
#                     # else:
#                     #    res += sum(atom**2)
#                     #    gx[i,j,] += np.sum((grad * atom[:,]),0) *2.
#         # TODO print
#         # //  window,34
#         # //  plg, tot
#
    def normalize_parameters(self, x):
        #  if (noscale) {
        #     x = x; // copy input in order to not modify it!
        #     if (!is_void(psoffset)) x(..,) -= poffset(-,-,);
        #     if (!is_void(pscale))   x(..,) /= pscale(-,-,);
        # }

        if self.dbg:
            print("DBG CALL normalize parameters with x")
        # normalize paramaters put roughly in -1,1 boundaries according to user
        # expectation
        #x = np.array(x)
        x_norm = np.copy(x)
        #print(f"x[49,49,:]{x[49,49,]}")
        #print(f"x[50,50,:]{x[50,50,]}")
        #print(f"x[51,51,:]{x[51,51,]}")

        #print(f"with pscale {self.pscale}")
        #print(f"with poffset {self.poffset}")

        #print(f"with scale {self.scale}")
        #print(f"with delta {self.delta}")


        # nx = (d=dimsof(x))(0)
        # d = x.shape
        # nx = x.shape[0]
        #nx = len(x.shape)
        if self.psoffset is not None:
            #x[:,:,] -= self.poffset[np.newaxis,np.newaxis,:]
            x_norm -= self.poffset[np.newaxis,np.newaxis,:]
        if self.pscale is not None:
            #x[:,:,] /= self.pscale[np.newaxis,np.newaxis,:]
            x_norm /= self.pscale[np.newaxis,np.newaxis,:]

        return x_norm

# fonction objective a faire manger a vmlmb qui doit retourner
# un tuple fonction a minimiser + gradient de cette fonction
# x cube carte parametre nx,ny,np
    # x is params ?
    def eval(self, x, noscale=None, returnmaps=None):
        """
        /* DOCUMENT cubefit.eval method
                criterion = fitobj(eval, x, gx)
                or maps = fitobj(eval, x, noscale=1, returnmaps=1)

        Like for the VIEW method, NOSCALE should almost always be set to 1
        when calling this method manually.

        Criterion is the sum of a weighted chi2 and a regularisation term.
        To remove regularisation altogether, set regularisation to nil.

        If RETURNMAPS is one, returns a stack of maps for each component of
        the criterion (chi2 term last) instead of the integral. pour debugger

        Side effect: if data is not known but weight is None, set
        weight to a cube of ones.

        SEE ALSO: cubefit, cubefit.l1l2, cubefit.markov,
                cubefit.model, cubefit.fit
        */

        """
        # beware: "model" below is a local variable, not the method.
        # local scale, delta, cube, fcn_x, fcn, weight, regularisation,
        # pscale, poffset, ptweak, decorrelate
        # restore, use, scale, delta, cube, fcn_x, fcn, weight, regularisation,
        # pscale, poffset, ptweak, decorrelate
        if self.dbg:
            print("DBG CALL eval func with x")
            print(f"{x}")

        # eval() requires self.data to be set
        if self.data is None:
            raise ValueError("eval() requires self.data to be set")

        # weights default to 1
        if self.weight is None:
            self.weight = np.ones(self.data.shape)

        #print(f"x[49,49,:]{x[49,49,]}")
        #print(f"x[50,50,:]{x[50,50,]}")
        #print(f"x[51,51,:]{x[51,51,]}")

        #x = np.array(x)
        d = x.shape
        #TODO index
        nx = d[0]
        ny = d[1]
        res = 0.
        # gx => fonction gradient
        gx = np.zeros(d)

        if (noscale):
            x = self.normalize_parameters(x)

        xs = self.denormalize_parameters(x)

        derivatives = None

        if self.dbg:
            print(f"shape xs {xs.shape}")

        if (self.ptweak is not None):
            # TODO use_method, ptweak, xs, derivatives
            xs, derivatives = self.ptweak(xs)
            print(f"use_method")

        if ((derivatives is not None) and (derivatives.shape != xs.shape)):
            raise Exception("ptweak derivatives should be []\
                            or same size as parameter array")

        # if (returnmaps):
        if (self.dbg):
            # TODO pourquoi ajouter 1 a la dim des parametres ?
            self.dbg_data["maps"] = np.zeros((nx, ny, d[2]+1))
        # //  tot=cube(1,1,)*0.
        # print(f"nx {nx} ny {ny}")
        for i in range(nx):
            for j in range(ny):
                # print(f"nx {i} ny {j}")
                #print(f"self.weight[197, j,: ] {self.weight[197, j,: ]}")
                if (any(self.weight[i, j,:])):

                    #print(f"cube shape {self.data.shape}")
                    spectrum = self.data[i, j,: ]
                    # TODO adapt with jacobian function
                    # model = self.profile(self.profile_xdata, xs[i,j,:], deriv=1)
                    #print(f"xs[i, j, :] {xs[i, j, :]}")
                    #print(f"self.profile_xdata {self.profile_xdata}")
                    # fcn_x array 1 dim contenant les wavelength (size nz)
                    # model = spectre unique d flux au pixel i,j (size nz)
                    # *xs[i,j,:] np parameters array minimum size_np >= 3
                    model, model_jacobian = self.profile(self.profile_xdata, *xs[i, j, :])
                    # model_jacobian = self.profile_jacobian(self.profile_xdata, xs[i, j, :])
                    # size_grad(nz,np)
                    grad = model_jacobian
                    if (self.pscale is not None):
                        # TODO add comment for a more pythonic code

                        for k in range(self.profile_xdata.size):
                            grad[k,:] *= self.pscale

                    if (derivatives is not None):
                        grad *= derivatives[i, j, :, ]

                    #print(f"model shape {model.shape}")
                    #print(f"spectrum shape {spectrum.shape}")
                    # print(f"atom shape {atom.shape}")
                    #print(f"weight shape {self.weight.shape}")

                    atom = (model - spectrum) * self.weight[i, j,: ]
                    # //        tot+=atom

                    if (self.dbg):
                        self.dbg_data["maps"][i, j, 0] = sum(atom**2)

                    # if (returnmaps):
                    #    maps[i, j, 0] = sum(atom**2)

                    res += sum(atom**2)
                    # TODO sum
                    #print(f"grad shape {grad.shape}")
                    #print(f"atom shape {atom.shape}")
                    #print(f"weight shape {self.weight.shape}")
                    # yorick line gx(i,j,) += (grad * atom(,-))(sum,) *2.;
                    gx[i, j, : ] += np.sum((grad * atom[:,np.newaxis ]), axis=0) * 2.
        # //  window,34
        # //  plg, tot
        # TODO implemente regularisation
        if self.dbg:
            print(f" not callable regularisation {not callable(self.regularisation)}")
        # if (!is_func(regularisation)) goto out
        #if (not callable(self.regularisation)):
            # if (self.regularisation is not None):
            print("goto out")
            # return None
        return res, gx

        # creation d une carte plus grande pour eviter effet de bord des bords
        xbig = np.zeros((d[0]*2, d[1]*2, d[2]))
        xbig[:d[0], :d[1], :] = x
        # on groupe par trois
        print(f"shape x[:,0:1:-1,:] {x[:,0:1:-1,:].shape}")
        print(f"shape y=np.flip(x,1) {np.flip(x,1).shape}")
        # xbig[:d[0], d[1]+1:,:] = x[:,0:1:-1,:]

        # TODO d[1]+1 ou pas?
        # print(f"xbig[:d[0], d[1]+1:,:].shape {xbig[:d[0], d[1]+1:,:].shape}")
        xbig[:d[0], d[1]:, :] = np.flip(x, 1)
        xbig[d[0]+1:, :, :] = xbig[d[0]:1:-1, :, :]
        # g is a grad
        g = np.empty(xbig.shape)
        # TODO for debug only
        for k in range(d[2]):
            # TypeError: unsupported operand type(s) for /: 'tuple' and 'float'
            #tmp = self.regularisation(xbig[:, :, k], g,scale=self.scale[k], delta=self.delta[k],returnmap=returnmaps) / 4.
            if (not self.dbg):
                tmp = self.regularisation(xbig[:, :, k], g,self.scale[k], self.delta[k]) / 4.
            else:
                if self.dbg:
                    print("dbgla")
                tmp = self.regularisation(xbig[:, :, k], g,self.scale[k], self.delta[k],returnmaps) / 4.
            # TODO change
            if (self.dbg):
                self.dbg_data["maps"][:, :, k] = tmp[:d[0], :d[1]]

            # if (returnmaps):
            #    maps[:,:,k] =  tmp[:d[0], :d[1]]
            #tmp=0
            res += tmp
            # TODO indices commence a 0 ?
            gx[:, :, k] += g[0:d[0]:+1, 1:d[1]:+1]
                       # Compared to Yorick version :
                       # TODO modify yorick version if needed to compare
                       # no need to add the 4 quadrants since
                       # we divide by 4 above +\
                       #g[0:d[0]:+1, -1:d[1]+1:-1] +\
                       #g[-1:d[0]+1:-1, 1:d[1]:+1] +\
                       #g[-1:d[0]+1:-1, -1:d[1]+1:-1]

        # if (returnmaps):
        #    return maps

        # a voir a la fin faible priorite
        if (self.decorrelate is not None):
            dd = self.decorrelate.shape
            #if (dd.shape[0] != 2):
            if (len(dd.shape) != 2):
                # TODO reform
                self.decorrelate = np.reshape(self.decorrelate, (2, self.decorrelate.size, 1))
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
                correl = cubefit.corr(xy, grad, deriv=1)
                res += w * correl**pow
                gx[:, :, [i1, i2]] += w*pow*correl**(pow-1)*grad

        #  out:
        return res, gx

    def printer(self, output, iter, eval, cpu, fx, gnorm, steplen, x):
        """
        /* DOCUMENT cubeview.printer method
                fitobj, printer, output, iter, eval, cpu, fx, gnorm, steplen, x

         */
        """
        d = x.shape

        npairs = 0
        if (self.decorrelate is not None):
            dd = self.decorrelate.shape
            if (dd.shape[0] != 2):
                self.decorrelate = np.reshape(self.decorrelate,
                                              [2, self.decorrelate.size, 1])
                dd = self.decorrelate.shape
            npairs = dd.shape[2]

        if (iter == 0):
            # write, output, format="# %s",
            # "ITER  EVAL   CPU (ms)        FUNC               GNORM   STEPLEN"
            print(f"ITER EVAL CPU (ms)\t FUNC\t  GNORM\t  STEPLEN")
            for k in range(d.shape[3]):
                # write, output, format=" REGUL[%i]", k
                print(f"REGUL[{k}]")

            for pair in range(npairs):
                i1 = np.floor(self.decorrelate(1, pair))
                i2 = np.floor(self.decorrelate(2, pair))
                #  write, format=" CORR[%i,%i]", i1, i2
                print(f"CORR[{i1},{i2}]")

            #  write, output, format="\n# %s",
            print(f"------------------------------------------------------")
            for k in range(d.shape[3]):
                #  write, output, format="%s","---------"
                print(f"---------")
            for pair in range(npairs):
                #  write, output, format="%s","----------"
                print(f"----------")
            #  write, output, format="%s\n", ""
            print(f"")
        #  format = " %5d %5d %10.3f  %+-24.15e%-9.1e%-9.1e"
        #  write, output, format=" %5d %5d %10.3f  %+-24.15e%-9.1e%-9.1e",
        #  iter, eval, cpu, fx, gnorm, step

        #  print(f"{iter}  {eval} {cpu} {fx} {gnorm} {step}")

        for k in range(d.shape[3]):
            # write, output, format="%-9.1e", regularisation(x(,,k), g,
            print(f'{self.regularisation(x[:,:,k], scale=self.scale[k],delta=self.delta[k])}')

        for pair in range(npairs):
            i1 = np.floor(self.decorrelate(1, pair))
            i2 = np.floor(self.decorrelate(2, pair))
            w = self.decorrelate(2, pair)
            if (dd.shape[2] >= 4):
                pow = self.decorrelate(4, pair)
            else:
                pow = 2
            xy = x[:, :, [i1, i2]]
            correl = cubefit.corr(xy)
            # write, output, format="%-10.1e", w * correl**pow
            print(f"{w*correl**pow}")

        print(f"")

    def criterion(self, x):
        return self.eval(x)[0]

    # __call__ ??  *fout *gout mis a None pour dbg
    # fout gout
    # Accept all vmlmb keywords (vmlmb_kwargs).
    # lower and upper need special treatment (for rescaling).
    # There's a bug in optm which requires verb > 0.
    def fit(self,x,
            lower=None, upper=None,
            verb=1,
            **vmlmb_kwargs):
        """Fit model to self.data.

        Wrapper around:
          (result, fx, gx, status) = vmlmb(self.eval, x,
                                           lower=lower, upper=upper,
                                           verb=verb,
                                           **vmlmb_kwargs)

        Arguments:
        x -- the stack of parameters to fit. Rescaled according to
          self.pscale and self.poffset prior to calling vmlmb and
          scaled back to physical values in self.eval(). RESULT is
          also scaled back before being returned.

        Keyword arguments:
        All keywords are passed untouched to vmlmb, except lower and
        upper which are rescaled as x.
        """
        if self.dbg:
            #pass
            print("DBG CALL fit func with x")
            print(f"{x}")
            print(f"{type(x)}")
            # print(f"x[49,49,:]{x[49,49,]}")
            # print(f"x[50,50,:]{x[50,50,]}")
            # print(f"x[51,51,:]{x[51,51,]}")

            # print(f"with pscale {self.pscale}")
            # print(f"with poffset {self:q.poffset}")

            # print(f"with scale {self.scale}")
            # print(f"with delta {self.delta}")

            # nx = (d=dimsof(x))(0)
        d = x.shape
        nx = x.shape[-1]
        if self.dbg:
            print(f"nx is {nx}")

        # pour DBG
        # op_viewer = None
        # view is a yorcik function
        # view = None

        # move inside a graphic/draw function
        # if (verb and (op_viewer is not None) and (view is not None)
        #    and callable(op_viewer) and callable(view)):
        #    for k in range(nx):
        #        #TODO cubeview
        #        winkill, k
        #        window, k
        #        cv_vpaspect,d(2),d(3)
        #        # force les axes de la fenetre pour ratio


        ## normalize paramaters put roughly in -1,1 boundaries according to user
        # expectation
        if (self.poffset is not None):
            print(f"self.poffset {self.poffset}")
            if (self.poffset.size != nx):
                print(f"self.poffset.size {self.poffset.size}")
                print(f"self.poffset.shape {self.poffset.shape}")
                print(f"self.poffset {self.poffset}")
                print(f"nx {nx}")
                self.poffset = np.full(nx, self.poffset)
            for k in range(nx):
                x[:,:, k] -= self.poffset[k]
            if (lower is not None):
                for k in range(nx):
                    lower[:,:, k] -= self.poffset[k]
            if (upper is not None):
                for k in range(nx):
                    upper[:,:, k] -= self.poffset[k]
        if self.dbg:
            print("after apply poffset")
            # print(f"{x}")
            print(f"x[49,49,:]{x[49,49,]}")
            print(f"x[50,50,:]{x[50,50,]}")
            print(f"x[51,51,:]{x[51,51,]}")


        if (self.pscale is not None):
            print("APPLY PSCALE")
            if (self.pscale.size != nx):
                self.pscale = np.full(nx, self.pscale)

            print(f"nx  {nx}")
            for k in range(nx):
                x[:,:, k] /= self.pscale[k]
                print(f"x[49,49,:]{x[49,49,]}")

            if self.dbg:
                print("in2 poffset")
                # print(f"{x}")
                print(f"x[49,49,:]{x[49,49,]}")
                print(f"x[50,50,:]{x[50,50,]}")
                print(f"x[51,51,:]{x[51,51,]}")


            if (lower is not None):
                for k in range(nx):
                    lower[:,:, k] /= self.pscale[k]
            if (upper is not None):
                for k in range(nx):
                    upper[:,:, k] /= self.pscale[k]

        if self.dbg:
            # end normalize
            print("after apply pscale")
            # print(f"{x}")
            print(f"x[49,49,:]{x[49,49,]}")
            print(f"x[50,50,:]{x[50,50,]}")
            print(f"x[51,51,:]{x[51,51,]}")



        if (self.scale is None):
            #self.scale = np.full(d.shape[3], 1.)
            self.scale = np.ones(d[2])
        else:
            if (self.scale.size == 1):
                self.scale = np.full(d[2], self.scale)

        if (self.delta is None):
            #self.delta = np.full(d.shape[3], 1.)
            self.delta = np.ones(d[2])
        else:
            if (self.delta.size == 1):
                self.delta = np.full(d[2], self.delta)

            # save, use, pscale, poffset, scale, delta

            # TODO omnipack op_f => eval
            # result = op_mnb(op_f, x, fout, gout, extra=use(), \
            #        lower=lower, upper=upper, method=method, \
            #        mem=mem, verb=verb, quiet=quiet,\
            #    viewer=op_viewer, printer=op_printer,\
            #    maxiter=maxiter, maxeval=maxeval,output=output,\
            #    frtol=frtol, fatol=fatol,\
            #    sftol=sftol, sgtol=sgtol, sxtol=sxtol )

        (result, fx, gx, status) = vmlmb(self.eval, x,
                                         lower=lower, upper=upper,
                                         verb=verb,
                                         **vmlmb_kwargs)
            # restore, use, pscale, poffset
        # denormalize?
        if (self.pscale is not None):
            for k in range(nx):
                result[:, k] *= self.pscale[k]

        if (self.poffset is not None):
            for k in range(nx):
                result[:, k] += self.poffset[k]

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

def noreg(x, *grad_obj,
                 scale=None, delta=None, returnmap=None):
    return x

def markov(x, *grad_obj,
                   scale=None, delta=None, returnmap=None):
    """
    DOCUMENT cubefit.markov(object,grad_object[,scale=,delta=])

    delta^2 . sum [|dx|/delta -ln(1+|dx|/delta) + |dy|/delta -ln(1+|dy|/delta)]
    where |dx| (m,n) = (o(m,n)-o(m-1,n))/scale is the x component of the
    object gradient normalized by scale.

    KEYWORDS :
    scale : Scale factor applied to Do (default value is 1)
    delta : Threshold on the gradient for switching between linear and
    quadratic behavour. When delta tends to infinity, the criterion
    becomes purely quadratic.

    AUTHOR: Damien Gratadour, borrowed from Yoda.
    */
    """
    # TODO scale et delta already in self
    if (scale is None):
        scale = 1.
    if (delta is None):
        delta = 1.

    # TODO object-roll?
    # dx = (object-roll(object,[1, 0])) / (delta * scale)
    # dy = (object-roll(object,[0, 1])) / (delta * scale)

    # TODO index should be [0,-1] ?
    dx = (x - np.roll(x, [1, 0])) / (delta * scale)
    dy = (x - np.roll(x, [0, 1])) / (delta * scale)

    # map is a python reserved keyword
    amap = np.abs(dx) - np.log(1. + np.abs(dx)) + \
        np.abs(dy) - np.log(1. + np.abs(dy))

    if (returnmap):
        return (delta**2) * amap

    crit = (delta**2) * np.sum(amap)

    dx /= (1. + abs(dx))
    dy /= (1. + abs(dy))

    # roll
    grad_obj = (dx - np.roll(dx, [-1, 0]) + dy - np.roll(dy, [0, -1])) * \
               (delta / scale)

    return crit, grad_obj

# voir si fonction de module plutot que de class satic
def l1l2(x, *grad_obj,
                 scale=None, delta=None, returnmap=None):
    """
    DOCUMENT cubefit.l1l2(object,grad_object[,scale=,delta=])
    delta^2 . sum [|dx|/delta -ln(1+|dx|/delta)+ |dy|/delta -ln(1+|dy|/delta)]
    where |dx| (m,n) = [o(m,n)-o(m-1,n)]/scale
    is the x component of the  object gradient normalized by scale.

    KEYWORDS :
    scale : Scale factor applied to Do (default value is 1)
    delta : Threshold on the gradient for switching between linear
    and quadratic behavour. When delta tends to infinity,
    the criterion becomes purely quadratic.

    AUTHOR: Damien Gratadour, borrowed from Yoda.
    """

    # if (!is_set(scale)) scale = 1.
    # if (!is_set(delta)) delta = 1.
    # TODO scale et delta already in self
    if (scale is None):
        scale = 1.
    if (delta is None):
        delta = 1.

    # TODO object-roll ?
    # realise un shift
    dx = (x - np.roll(x, [1, 0])) / (delta * scale)
    dy = (x - np.roll(x, [0, 1])) / (delta * scale)

    r = np.sqrt(dx**2 + dy**2)

    # map is a reserved python keyword
    amap = r - np.log(1.+r)

    if (returnmap):
        return (delta**2) * amap

    crit = (delta**2) * np.sum(amap)

    dx /= (1. + r)
    dy /= (1. + r)

    # TODO dx-roll is that ok (x - roll) en place de object-roll
    # ou est ce object - roll ?
    grad_obj = (dx - np.roll(dx, [-1, 0]) + dy - np.roll(dy, [0, -1])) * \
               (delta / scale)

    return crit, grad_obj

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

# cubefit = save(
#        //methods
#        new=new, view=view, model=model,
#        printer=printer, fit=fit, eval=eval,
#        // static functions
#        l1l2=l1l2, markov=markov, corr=corr,
#        op_f=op_f, op_viewer=op_viewer,
#        op_printer=op_printer
#        )

# // clean namespace
# restore, scratch


#
# // fog' = g'.f'og
# // (1/v)' = v'.(1/x)'ov =-v'/v^2
# // (u.v)' = u'.v + u.v'
# // (u/v)' = u'/v - u.v'/v^2 = (u'.v - u.v') / v^2.
# u(xj) = Sum [(xi-<x>)(yi-<y>)] = Sum [xi.(yi-<y>)] - <x> Sum (yi-<y>)
# u'(xj) = (yj-<y>) - d<x>/dxj * Sum(yi-<y>) = (yj-<y>) - < (yi -<y>) >
# v(xj) = sqrt( Sum (xi-<x>)^2 ) * sqrt(Sum (yi-<y>)^2)
#
# */

def test_gauss():
    # debug model
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

    print("test gauss call")
    y = gauss(gauss_xdata,*gauss_param)[0]
    y_noise = y + np.random.standard_normal(y.size) * sigma

    #plt.figure()
    #plt.plot(gauss_xdata, y)
    #plt.show()


    model_param_gauss = np.zeros((nx,ny,nparms))

    for i in range(nx):
        for j in range(ny):
            model_param_gauss[i,j,:] = gauss_param

    print(f" model_param_gauss[4,4,:]{model_param_gauss[4,4,:]}")

    fcn_gauss = gauss
    fcn_x_gauss = gauss_xdata

    # cube_gauss[...,0:nz] = gauss_param
    # np.c_[cube_gauss,gauss_param]

    print("creating line obj")
    lineobj_gauss=DopplerLines(lines=gauss_param[0], profile=ngauss)

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

    # debug eval
    print(f"============#debug eval")

    # TODO nz=433
    # cube_empty = np.ndarray((nx, ny, nz))

    # on recree un obj cubefit
    fitobj_eval_gauss = CubeModel(cube_noise_gauss,
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
    print(f"fx {fx}")
    print(f"gx {gx}")
    print(f"status {status}")


def test():
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

    model_param_doppler = np.zeros((nx, ny, doppler_param.size))

    # print(f"doppler_param is {doppler_param}")

    # TODO find a more pythonic expression
    for i in range(doppler_param.size):
        model_param_doppler[:, :, i] = doppler_param[i]

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
    #            pscale,poffset, ptweak,regularisation=None,decorrelate=None)

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
    cube_empty = np.ndarray((nx, ny, nz))

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
    #def vmlmb_printer(output, iters, evals, rejects, t, x, fx, gx, pgnorm, alpha, fg):
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
    #write_fits(res_x, "mycube_res_x_dop.fits")
    #write_fits(fx, "mycube_res_fx_dop.fits")
    #write_fits(gx, "mycube_res_gx_dop.fits")
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

    model_param_doppler = np.zeros((nx, ny, doppler_param.size))

    # TODO find a more pythonic expression
    for i in range(doppler_param.size):
        model_param_doppler[:, :, i] = doppler_param[i]

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
    cube_empty = np.ndarray((nx, ny, nz))

    # on recree un obj cubefit
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

    print(f"x0_test {x0_test}")
    print(f"calling fit function")

    (res_x, fx, gx, status) = fitobj_eval_doppler.fit( x0_test)

    print("# Iter.   Time (ms)   Eval. Reject.    Obj. Func.      Grad.  Step")
    print("res_x for test_fit")
    print(f"{res_x}")
    print(f"fx {fx}")
    print(f"gx {gx}")
    print(f"status {status}")




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
    psigma = sigma
    tmp_cube = np.copy(cube)
    tmp_cube = cube + np.random.standard_normal(cube.shape) * psigma
    return tmp_cube


"""Unconstrained non-linear optimization problems from MINPACK-1 Project.

The Python functions implementing the MINPACK-1 problems are named `prob_P`
with `P` the problem number in the range 1:18.  Any problem function can be
called as:

    prob_P() -> yields the name of the problem.

    prob_P(0, n=None, factor=None) -> yields the starting point for the problem
        as a vector `x` of `n` elements which is a multiple (times `factor`,
        default 1) of the standard starting point. For the 7-th problem the
        standard starting point is 0, so in this case, if `factor` is not
        unity, then the function returns `x` filled with `factor`. The values
        of `n` for problems 1, 2, 3, 4, 5, 10, 11, 12, 16, and 17 are 3, 6, 3,
        2, 3, 2, 4, 3, 2, and 4, respectively. For problem 7, `n` may be 2 or
        greater but is usually 6 or 9. For problems 6, 8, 9, 13, 14, 15 and 18,
        `n` may be variable, however it must be even for problem 14, a multiple
        of 4 for problem 15, and not greater than 50 for problem 18.

    prob_P(1, x) -> yields the value of the objective function of the problem.
        `x` is the parameter array: a vector of length `n`.

    prob(2, x) -> yields the gradient of the objective function of the problem.

Since the execution time may change, you can compare the outputs after
filtering with:

    sed -e 's/^\( *[0-9]*\) *[^ ]*/\1/'

to get rid of the 2nd column.
"""

# syntax introduit en python 3.8 / => parametre forcement positionnel
# python 3.0 * => obligation de nommer ces parametres
# def test(probs = range(1, 19), /, *, n=None, factor=None, fmin=0,

# def test_optm_vmlmb(probs = range(1, 19),  n=None, factor=None, fmin=0,
#         mem="max", verb=1, blmvm=False, output=sys.stdout, **kwds):
#    """Run one or several tests from the MINPACK-1 Project.
#
#    Usage:
#
#         optm_minpack1_test(probs=range(1,19))
#
# Arg `probs` is a single problem number (in the range 1:18) or a list of
#    problem numbers.  By default, all problems are tested with keywords
#    `mem="max"`, `fmin=0`, and `verb=1`.
#
#    Keywords:
#      n - Size of the problem.
#
#      factor - Scaling factor for the starting point.
#
#      mem, fmin, lnsrch,
#      xtiny, epsilon, f2nd,
#      ftol, gtol, xtol,
#      blmvm, maxiter, maxeval,
#      verb, cputime, output - These keywords are passed to `optm_vmlmb` (which
#          to see).  All problems can be tested with `fmin=0`.  By default,
#       `mem="max"` to indicate that the number of memorized previous iterates
#          should be equal to the size of the problem.
#
#    See also: `optm.vmlmb`.
#
#    """
#    # Output stream.
#    if type(output) == str:
#        output = open(output, mode="a")
#    if probs is None:
#        probs = range(1, 19)
#    elif isinstance(probs, int):
#        probs = [probs]
#    for j in probs:
#        f = eval(f"prob_{j:d}")
#        x0 = f(0, n=n, factor=factor)
#        if mem == "max":
#            m = x0.size
#        else:
#            m = mem
#        if verb != 0:
#            name = f()
#            if blmvm:
#                algo = "BLMVM"
#            else:
#                algo = "VMLMB"
# print(f"# MINPACK-1 Unconstrained Pblem #{j:d} ({x0.size} vars): {f():s}.",
#                  file=output)
#            print(f"# Algorithm: {algo:s} (mem={m:d}).",file=output)
#        (x, fx, gx, status) = optm.vmlmb(lambda x: (f(1, x), f(2, x)), x0,
#                                         mem=m, blmvm=blmvm, fmin=fmin,
#                                         verb=verb, output=output, **kwds)
#
# -----------------------------------------------------------------------


def numerical_gradient3(f, x, epsilon=1e-6):
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


if __name__ == '__main__':
    testpourgauss = 0
    if testpourgauss:
        test_gauss()
    else:
        test_fit()
