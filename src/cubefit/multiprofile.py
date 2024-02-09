#!/usr/bin/env python3

import numpy as np
# import math
# from astropy.modeling.functional_models import Moffat1D


import matplotlib.pyplot as plt
from scipy import optimize

"""multiprofile.py is a helper library for use with curve_fit. It allows
    creating complex user functions by adding up already existing, simpler
    ones.

    Remember the curve_fit calling sequence:
      result=lmfit(f, x, a, y, w, ...).
    where F is the model function to fit, X the "independent
    variables", which can indeed be anything required by F, A is the
    vector of parameters, and Y the observed values. lmfit finds the
    optimal values for A, in order to minimize the distance between
    F(X,A) and Y.

    In order for curve_fit to converge efficiently, it is better if F is
    able to compute its own gradient when called as f(x, *a)
    Writing model functions that provide derivatives can be
    tiresome and error-prone. The goal of multiprofile.py is to make
    this process faster and more reliable, by allowing one to build
    complex model functions from pre-existing, well-optimized and
    well-tested primitives with derivatives.

    Of course, multiprofile.py is limited to a particular sub-set of
    all the types of functions one might want to use curve_fit on: it
    concentrates on the case where the model function is the sum of
    simpler profiles, either of the same type (produced by the same
    primitive function, like for Gaussian), or different from each other
    (e.g. a Gaussian plus a Moffat).

    To do that, the multiprofile model function mp_func(x, *a)
    accepts as first positional argument X a complex object
    which has to be set-up using the helper function mp_setx. X
    contains in itself a description of the profile: the primitive
    functions to use, the number of instances of each type of
    primitive function, and, naturally, whatever X parameter each
    primitive function requires to function properly. You then call
    curve_fit with mp_func as its first argument and this complex X as its
    second argument.

   Example
   -------
    For instance, assume you want to fit an observed Y, which seems to
    be well described as a sum of 3 Gaussian profiles. The usual
    process would require you to write a new function (e.g. gaussx3)
    specifically for this purpose. gaussx3() would have to compute
    both the sum of 3 Gaussian profiles, and the corresponding
    gradient.

    This is how you can do it with multiprofile.py (assuming x and y
    are already known, and you have found a reasonable first guess a1,
    a2 and a3 for each of the 3 components).

        from lineprofiles import gauss
        from multiprofile import *
        MultiX=mp_setx(profile=gauss, realX=x, npar=3, ncomp=3)
        a=(a1, a2, a3)
        result=mpfit(mp_func, MultiX, a, y)

    Now, assume you want to add a linear baseline to the three
    Gaussian profiles (note that linear() is provided by
    multiprofile.py). You have "guessed" as l0 and l1 the two
    corresponding parameters:

        linX=mp_setx(profile=linear, realX=x, npar=2)
        MultiX=mp_setx(profile=gauss, realX=x, npar=3, ncomp=3, more=linX)
        a=(a1, a2, a3, [l0, l1])
        result=mpfit(mp_func, MultiX, a, y)

   FUNCTIONS PROVIDED BY MULTIPROFILE.PY
    mp_func : the F parameter to curve_fit when using multiprofile.i
    mp_setx : helper function to set-up the complex model function
    mp_getx : reverse from mp_setx
    mp_seta : helper function to combine individual first guesses for
             each component into a first guess for the complex model
             function (use GET keyword fro the reverse)
    linear : a*x+b, with curve_fit-friendly calling sequence and
             derivatives
    linear2d : a*x+b*y+c, with curve_fit-friendly calling sequence and
             derivatives
    poly_lmfit  : same as poly() (compute 1D polynomials), with
             curve_fit-friendly calling sequence and derivatives.

    offsetlines : an curve_fit-friendly function for fitting lines on a
             spectrum. It exemplifies advanced usage of multiprofile.
    ol_setx : helper function akin to mp_setx, for use with
             offsetlines.

   See Also
   --------
    lmfit, mp_func, mp_setx, mp_getx, mp_seta, linear,
    linear2d, poly_lmfit, gauss, gauss2d, moffat1d, moffat2d,
    offsetlines, ol_setx
"""

class MultiProfile:
    '''Manage multiple profile

    Parameters
    ----------
    profiles : callable or tuplet of callables
        One or several curvefit.lineprofiles profiles.
    nterms : int
        Number of parameters for each profile
    ncomps : int
        Number of components of each profile
    '''
    def __init__(self, profiles, nterms, ncomps, equals):
        # Ensure all parameters are in arrays
        self.profiles = np.atleast_1d(profiles)
        self.nterms   = np.atleast_1d(nterms)
        self.ncomps   = np.atleast_1d(ncomps)
        self.equals   = np.atleast_1d(equals)
        self.nterms_tot = np.sum(self.nterms*self.ncomps)

    def __call__(self, xdata, *params):
        '''self(xdata, *params) -> ydata, jacobian 
        Suitable as profile for CubeModel.
        '''
        assert len(params) == self.nterms_tot, "Wrong number of parameters" 

        params=np.asarray(params)

        ydata=np.zeros_like(xdata)
        jacobian=np.zeros(ydata.shape + np.asarray(params).shape)

        curp=0
        for k in range(len(self.profiles)):
            profile=self.profiles[k]
            nparms=self.nterms[k]
            nc=self.ncomps[k]
            for c in range(nc):
                y, j = profile(xdata, *params[curp:curp+nparms])
                ydata += y
                jacobian[..., curp:curp+nparms] = j
                curp += nparms

        return ydata, jacobian

    def cf_func(self, xdata, *params):
        '''self.cf_func(xdata, *params) -> ydata
        Suitable as objective function for curve_fit
        '''
        return self(xdata, *params)[0]

    def cf_jac(self, xdata, *params):
        '''self.jac() -> profile_jac
        Suitable as jac parameter for curve_fit
        '''
        return self(xdata, *params)[1]

# func mp_func(x,a,&grad,&comps,deriv=,returncomps=){
def mp_func(x, *a):
    """A general  purpose routine to easily create  multicomponents profiles.
    The parameter scheme may  seem odd, but it is intended to  be easily used
    with lmfit. See "multiprofile" for an introduction.

    X : this parameter should be set using MP_SETX (which see). It
        contains both the "real" independent variables and a
        description of the model, split into several components.
    A : vector of parameters. MP_SETA can be used to set it up. If base
        profile needs NPAR parameters, MP_FUNC will transmit
        A(NPAR*I+1:NPAR*(I+1)) to Ith instance of PROFILE. In case some
        parameters must be equal for every components, give their index
        in X (using mp_setx(equal=...)), and simply suppress this
        parameter from the list of parameters for all components except
        the first. For instance, if three components of parameters
        [a1,b1,c1], [a2,b2,c2], and [a3,b3,c3] are to be adjusted with
        the restriction that b1=b2=b3, A is of the form
        A=[a1,b1,c1,a2,c2,a3,c3] and equal=[2] in the call to mp_setx.
    GRAD : upon return, derivatives of output Y relative to each
        parameter in A, if DERIV is set to non void an non null. Can be
        used only if base profile function is able to compute derivatives.
    DERIV : whether or not to compute derivatives.
    COMPS : multiprofile can return the individual profiles of each
        component in a 4th positional argument. Set RETURNCOMPS for
        this to happen. COMPS(,C) is the C-th component.


    Examples
    --------

    from lineprofiles import gauss
    from multiprofile import *
    axis = np.linspace(-10, 10, 101)
    more = mp_setx(profile=linear, npar=2)
    x = mp_setx(profile=gauss, npar=3, ncomp=2, realX=axis, more=more)
    a=[10, -5, 2., 7, 4, 1.5, 100, 0.5]
    y=mp_func(x,a)
    plt.plot(y,axis)

    See Also
    --------
    lmfit, multiprofile
    """
    # TODO: python return more than one variable
    # mp_getx call
    # mp_getx, x, profile, realX, npar, ncomp, equal, more;
    # ie

    # profile=x[0]
    # realX=x[1]
    # npar=x[2]
    # ncomp=x[3]
    # equal=x[4]
    # more=x[5]
    profile, realX, npar, ncomp, equal, more = x
    # profile, realX, npar, ncomp, equal, more = mp_getx(x)

    # profile = func set by mp_set
    print(f"a[0:npar] {a[0:npar]}")
    y, gradc = profile(realX, *a[0:npar])

    # call jac
    # gradc=jac(realX,*a[0:npar])
    # TODO compute returncomps or ....
    # if (returncomps is not None and returncomps is True):
    #    comps=np.full((y.size),ncomp)
    #    #comps(,1)=y
    #    np.append(comps,y)

    # if (deriv):
    # jac call
    #grad = np.array((len(a)), y)
    grad = np.full((len(a), y.size), y)
    # grad(..,1:npar)=gradc ??
    grad[1:npar] = gradc

    if (equal is not None):
        for i in range(ncomp):
            y2, jac2 = profile(realX, a[i*npar+1:(i+1)*npar])
            # jac2=profile(realX,a[i*npar+1:(i+1)*npar])
            y = y+y2

            # if (returncomps):
            #    comps[:,i+1]=y2
            # if (deriv):
            grad[:, i*npar+1:(i+1)*npar] = gradc

        next = ncomp*npar+1
    else:
        template = np.array(npar)
        template[equal] = 1
        # where return tuple of array
        ind = np.where(template == 0)[0]
        template[equal] = a[equal]
        np2 = npar-equal.size
        for i in range(ncomp):
            template[ind] = a[npar+(i-1)*np2+1:npar+i*np2]
            # TODO on ecrase gradc ?
            y2, gradc = profile(realX, *template)
            # gradc=profile(realX,*template)
            y = y+y2
            # if (returncomps):
            #    comps[:,i+1]=y2
            # if (deriv):
            grad[:, npar+(i-1)*np2+1:npar+i*np2] = gradc[:, ind]
            grad[:, equal] += gradc[:, equal]
        next = npar+np2*(ncomp-1)+1

    if (more is not None):
        if (more[1] is None):
            more[1] = realX
        # a(next:0) => a[-a.size-1+next:]
        y += mp_func(more, a[-a.size-1+next:])
        # if (deriv):
        grad[:, next:a.size] = gradc

    return y, grad

#
# def mp_func(x, a, grad=None, comps=None, deriv=None, returncomps=None):
#    npar = len(x)
#    ncomp = len(a) // npar
#    y = np.zeros_like(x)
#    gradc = np.zeros((len(x), npar)) if grad is not None else None
#
#    if returncomps:
#        comps = np.zeros((len(x), ncomp))
#        comps[:, 0] = y
#
#    if deriv:
#        grad[..., :npar] = gradc
#
#    if deriv is None:
#        deriv = False
#
#    if not equal:
#        for i in range(1, ncomp):
#            y2, gradc = profile(realX,a[i*npar:(i+1)*npar],gradc,deriv=deriv)
#            y += y2
#            if returncomps:
#                comps[:, i] = y2
#            if deriv:
#                grad[..., i*npar:(i+1)*npar] = gradc
#        next = ncomp*npar
#
#    else:
#        template = np.ones(npar)
#        template[equal] = a[equal]
#        ind = np.where(template != 1)[0]
#        template[equal] = 1
#        np2 = npar - len(equal)
#
#        for i in range(1, ncomp):
#            template[ind] = a[npar+(i-1)*np2:npar+i*np2]
#            y2, gradc = profile(realX, template, gradc, deriv=deriv)
#            y += y2
#            if returncomps:
#                comps[:, i] = y2
#            if deriv:
#                grad[..., npar+(i-1)*np2:npar+i*np2] = gradc[:, ind]
#                grad[..., equal] += gradc[..., equal]
#
#        next = npar + np2*(ncomp-1)
#
#    if more is not None:
#        if len(more) < 2:
#            more.insert(1, realX)
#        y += mp_func(more, a[next:], gradc, deriv=deriv)
#        if deriv:
#            grad[..., next:] = gradc
#
#    return y
#

# func mp_setx (profile=, npar=, ncomp=, realX=, more=, equal=) {


def mp_setx(profile=None, npar=None, ncomp=None,
            realX=None, more=None, equal=None):
    """
    DOCUMENT x=mp_setx(profile=myfunc, npar=npar, ncomp=ncomp,
                      realX=realX, more=more, equal=equal)

    Set x parameter for use with mp_func

    PROFILE: function to be used as base profile, same restrictions as
             for LMFIT
    NPAR:    number of parameters needed by base profile
    NCOMP:   number of components to use (default: 1)
    realX:   real X parameter to pass to base function PROFILE
    MORE: optionally, result from a previous call to MP_SETX. Use when
             all the components of the complex profile you are
             building are of the same type (i.e. not the same PROFILE
             function)
    EQUAL: vector containing the indices of the parameters of the base
             profile which should be the same for every components.

   SEE ALSO: mp_func, mp_seta, mp_getx, multiprofile
    """
    # ncomp=ncomp?ncomp:1
    if ncomp is None:
        ncomp = 1
    return profile, realX, npar, ncomp, equal, more


# TODO: do we need this func?
# func mp_getx(x, &profile, &realX, &npar, &ncomp, &equal, &more) {
def mp_getx(x):
    """
    /* DOCUMENT mp_getx, multiX, profile, realX, npar, ncomp, equal, more

        Reverse from mp_setx: get information out of the complex lmfit "X"
        parameter used with mp_func.

        SEE ALSO: mp_func, mp_seta, mp_setx, multiprofile
    */
    """
#  profile=_car(x, 1); # _car(list,i) returns the i-th item of the list
#  realX  =_car(x, 2);
#  npar   =_car(x, 3);
#  ncomp  =_car(x, 4);
#  equal  =_car(x, 5);
#  more   =_car(x, 6);
    return x[0], x[1], x[2], x[3], x[4], x[5]
# }


# func mp_seta(params,equal=,more=,get=){


def mp_seta(params, equal=None, more=None, get=None):
    """
    DOCUMENT multiprofileparams(params,equal=,more=,get=)

    Helps setting parameter A for MP_FUNC. (Note: for simple cases,
    using this function is overkill).

    PARAMS is a 2D array where PARAMS(i,) is the set of parameters for
    the i-th component, and conversely PARAMS(,n) is the vector of
    values of the n-th parameter. EQUAL can be set to a vector of
    indices which should be fitted as equal, and MORE to a set of
    parameters for the supplementary function. When a parameter is set
    in equal, its value is taken from PARAMS(1,).

    If keyword GET is set to a vector, does the contrary, setting
    PARAMS accordingly to GET. However, PARAMS and MORE must have the
    right shape before calling MP_SETA.

    Example: you want to form parameter A of multiprofile for fitting
    Gaussian lines, having initial guesses for three parameters
    AMPLITUDE, VELOCITY, WIDTH, every component having the same width.
    A=mp_seta([AMPLITUDE,VELOCITY,WIDTH],equal=[3])

    On the other hand, if you have an initial guess for each component
    in a1, a2 and a3:
    A=mp_seta(transpose([a1, a2, a3]), equal=[3]);

    SEE ALSO: multiprofile, mp_func, lmfit, mp_setx

    """
    sz = params.shape
    ncomp = sz[0]
    npars = sz[1]
    npeq = np.asarray(equal).size

    if (get is not None):
        a = get
    else:
        a = np.empty((ncomp*npars-(ncomp-1)*npeq+np.asarray(more).size))

    # TODO return index
    # peigne=(indgen[ncomp]-1)*(npars-npeq)
    # -1 ou pas ?
    peigne = np.fromiter(range(0, ncomp), int)*(npars-npeq)

    if (ncomp > 1 and npeq > 0):
        peigne[1:] += npeq

    for p in range(npars):
        if (np.where(equal == p)[0].size == 0):
            if (get is not None):
                # params(,p)=a(p+peigne);
                params[:, p] = a[p+peigne]
            else:
                a[p+peigne] = params[:, p]
        else:
            if (get is not None):
                params[:, p] = a[p]
            else:
                a[p] = params[0, p]

            if (ncomp > 1):
                peigne[2:] -= 1

    if (more is not None):
        if (get is not None):
            # more=a(1-numberof(more):);
            more = a[1-more.size:]
        # a(1-numberof(more):)=more;
        a[1-more.size:] = more

    return a

# primitives

# func linear(x,a,&grad,deriv=) {


def poly(xdata, *params):
    """ Return a polynomial function and its Jacobian

    ydata = params[0] + ...+  params[k] * xdata**k


    """
    nterms = len(params) # nterms = degree + 1

    # x**0
    ydata = np.full_like(xdata, params[0])
    jacobian = np.zeros(ydata.shape + (nterms,))

    if nterms > 1:
        # x**1
        ydata += params[1] * xdata
        jacobian[..., 1] = xdata

        if nterms > 2:
            xpow = np.copy(xdata)

        # x**2 and above
        for k in range(2, nterms):
            xpow *= xdata
            ydata += params[k] * xpow
            jacobian[..., k] = xpow

    return ydata, jacobian

def poly_lmfit(x, a, deriv=None):
    """
    DOCUMENT poly_lmfit(x,a)
         or poly_lmfit(x,a,grad,deriv=1)

    Returns the polynomial sum(a(n)*x^(n-1)), with derivatives in GRAD
    if DERIV set to a "true" value.  Very simplistic, but might come
    in handy, as it is compatible with lmfit (and multiprofile).

    SEE ALSO: poly, linear, lmfit, multiprofile
    """

    degp1 = a.size

    if (deriv):
        # grad=array(1.,dimsof(x),degp1);
        grad = np.array[(1., x.shape, degp1)]
        # //grad(,1)=0; //useless
        # grad(..,1)=1;
        grad[:, 0] = 1
        if (degp1 >= 1):
            grad[:, 1] = x
        for n in np.arange(2, degp1):
            grad[:, n+1] = x**n

    if (degp1 == 1):
        # return np.array[a(1), dimsof(x)];
        return np.full(x.shape, a[1])

    y = a[0] + a[1]*x

    for n in np.arange(2, degp1):
        y += a[n]*x**(n-1)

    return y

# Fit doppler-shifted lines over a spectrum

# TODO ol_setx doesn t touch to  realX=None, lines=None, positivity=None, !
# func ol_setx(profile=, realX=, lines=, positivity=,
# intensities=, fixedratio=) {


def ol_setx(profile=None, realX=None, lines=None, positivity=None,
            intensities=None, fixedratio=None):
    """Set up X parameter for offsetlines().

    Parameters
    ----------

    profile :  model function for individual line (default: moffat1d)
    realX  : independent variables used by PROFILE
    lines  : vector containing the list of lines
    positivity : for each line, 1 if the line should be forced a
             positive amplitude (emission line), -1 for a negative
             amplitude (absorption line)
    intensities : relative intensities of the lines

    Notes
    -----

    The intensities of the lines can be unconstrained (default),
    constrained to be emission lines or absorption lines (POSITIVITY
    set and either INTENSITIES not set or FIXEDRATIO set to 0), or
    constrained to have predefined relative intensities (INTENSITIES
    set and FIXEDRATIO not set or set to 1). If FIXEDRATIO is set to 1
    and INTENSITIES is not set, it defaults to array(1.,
    numberof(LINES)).

    See Also:
    offsetlines
    """

    if (profile is None):
        # see https://docs.astropy.org/en/stable/modeling/reference_api.html
        # TODO
        from moffat import moffat1d
        profile = moffat1d

    if (fixedratio is None):
        # fixedratio=!is_void(intensities);
        if intensities is None:
            fixedratio = False

    if (fixedratio and intensities is None):
        intensities = np.ones(len(lines))

    # return _lst(profile, realX, lines, positivity, intensities, fixedratio)
    print(f" ol_setx {profile}")
    print(f"{ realX, lines, positivity, intensities, fixedratio}")
    return (profile, realX, lines, positivity, intensities, fixedratio)

# func offsetlines(x,a,&grad,&comps,deriv=,returncomps=){


def offsetlines(x, a, deriv=None, returncomps=None):
    """Fit several lines of identical shape over a spectrum, sharing a
     common displacement (for instance Doppler shift, if the
     wavelength range is short enough).

     This function is suitable for call by lmfit. It returns a complex
     profile made of the sum of several lines (Moffat profiles, by
     default), which are moved only together (their relative distances
     remain unchanged). As long as the primitive profile is able to
     return derivatives, offsetlines does, too.

    Parameters
    ----------
     X : np.array
        the result of a call to ol_setx, which see. X contains
        information on the lines to fit and the type of profile to use
        (Moffat by default). X also contains the wavelengths or
        frequencies.
     A : np.array
        the vector of parameters to fit. If ol_setx() has been called
        with INTENSITIES set and FIXEDRATIO either not set or set to
        1, A(1) is the multiplicative coefficient by which to multiply
        each of these individual relative intensities. In all other
        cases, the first numberof(lines) elements of A are the
        individual intensities of the various lines. The remaining
        parameters are always common to all the lines: the offset
        relative to the rest position set with the LINES keyword of
        ol_setx, and then the other parameters for the PROFILE set
        using ol_setx. By default, the PROFILE==moffat1d, and requires
        two parameters for the line shape (line width and beta; see
        moffat1d()).

     Examples
     --------

      // Basic set-up
      x = span(2.0, 2.4, 200)  // set up wavelength (or frequency) vector

      lines=[2.058, 2.15, 2.16, 2.3]; // give rest wavelength or
                                      // frequency of each line

      // Prepare spectrum
      olx=ol_setx(realX=x, lines=lines);
      A=[   1, 0.5,   0.6, 1.2,       // individual intensities
         0.02, 0.005, 1.1];           // displacement, width, beta
      y=offsetlines(olx, A);
      plg, y, x;

      // Fit with free intensities
      y_obs= y+0.2*random_n(dimsof(y));
      res=lmfit(offsetlines, olx, A, y_obs,deriv=1);
      fma; plg, y_obs, x;
      plg, offsetlines(olx, A), x, color="red";

      // Prepare spectrum, setting INTENSITIES in ol_setx
      olx=ol_setx(realX=x, lines=lines, intensities=[1., 0.5, 0.6, 1.2]);
      A=[1., 0.02, 0.005, 1.1];
      y=offsetlines(olx, A);
      fma; plg, y, x;

      // Fit with tied intensities
      y_obs= y+0.2*random_n(dimsof(y));
      res=lmfit(offsetlines, olx, A, y_obs, deriv=1);
      fma; plg, y_obs, x;
      plg, offsetlines(olx, A), x, color="red";

      See Also
      --------
      ol_setx, lmfit, multiprofile, moffat1d.
    """

    profile = x[0]
    realX = x[1]
    lines = x[2]
    positivity = x[3]
    intensities = x[4]
    fixedratio = x[5]

    print(f"call offsetline with x {x}")

    nlines = len(lines)
    print(f"nlines {nlines}")
    npars = len(a)
    print(f"npars {npars}")

    if (not fixedratio):
        npars -= nlines-1

    pars = np.empty((nlines, npars))

    if (fixedratio):
        print("fixedratio")
        pars[:, 0] = intensities
        pars[:, 1] = lines+a[1]
        # pars(,3:)=a(-,3:);
        pars[:, 2] = a[np.newaxis, 2:]
    else:
        pars[:, 0] = a[0:nlines]

        if (positivity is not None):
            ind = np.where(positivity == -1)
            if (ind.size):
                pars[ind, 0] = -abs(pars[ind, 0])

            ind = np.where(positivity == 1)
            if (ind.size):
                pars[ind, 0] = abs(pars[ind, 0])

        pars[:, 1] = lines + a[nlines+1]
        # pars[:,2]=a(-,nlines+2:);
        pars[:, 2] = a[np.newaxis, nlines+2:]

    a2 = mp_seta(pars)
    X = mp_setx(npar=npars, ncomp=nlines, realX=realX, profile=profile)
    # TODO call jac sp=mp_func(X,a2, grad2, deriv=deriv);
    sp, grad2 = mp_func(X, *a2)

    # if (deriv):
    # peigne=(indgen(nlines)-1)*npars
    peigne = np.fromiter(range(0, nlines), int)*npars
    grad = np.array(sp.shape, a.size)
    if (fixedratio):
        grad[:, 0] = sp
    else:
        grad[:, 0:nlines] = grad2[:, 0+peigne]
    if fixedratio:
        offset = nlines-1
    else:
        offset = 0
    for i in range(npars):
        # TODO sum
        # grad[:,i+offset]=grad2[:,peigne+i](:,sum)
        print("sum")

    if (fixedratio):
        sp *= a[0]

    return sp, grad


def test_offsetlines():
    # Basic set-up

    # set up wavelength (or frequency) vector
    x = np.linspace(2.0, 2.4, 200)

    # give rest wavelength or frequency of each line
    lines = np.array([2.058, 2.15, 2.16, 2.3])

    # Prepare spectrum
    print("Prepare spectrum call ol_setx")
    olx = ol_setx(realX=x, lines=lines)
    print(f"olx {olx}")

    # individual intensities  displacement, width, beta
    A = np.array([1, 0.5, 0.6, 1.2,  0.02, 0.005, 1.1])

    print("call offsetlines")
    y = offsetlines(olx, A)
    # plg, y, x;

    # Fit with free intensities
    print("Fit with free intensities")
    # y_obs= y+0.2*random_n(dimsof(y))
    y_obs = y+0.2*np.random.normal(0, 1, y.shape)
    # res=lmfit(offsetlines, olx, A, y_obs,deriv=1)
    res, req = optimize.curve_fit(offsetlines, olx, A, y_obs)
    # resopt_jac, reqcov_jac = optimize.curve_fit(curve_fit_func, nx, y, p0=a0,
    #                                            jac=curve_fit_func.jac)
    # fma; plg, y_obs, x
    # plg, offsetlines(olx, A), x, color="red";

    # Prepare spectrum, setting INTENSITIES in ol_setx
    olx = ol_setx(realX=x, lines=lines, intensities=[1., 0.5, 0.6, 1.2])
    A = [1., 0.02, 0.005, 1.1]
    y = offsetlines(olx, A)

    # fma; plg, y, x;

    # Fit with tied intensities
    y_obs = y+0.2*np.random.standard_normal(y.shape)
    # res=lmfit(offsetlines, olx, A, y_obs, deriv=1);
    resopt, reqcov = optimize.curve_fit(offsetlines, olx, A, y_obs)
    # fma; plg, y_obs, x;
    plt.figure()
    plt.plot(offsetlines(olx, A), x)
    plt.show()


if __name__ == '__main__':
    test_offsetlines()
