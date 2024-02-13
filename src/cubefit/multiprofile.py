#!/usr/bin/env python3

import numpy as np
# import math
# from astropy.modeling.functional_models import Moffat1D


import matplotlib.pyplot as plt
from scipy import optimize

"""Compound profiles for cubefit.cubemodel.CubeModel
"""

class MultiProfile:
    '''Compound profile for cubefit.cubemodel.CubeModel

    A MultiProfile instance is a Python callable suitable as a profile
    for CubeModel, defined as the sum of profiles.

    Parameters
    ----------
    profiles : callable or tuple of callables
        One or several CubeModel profiles.
    nterms : int or tuple of int
        Number of parameters for each profile
    ncomps : int or tuple of int, optional
        Number of components of each profile
    tiespecs : None, tuple of ((tuple of int) or (dict of dict)), optional

        If not None: each item is either a tuple of int or a dict of dict.

        If an item is a tuple of int: the indices of parameters that
        should be equal to the value of this parameter of the first
        component of each profile.

        If an item is a dict of dict: the keys of the outer dict are
        the indices of parameters that need to be tied to the value of
        this parameter in the first instance of this component. The
        values are dicts where the keys are one or several of "ratios"
        and "offsets" and the values tuples of ratios and values
        relative to the first instance, with one less item than
        ncomps[this profile].

    Examples
    --------

    >>> import numpy
    >>> from cubefit.profiles import gauss, poly
    >>> from cubefit.multiprofile import MultiProfile
    >>> x = numpy.linspace(-10, 10, 3000)

    Create a profile as the sum of two Gaussian profiles:

    >>> profile = MultiProfile(gauss, 3, 2)
    >>> a = (1., 0., 1., 2., 2., 0.5)
    >>> y, jac = profile(x, *a)

    Same on top of a linear continum:

    >>> profile = MultiProfile((gauss, poly), (3, 2), (2, 1))
    >>> a = (1., 0., 1., 2., 2., 0.5, 1, 0.1)
    >>> y, jac = profile(x, *a)

    Same, forcing the width of the two Gaussian to be equal:

    >>> profile=MultiProfile((gauss,poly),(3,2),(2,1),tiespecs=((2,),()))
    >>> a = (1., 0., 0.5, 2., 2., 1, 0.1)
    >>> y, jac = profile(x, *a)

    Same, with three Gaussians, forcing specific offset between the
    two lines, specific line ratios and equal widths:

    >>> profile=MultiProfile((gauss,poly),(3,2),(3,1), \
           tiespecs=({0: {"ratios": (0.5, 2)}, \
                        1: {"offsets": (-5, 5.)}, \
                        2: {} },()))
    >>> a = (1., 0., 0.5, 1, 0.1)
    >>> y, jac = profile(x, *a)

    '''
    def __init__(self, profiles, nterms, ncomps=1, tiespecs=None):
        # Ensure all parameters are in arrays
        self.profiles = np.atleast_1d(profiles)
        self.nterms   = np.broadcast_to(nterms, self.profiles.shape)
        self.ncomps   = np.broadcast_to(ncomps, self.profiles.shape)
        self.nterms_tot = np.sum(self.nterms*self.ncomps)
        # Initialize self.tiespecs: to empy lists is tiespecs is None,
        # else to a sorted and uniq-ified tuple
        if tiespecs is None:
            self.tiespecs  = tuple( () for k in range(self.profiles.size))
        else:
            assert len(tiespecs) == self.profiles.size
            for k in range(self.profiles.size):
                tprec=-1
                for tind in tiespecs[k]:
                    assert tprec < tind, \
                        "tiespecs must be uniq and sorted"
                    assert tind < self.nterms[k], \
                        "tiespecs must be < nterms"
                    tprec=tind
                self.nterms_tot -= (self.ncomps[k]-1) * len(tiespecs[k])
            self.tiespecs = tiespecs

    def __call__(self, xdata, *params):
        '''self(xdata, *params) -> ydata, jacobian 
        Suitable as profile for CubeModel.
        '''
        assert len(params) == self.nterms_tot, "Wrong number of parameters" 

        params=np.asarray(params)

        ydata=np.zeros_like(xdata)
        jacobian=np.zeros(ydata.shape + np.asarray(params).shape)

        curp=0 # index of current parameter
        for k in range(len(self.profiles)):
            # loop over profiles
            fpfp=curp # index of First Parameter of first instance For
                      # this Profile
            profile=self.profiles[k]
            nparms=self.nterms[k]
            nc=self.ncomps[k]
            tspecs=self.tiespecs[k]
            tpl = params[curp:curp+nparms]
            y, j = profile(xdata, *tpl)
            ydata += y
            jacobian[..., curp:curp+nparms] = j
            curp += nparms
            for c in range(1,nc):
                # loop over components for this profile
                prms = list(params[curp:curp+nparms-len(tspecs)])
                for i in tspecs:
                    # loop over tiespecs for this profile, if any
                    prms.insert(i, tpl[i])
                    if isinstance(tspecs, dict):
                        # if tiespecs[k] is a dict, appliy ratios
                        # and offsets
                        if "ratios" in tspecs[i]:
                            prms[i] *= tspecs[i]["ratios"][c-1]
                        if "offsets" in tspecs[i]:
                            prms[i] += tspecs[i]["offsets"][c-1]
                y, j = profile(xdata, *prms)
                ydata += y
                ij=0
                for k in range(nparms):
                    if k in tspecs:
                        if isinstance(tspecs, dict):
                            # if tiespecs[k] is a dict, apply ratios
                            # and offsets
                            if "ratios" in tspecs[k]:
                                j[...,ij] *= tspecs[k]["ratios"][c-1]
                            # if "offsets" in tspecs[k]:
                            #    pass
                        jacobian[..., fpfp+ij] += j[..., ij]
                    else:
                        jacobian[..., curp] = j[..., ij]
                        curp += 1
                    ij += 1

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
