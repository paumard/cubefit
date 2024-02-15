#!/usr/bin/env python3

import numpy as np
# import math
# from astropy.modeling.functional_models import Moffat1D


import matplotlib.pyplot as plt
from scipy import optimize

"""Compound profile for cubefit.cubemodel.CubeModel
"""

class MultiProfile:
    '''Compound profile for cubefit.cubemodel.CubeModel

    A MultiProfile instance is a Python callable suitable as a profile
    for CubeModel, defined as the sum of profiles.

    The compound profile is composed of one or several instances of
    one of several profiles (i.e. Python callables usable as "profile"
    parameter for the cubefit.cubemodel.CubeModel constructor). The
    parameters of the various instances of the same profile can be
    tied: set equal, forced to have a certain ratio, forced to have a
    certain offset (see tiespecs parameter). This can be used for
    instance for fitting several spectral lines with known offsets in
    velocity or wavelength, sharing the same width and with known
    intensity ratio. Parameters that are tied are present only once,
    for the first instance of a group, since the value of this
    parameter for the other instances is fully determined by its value
    for the first instance.

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
