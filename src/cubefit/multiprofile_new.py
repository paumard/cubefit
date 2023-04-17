#!/usr/bin/env python3

import numpy as np


class MultiProfile:
    """Manage multiple profile

    Parameters
    ----------
    profile : callable
        A complex curvefit.lineprofiles profile.
    """
    def __init__(self, profiles, nterms, ncomp, equals):
        self.profiles = profiles
        self.nterms = nterms
        self.ncomp = ncomp
        self.equals = equals

    def __call__(self):
        """self() -> profile
        """
        profile = np.zeros()
        profile_jac = self.jac()
        return profile, profile_jac

    def jac(self):
        """self.jac() -> profile_jac
        """
        profile_jac = np.zeros()
        return profile_jac
