# CubeFit

CubeFit is an original method for extracting maps of continuous
physical parameters (such as the velocity field and velocity
dispersion) from integral-field spectroscopy data, using
regularization to largely preserve spatial resolution in regions of
low signal-to-noise ratio.

See: https://doi.org/10.1051/0004-6361/202243228

## Prerequisites

CubeFit uses the following third-party Python modules:
numpy, matplotlib, optm

optm can be found at:
https://github.com/emmt/VMLMB/blob/main/python/optm.py

## Installation instructions

pip3 install ./

## Difference from the original implementation

Notable differences from the original Yorick implementation are listed
in README.Python-vs-Yorick.md.