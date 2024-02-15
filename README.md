# CubeFit for Python (and Yorick)

CubeFit is an original method for extracting maps of continuous
physical parameters (such as the velocity field and velocity
dispersion) from integral-field spectroscopy data, using
regularization to largely preserve spatial resolution in regions of
low signal-to-noise ratio.

Please cite:
  https://doi.org/10.1051/0004-6361/202243228
(see CITATION.md).

CubeFit was originally written for the Yorick language. This original
implementation can be found in the yorick subdirectory. The rest of
this package constitutes Python reimplementation in which all further
development will occur. Notable differences between the two
implementations are listed in README.Python-vs-Yorick.md.

## Prerequisites

CubeFit uses the following third-party Python modules:
numpy, matplotlib, optm

optm can be found at:
https://github.com/emmt/VMLMB/blob/main/python/optm.py

## Installation instructions

pip3 install ./


