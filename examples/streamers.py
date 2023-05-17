#!/usr/bin/env python3
#    Copyright (C) 2023  Thibaut Paumard <thibaut.paumard@obspm.fr>
#            Julien Brulé
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

"""Create and fit mock data with a single streamer

The object is made of two thin filaments with velocity gradient. One
is brighter than the other, with SNR above one only for the brighter
filament (here SNR= peak/stddev). Line width is anti-correlated with
line flux.

This script create then fits fake data.

The conclusion is that, in the absence of regularization, the fit is
OK for SNR>1 and does not work for SNR<=1, whereas it is good with
regularization until SNR~0.5 on this example with very crude initial
guess.
"""

import cProfile
import pstats
import io
from pstats import SortKey

import numpy as np
from matplotlib import pyplot as plt
from cubefit.dopplerlines import DopplerLines
from cubefit.cubemodel import CubeModel, markov,\
                              l1l2, RegularizationWithNumericalGradient
from cubefit.lineprofiles import gauss, ngauss

l1l2_num = RegularizationWithNumericalGradient(l1l2)

DEBUG = False
PROF = True

if PROF:
    pr = cProfile.Profile()


def print_profiling_results(string_context, pr):
    print(f"profiling result for {string_context}")
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


# regularizations=[l1l2_num]
regularizations = [None, markov, l1l2, l1l2_num]


def add_noise(cube, sigma=0.02):
    '''Add noise to data.

    This method always uses the same pseudo-random sequence as it is
    intended for use in a reproduceable test suite. Don't use it for
    Monte-Carlo simulations or such.

    Parameters
    ----------
    cube : ndarray
    the cube on which the noise will be added
    sigma : float64
    the value / levelocity of noise
    '''
    # instanciate a random number generator with fixed seed
    # warning: changing the seed may affect the success of certain tests below
    rng = np.random.default_rng(3)
    psigma = sigma
    tmp_cube = np.copy(cube)
    tmp_cube = cube + rng.standard_normal(cube.shape) * psigma
    return tmp_cube


# Shape of data cube (nx, ny, nz)
nx, ny, nz = 16, 16, 21
# Cube with x(=alpha) and y(=delta) coordinate of each data point
alpha = np.linspace(-ny/2, ny/2, ny)[np.newaxis, :]*np.ones((nx, ny))
delta = np.linspace(-nx/2, nx/2, nx)[:, np.newaxis]*np.ones((nx, ny))
# Build "true" parameter maps
# intensity map
# velocity map
velocity = 2*(alpha+delta)
# I=gauss(alpha+delta, 1, 0, 100)[0] \
intensity = (velocity-np.min(velocity))/np.max(velocity)*2 \
    * (gauss(alpha-delta, 1, 4, 2)[0]
       + gauss(alpha-delta, 0.5, -4, 2)[0])
# width map
dvelocity = (2-intensity/np.max(intensity))*5
# display maps if DEBUG is True


# Model we want to test
w0 = 2.166120e-6
dw = 5e-11
Dw = dw*(nz-1)
profile = DopplerLines(w0, profile=ngauss)
waxis = np.linspace(w0-Dw/2, w0+Dw/2, nz)
model_none = CubeModel(profile=profile, profile_xdata=waxis,
                       regularization=None)
vaxis = (waxis-w0)/w0*profile.light_speed

# Parameters for "true" cube. Can be 1D or 3D.
xreal = np.transpose(np.asarray([intensity, velocity, dvelocity]), (1, 2, 0))

if PROF:
    pr.enable()

reality = model_none.model(xreal)

if PROF:
    pr.disable()
    print_profiling_results("model computing real values", pr)

# Sigma of errors to add to "true" cube to get "observational" data
sigma = 0.2*np.max(reality)
snr = np.max(reality, axis=2)/sigma
data = add_noise(reality, sigma)
weights = np.full(data.shape, 1/sigma)

# Display "truth"
imshow_kwds = [{"vmin": np.min(intensity),
               "vmax": np.max(intensity)},
               {"vmin": np.min(velocity),
               "vmax": np.max(velocity)},
               {"vmin": np.min(dvelocity),
               "vmax": np.max(dvelocity)}]
model_none.view_data["imshow_kwds"] = imshow_kwds


def view_more(fig, axes):
    fig.suptitle("Truth")
    axes[0].set_title("Flux")
    axes[1].set_title("Velocity")
    axes[2].set_title("Width")
    axes[0].contour(snr, [0.5, 1])
    axes[1].contour(snr, [0.5, 1])
    axes[2].contour(snr, [0.5, 1])


model_none.view_more = view_more
model_none.view(xreal, noscale=True)
model_none.view_data["fig"] = None  # detach figure

# Initial guess for fit. Can be 1D or 3D.
xtest_1d = [np.max(intensity), 0., np.std(velocity)]
xtest = np.full((nx, ny, len(xtest_1d)), xtest_1d)


def view_more(fig, axes):
    fig.suptitle("Initial guess")
    axes[0].set_title("Flux")
    axes[1].set_title("Velocity")
    axes[2].set_title("Width")
    axes[0].contour(snr, [0.5, 1])
    axes[1].contour(snr, [0.5, 1])
    axes[2].contour(snr, [0.5, 1])


model_none.view_more = view_more
model_none.view(xtest, noscale=True)
model_none.view_data["fig"] = None  # detach figure


if PROF:
    pr.enable()

    profile(waxis, *xtest_1d)[0]

    pr.disable()
    print_profiling_results("profile(waxis,...) computing real values", pr)


if DEBUG:
    fig = plt.figure()
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[6, 10, :],
             label="bright spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[6, 10, :],
             label="bright spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[10, 6, :],
             label="faint spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[10, 6, :],
             label="faint spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, profile(waxis, *xtest_1d)[0],
             label="first guess")
    plt.plot((waxis-w0)/w0*profile.light_speed,
             np.sum(np.sum(reality, axis=1), axis=0)/nx,
             label=f"integral spectrum/{nx}")
    plt.legend()
    fig.show()
    plt.pause(1)

if None in regularizations:
    # Do fit
    model_none.data = data
    model_none.weight = weights

    def view_more(fig, axes):
        fig.suptitle("No-regularization fit")
        axes[0].set_title("Flux")
        axes[1].set_title("Velocity")
        axes[2].set_title("Width")
        axes[0].contour(snr, [0.5, 1])
        axes[1].contour(snr, [0.5, 1])
        axes[2].contour(snr, [0.5, 1])

    model_none.view_more = view_more
    model_none.view_data["imshow_kwds"] = imshow_kwds

    res_x_none, fx_none, gx_none, status_none = model_none.fit(xtest)

    if PROF:
        pr.enable()

        res_x_none, fx_none, gx_none, status_none = model_none.fit(xtest)

        pr.disable()
        print_profiling_results("fit(xtest) computing ", pr)

    # Compute model cube
    model_none_cube = model_none.model(res_x_none)

    chi2 = np.sum(((data - model_none_cube)/sigma)**2)\
        / (data.size - res_x_none.size)
    print(f"reduced chi2  =  =  {chi2}")

    def plot3(i, j):
        fig = plt.figure()
        plt.plot(vaxis, reality[i, j, :], label=f"true spectrum at {j} {i}")
        plt.plot(vaxis, data[i, j, :], label=f"data spectrum at {j} {i}")
        plt.plot(vaxis, model_none_cube[i, j, :],
                 label=f"fitted spectrum at {j} {i}")
        plt.legend()
        fig.show()

    if DEBUG:
        # Display fit for central pixel
        plot3(6, 10)
        # Display fit for lower left pixel
        plot3(10, 6)

if markov in regularizations:
    # Do fit with regularization
    model_markov = CubeModel(profile=profile, profile_xdata=waxis,
                             regularization=markov)
    model_markov.data = data
    model_markov.weight = weights
    weight_markov = [2, 0.1, 0.1]
    mu_markov = [1., 1., 2.]
    model_markov.delta = np.sqrt(np.asarray(mu_markov)
                                 * np.asarray(weight_markov))
    model_markov.scale = np.asarray(mu_markov)/model_markov.delta

    def view_more(fig, axes):
        fig.suptitle("markov fit")
        axes[0].set_title("Flux")
        axes[1].set_title("Velocity")
        axes[2].set_title("Width")
        axes[0].contour(snr, [0.5, 1])
        axes[1].contour(snr, [0.5, 1])
        axes[2].contour(snr, [0.5, 1])

    model_markov.view_more = view_more
    model_markov.view_data["imshow_kwds"] = imshow_kwds

    if PROF:
        pr.enable()

        res_x_markov, fx_markov, gx_markov, \
        status_markov = model_markov.fit(xtest, ftol=1e-10, xtol=1e-8)

        pr.disable()
        print_profiling_results("fit(xtest) computing ", pr)


    res_x_markov, fx_markov, gx_markov, \
        status_markov = model_markov.fit(xtest, ftol=1e-10, xtol=1e-8)

    # Compute model cube
    model_markov_cube = model_markov.model(res_x_markov)

    chi2 = np.sum(((data-model_markov_cube)/sigma)**2)\
        / (data.size-res_x_markov.size)
    print(f"reduced chi2 == {chi2}")

if l1l2 in regularizations:
    # Do fit with regularization
    model_l1l2 = CubeModel(profile=profile, profile_xdata=waxis,
                           regularization=l1l2)
    model_l1l2.data = data
    model_l1l2.weight = weights
    weight_l1l2 = [2, 0.1, 0.1]
    mu_l1l2 = [1., 1., 2.]
    model_l1l2.delta = np.sqrt(np.asarray(mu_l1l2)*np.asarray(weight_l1l2))
    model_l1l2.scale = np.asarray(mu_l1l2)/model_l1l2.delta

    def view_more(fig, axes):
        fig.suptitle("l1l2 fit")
        axes[0].set_title("Flux")
        axes[1].set_title("Velocity")
        axes[2].set_title("Width")
        axes[0].contour(snr, [0.5, 1])
        axes[1].contour(snr, [0.5, 1])
        axes[2].contour(snr, [0.5, 1])

    model_l1l2.view_more = view_more
    model_l1l2.view_data["imshow_kwds"] = imshow_kwds

    res_x_l1l2, fx_l1l2, gx_l1l2, status_l1l2 = model_l1l2.fit(xtest)

    # Compute model cube
    model_l1l2_cube = model_l1l2.model(res_x_l1l2)

    chi2 = np.sum(((data-model_l1l2_cube)/sigma)**2)\
        / (data.size-res_x_l1l2.size)
    print(f"reduced chi2 == {chi2}")

if l1l2_num in regularizations:
    # Do fit with regularization
    model_l1l2_num = CubeModel(profile=profile, profile_xdata=waxis,
                               regularization=l1l2_num)
    model_l1l2_num.data = data
    model_l1l2_num.weight = weights
    weight_l1l2_num = [2, 0.1, 0.1]
    mu_l1l2_num = [1., 1., 2.]
    model_l1l2_num.delta = np.sqrt(np.asarray(mu_l1l2_num)
                                   * np.asarray(weight_l1l2_num))
    model_l1l2_num.scale = np.asarray(mu_l1l2_num)/model_l1l2_num.delta

    def view_more(fig, axes):
        fig.suptitle("l1l2_num fit")
        axes[0].set_title("Flux")
        axes[1].set_title("Velocity")
        axes[2].set_title("Width")
        axes[0].contour(snr, [0.5, 1])
        axes[1].contour(snr, [0.5, 1])
        axes[2].contour(snr, [0.5, 1])

    model_l1l2_num.view_more = view_more
    model_l1l2_num.view_data["imshow_kwds"] = imshow_kwds

    if PROF:
        pr.enable()

        res_x_l1l2_num, fx_l1l2_num, gx_l1l2_num,\
        status_l1l2_num = model_l1l2_num.fit(xtest)

        pr.disable()
        print_profiling_results("fit(xtest) computing ", pr)


    res_x_l1l2_num, fx_l1l2_num, gx_l1l2_num,\
        status_l1l2_num = model_l1l2_num.fit(xtest)

    # Compute model cube
    model_l1l2_num_cube = model_l1l2_num.model(res_x_l1l2_num)

    chi2 = np.sum(((data-model_l1l2_num_cube)/sigma)**2)\
        / (data.size-res_x_l1l2_num.size)
    print(f"reduced chi2  =  =  {chi2}")

# Ensure windows don't close automatically
plt.show()
