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


import numpy as np
from matplotlib import pyplot as plt
from cubefit.dopplerlines import DopplerLines
from cubefit.cubemodel import CubeModel, markov, l1l2
from cubefit.lineprofiles import gauss, ngauss

DEBUG=True

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


# Shape of data cube (nx, ny, nz)
nx, ny, nz = 16, 16, 21
# Cube with x(=alpha) and y(=delta) coordinate of each data point
alpha=np.linspace(-ny/2, ny/2, ny)[np.newaxis, :]*np.ones((nx, ny))
delta=np.linspace(-nx/2, nx/2, nx)[:, np.newaxis]*np.ones((nx, ny))
# Build "true" parameter maps
# internsity map
I=gauss(alpha+delta, 1, 0, 10)[0] \
    * (  gauss(alpha-delta, 1, 4, 2)[0]
       + gauss(alpha-delta, 0.25, -4, 2)[0])
# velocity map
v=2*(alpha+delta)
# width map
dv=(2-I)*5
# display maps if DEBUG is True
if DEBUG:
    plt.imshow(I)
    plt.title("'True' flux")
    plt.show()
    plt.imshow(v)
    plt.title("'True' velocity")
    plt.show()
    plt.imshow(dv)
    plt.title("'True' line width")
    plt.show()

# Model we want to test
w0=2.166120e-6
dw=5e-11
Dw=dw*(nz-1)
profile = DopplerLines(w0, profile=ngauss)
waxis = np.linspace(w0-Dw/2, w0+Dw/2, nz)
model_none = CubeModel(profile=profile, profile_xdata=waxis, regularization=None)
vaxis = (waxis-w0)/w0*profile.light_speed

# Parameters for "true" cube. Can be 1D or 3D.
xreal=np.transpose(np.asarray([I, v, dv]), (1, 2, 0))
reality=model_none.model(xreal)

# Sigma of errors to add to "true" cube to get "observational" data
sigma = 0.2*np.max(reality)
data = add_noise(reality, sigma)

model_none.data = data

# Initial guess for fit. Can be 1D or 3D.
xtest_1d=[np.max(I), 0., np.std(v)]
xtest = np.full((nx, ny, len(xtest_1d)), xtest_1d)

if DEBUG:
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[6,10, :], label="bright spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[6,10,:], label="bright spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[10,6,:], label="faint spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[10,6,:], label="faint spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, profile(waxis, *xtest_1d)[0], label="first guess")
    plt.plot((waxis-w0)/w0*profile.light_speed, np.sum(np.sum(reality, axis=1), axis=0)/nx,
             label=f"integral spectrum/{nx}")
    plt.legend()
    plt.show()

# Do fit
res_x_none, fx_none, gx_none, status_none = model_none.fit(xtest)


# Compute model cube
model_none_cube=model_none.model(res_x_none)

chi2=np.sum(((data-model_none_cube)/sigma)**2)/(data.size-res_x_none.size)
print(f"reduced chi2 == {chi2}")

def plot3(i, j):
    plt.plot(vaxis, reality[i, j, :], label=f"true spectrum at {j} {i}")
    plt.plot(vaxis, data[i, j, :], label=f"data spectrum at {j} {i}")
    plt.plot(vaxis, model_none_cube[i, j, :], label=f"fitted spectrum at {j} {i}")
    plt.legend()
    plt.show()

if DEBUG:
    # Display fit for central pixel
    plot3(6, 10)
    # Display fit for lower left pixel
    plot3(10, 6)

# Display intensity map
plt.imshow(res_x_none[:,:,0])
plt.contour(np.max(reality, axis=2)/sigma, [1])
plt.title("fitted flux")
plt.show()

# Display velocity map
plt.imshow(res_x_none[:,:,1])
plt.contour(np.max(reality, axis=2)/sigma, [1])
plt.title("fitted velocity")
plt.show()

# Display width map
plt.imshow(res_x_none[:,:,2])
plt.contour(np.max(reality, axis=2)/sigma, [1])
plt.title("fitted width")
plt.show()

# Do fit with regularization
model_markov = CubeModel(profile=profile, profile_xdata=waxis, regularization=markov)
model_markov.data = data
model_markov.delta=np.asarray([1e-4, 1e-4, 1e-4])
model_markov.scale=np.asarray([1e-23, 1e-21, 1e-21])
res_x_markov, fx_markov, gx_markov, status_markov = model_markov.fit(xtest, ftol=1e-10, xtol=1e-8)


# Compute model cube
model_markov_cube=model_markov.model(res_x_markov)

chi2=np.sum(((data-model_markov_cube)/sigma)**2)/(data.size-res_x_markov.size)
print(f"reduced chi2 == {chi2}")

# Display intensity map
plt.imshow(res_x_markov[:,:,0], vmin=-0.2, vmax=1.2)
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted flux")
plt.show()

# Display velocity map
plt.imshow(res_x_markov[:,:,1], vmin=-16, vmax=16)
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted velocity")
plt.show()

# Display width map
plt.imshow(res_x_markov[:,:,2])
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted width")
plt.show()

# Do fit with regularization
model_l1l2 = CubeModel(profile=profile, profile_xdata=waxis, regularization=l1l2)
model_l1l2.data = data
model_l1l2.delta=np.asarray([1e-4, 1e-4, 1e-4])
model_l1l2.scale=np.asarray([1e-23, 1e-21, 1e-21])
res_x_l1l2, fx_l1l2, gx_l1l2, status_l1l2 = model_l1l2.fit(xtest, ftol=1e-10, xtol=1e-8)


# Compute model cube
model_l1l2_cube=model_l1l2.model(res_x_l1l2)

chi2=np.sum(((data-model_l1l2_cube)/sigma)**2)/(data.size-res_x_l1l2.size)
print(f"reduced chi2 == {chi2}")

# Display intensity map
plt.imshow(res_x_l1l2[:,:,0], vmin=-0.2, vmax=1.2)
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted flux")
plt.show()

# Display velocity map
plt.imshow(res_x_l1l2[:,:,1], vmin=-16, vmax=16)
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted velocity")
plt.show()

# Display width map
plt.imshow(res_x_l1l2[:,:,2])
plt.contour(np.max(reality, axis=2)/sigma, [0.5])
plt.title("fitted width")
plt.show()
