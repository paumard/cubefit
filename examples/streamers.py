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
from cubefit.cubemodel import CubeModel, markov, l1l2, RegularizationWithNumericalGradient
from cubefit.lineprofiles import gauss, ngauss

l1l2_num=RegularizationWithNumericalGradient(l1l2)

DEBUG=True

#regularizations=[None, l1l2]
regularizations=[None, markov, l1l2, l1l2_num]

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

def plot_res(res, figsize=(7, 7), fig=None, axes=None):
    # How many parameters
    nterms=res.shape[2]
    # Create subplots if not provided
    if axes is None:
        # How many rows (ny) and columns (nx) in plot?
        nx=int(np.sqrt(nterms))
        if nx*nx == nterms:
            ny=nx
        else:
            ny=nx+1
            nx=ny
        fig=plt.figure(figsize=figsize)
        axes = [fig.add_subplot(ny, nx, p) for p in range(1, nterms+1)]
    # Plot one parameters map in each subplot
    for k in range(nterms):
        axes[k].imshow(res[:,:,k])
    axes[0].figure.show()

    return fig, axes

# Shape of data cube (nx, ny, nz)
nx, ny, nz = 16, 16, 21
# Cube with x(=alpha) and y(=delta) coordinate of each data point
alpha=np.linspace(-ny/2, ny/2, ny)[np.newaxis, :]*np.ones((nx, ny))
delta=np.linspace(-nx/2, nx/2, nx)[:, np.newaxis]*np.ones((nx, ny))
# Build "true" parameter maps
# internsity map
# velocity map
v=2*(alpha+delta)
#I=gauss(alpha+delta, 1, 0, 100)[0] \
I=(v-np.min(v))/np.max(v)*2 \
    * (  gauss(alpha-delta, 1, 4, 2)[0]
       + gauss(alpha-delta, 0.5, -4, 2)[0])
# width map
dv=(2-I/np.max(I))*5
# display maps if DEBUG is True


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

freal, areal=plot_res(xreal)
freal.suptitle("Truth")
areal[0].set_title("Flux")
areal[1].set_title("Velocity")
areal[2].set_title("Width")
freal.show()

# Sigma of errors to add to "true" cube to get "observational" data
sigma = 0.2*np.max(reality)
snr=np.max(reality, axis=2)/sigma
data = add_noise(reality, sigma)
weights = np.full(data.shape, 1/sigma)

# Initial guess for fit. Can be 1D or 3D.
xtest_1d=[np.max(I), 0., np.std(v)]
xtest = np.full((nx, ny, len(xtest_1d)), xtest_1d)

if DEBUG:
    fig=plt.figure()
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[6,10, :], label="bright spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[6,10,:], label="bright spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, reality[10,6,:], label="faint spectrum (true)")
    plt.plot((waxis-w0)/w0*profile.light_speed, data[10,6,:], label="faint spectrum (data)")
    plt.plot((waxis-w0)/w0*profile.light_speed, profile(waxis, *xtest_1d)[0], label="first guess")
    plt.plot((waxis-w0)/w0*profile.light_speed, np.sum(np.sum(reality, axis=1), axis=0)/nx,
             label=f"integral spectrum/{nx}")
    plt.legend()
    fig.show()
    plt.pause(1)

if None in regularizations:
    # Do fit
    model_none.data = data
    model_none.weight = weights

    res_x_none, fx_none, gx_none, status_none = model_none.fit(xtest)


    # Compute model cube
    model_none_cube=model_none.model(res_x_none)

    chi2=np.sum(((data-model_none_cube)/sigma)**2)/(data.size-res_x_none.size)
    print(f"reduced chi2 == {chi2}")

    def plot3(i, j):
        fig=plt.figure()
        plt.plot(vaxis, reality[i, j, :], label=f"true spectrum at {j} {i}")
        plt.plot(vaxis, data[i, j, :], label=f"data spectrum at {j} {i}")
        plt.plot(vaxis, model_none_cube[i, j, :], label=f"fitted spectrum at {j} {i}")
        plt.legend()
        fig.show()

    if DEBUG:
        # Display fit for central pixel
        plot3(6, 10)
        # Display fit for lower left pixel
        plot3(10, 6)

    fnone, anone=plot_res(res_x_none)
    fnone.suptitle("No-regularization fit")
    anone[0].set_title("Flux")
    anone[1].set_title("Velocity")
    anone[2].set_title("Width")
    anone[0].contour(snr, [0.5, 1])
    anone[1].contour(snr, [0.5, 1])
    anone[2].contour(snr, [0.5, 1])
    fnone.show()
    plt.pause(1)

if markov in regularizations:
    # Do fit with regularization
    model_markov = CubeModel(profile=profile, profile_xdata=waxis, regularization=markov)
    model_markov.data = data
    model_markov.weight = weights
    weight_markov=[2, 0.1, 0.1]
    mu_markov    =[1., 1., 2.]
    model_markov.delta=np.sqrt(np.asarray(mu_markov)*np.asarray(weight_markov))
    model_markov.scale=np.asarray(mu_markov)/model_markov.delta
    res_x_markov, fx_markov, gx_markov, status_markov = model_markov.fit(xtest, ftol=1e-10, xtol=1e-8)


    # Compute model cube
    model_markov_cube=model_markov.model(res_x_markov)

    chi2=np.sum(((data-model_markov_cube)/sigma)**2)/(data.size-res_x_markov.size)
    print(f"reduced chi2 == {chi2}")

    fmarkov, amarkov=plot_res(res_x_markov)
    fmarkov.suptitle("markov fit")
    amarkov[0].set_title("Flux")
    amarkov[1].set_title("Velocity")
    amarkov[2].set_title("Width")
    amarkov[0].contour(snr, [0.5, 1])
    amarkov[1].contour(snr, [0.5, 1])
    amarkov[2].contour(snr, [0.5, 1])
    fmarkov.show()
    plt.pause(1)

if l1l2 in regularizations:
    # Do fit with regularization
    model_l1l2 = CubeModel(profile=profile, profile_xdata=waxis, regularization=l1l2)
    model_l1l2.data = data
    model_l1l2.weight = weights
    weight_l1l2=[2, 0.1, 0.1]
    mu_l1l2    =[1., 1., 2.]
    model_l1l2.delta=np.sqrt(np.asarray(mu_l1l2)*np.asarray(weight_l1l2))
    model_l1l2.scale=np.asarray(mu_l1l2)/model_l1l2.delta
    res_x_l1l2, fx_l1l2, gx_l1l2, status_l1l2 = model_l1l2.fit(xtest)


    # Compute model cube
    model_l1l2_cube=model_l1l2.model(res_x_l1l2)

    chi2=np.sum(((data-model_l1l2_cube)/sigma)**2)/(data.size-res_x_l1l2.size)
    print(f"reduced chi2 == {chi2}")

    fl1l2, al1l2=plot_res(res_x_l1l2)
    fl1l2.suptitle("l1l2 fit")
    al1l2[0].set_title("Flux")
    al1l2[1].set_title("Velocity")
    al1l2[2].set_title("Width")
    al1l2[0].contour(snr, [0.5, 1])
    al1l2[1].contour(snr, [0.5, 1])
    al1l2[2].contour(snr, [0.5, 1])
    fl1l2.show()
    plt.pause(1)

if l1l2_num in regularizations:
    # Do fit with regularization
    model_l1l2_num = CubeModel(profile=profile, profile_xdata=waxis, regularization=l1l2_num)
    model_l1l2_num.data = data
    model_l1l2_num.weight = weights
    weight_l1l2_num=[2, 0.1, 0.1]
    mu_l1l2_num    =[1., 1., 2.]
    model_l1l2_num.delta=np.sqrt(np.asarray(mu_l1l2_num)*np.asarray(weight_l1l2_num))
    model_l1l2_num.scale=np.asarray(mu_l1l2_num)/model_l1l2_num.delta
    res_x_l1l2_num, fx_l1l2_num, gx_l1l2_num, status_l1l2_num = model_l1l2_num.fit(xtest)


    # Compute model cube
    model_l1l2_num_cube=model_l1l2_num.model(res_x_l1l2_num)

    chi2=np.sum(((data-model_l1l2_num_cube)/sigma)**2)/(data.size-res_x_l1l2_num.size)
    print(f"reduced chi2 == {chi2}")

    fl1l2_num, al1l2_num=plot_res(res_x_l1l2_num)
    fl1l2_num.suptitle("l1l2 (numerical gradient) fit")
    al1l2_num[0].set_title("Flux")
    al1l2_num[1].set_title("Velocity")
    al1l2_num[2].set_title("Width")
    al1l2_num[0].contour(snr, [0.5, 1])
    al1l2_num[1].contour(snr, [0.5, 1])
    al1l2_num[2].contour(snr, [0.5, 1])
    fl1l2_num.show()
    plt.pause(1)


# Ensure windows don't close automatically
plt.show()
