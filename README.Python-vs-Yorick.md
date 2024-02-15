# Differences between CubeFit for Python and CubeFit for Yorick

CubeFit was originally implemented in the Yorick language. The
following paper where using the Yorick implementation:

- Paumard, Thibaut; Ciurlo, Anna; Morris, Mark R.; Do, Tuan & Ghez,
  Andrea M. 2022; Regularized 3D spectroscopy with CubeFit: Method and
  application to the Galactic Center circumnuclear disk; Astronomy &
  Astrophysics, Volume 664, id.A97; doi: 10.1051/0004-6361/202243228
- Ciurlo, Anna ; Paumard, Thibaut ; Rouan, Daniel & Clénet, Yann 2019;
  Clumpiness of the interstellar medium in the central parsec of the
  Galaxy from H2 flux-extinction correlation; Astronomy &
  Astrophysics, Volume 621, id.A65; doi: 10.1051/0004-6361/201731763
- Ciurlo, Anna ; Paumard, Thibaut ; Rouan, Daniel & Clénet, Yann 2016;
  Hot molecular hydrogen in the central parsec of the Galaxy through
  near-infrared 3D fitting; Astronomy & Astrophysics, Volume 594,
  id.A113; doi: 10.1051/0004-6361/201527173

The Python re-implementation has the following notable differences
compared to the original Yorick implementation:

- Cube dimensions in Yorick are X, Y, W; in Python Y, X, W;
- Yorick uses 1-indexing while Python defaults to 0-indexing;
- Regularisation term in Yorick is 4 times larger than in Python
  (multiply delta by 2 and keep delta*scale the same);
- The Python profile functions return a tuple with the objective
  function and its Jacobian matrix: ```profile(x, *a) -> y,
  jacobian``` while the Yorick profile functions return the objective
  function as their return value and the JAcobian as an output
  parameter: ```profile(x, a, &jacobian, deriv=) -> y```;
- The ```multiprofile``` and ```offsetlines``` features have been
  merged (and Pythonified) in the
  ```cubefit.multiprofile.MultiProfile``` class.

