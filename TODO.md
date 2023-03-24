# CubeFit to-do list and developer information

## Coding standards

### Quality assurance

1. Write tests in parallel to developing the code:
   - Put manual testing routines that can serve as examples in
     examples/;
   - Write at least one unit test for each function or method:
     - Verify the return value of functions for a few known cases,
       especially corner cases, using assertEqual() or
       assertAlmostEqual().;
     - Validate derivatives, Jacobian matrices, gradients using
       numerical methods;
     - Verify that each parameter does the right thing;
     - Test all supported use cases (e.g. scalar and array input if
       both are legitimate);
     - Verify that error conditions actually raise an error using
       assertRaises().
1. Document objects (classes, methods, functions...) in parallel to
   writing tests since this is the moment when supported use cases
   should be defined;
1. Run the full unit test suite very often, at least once before
   committing code;
1. Commit often, keeping logical blocks together and avoiding
   unrelated changes; if relevant and before pushing, rebase
   successive commits (especially small ones) that should have been
   one (e.g. typo correction, or one commit for code and the other for
   documentation);
1. Push (and pull!) often; before pushing, run git pull --rebase,
   resolve conflicts and run the test suite again.

### Docstrings

As much as it makes sense, follow numpydoc style:
https://numpydoc.readthedocs.io/en/latest/format.html

### Markdown

ASCII documents in the source are in gitlab-flavored markdown:
https://docs.gitlab.com/ee/user/markdown.html

For instance the checklists below follow this syntax:
1. [x] Completed task
1. [~] Inapplicable task
1. [ ] Incomplete task

## To-do list

### P0: Towards a full CubeFit

1. [ ] Activate regularization:
   1. [ ] Write unit tests for l1l2 and markov:
      - [ ] Create a uniform image (2D array), feed it l1l2 and
            markov: result should be 0;
      - [ ] Add a constant: result should not change.;
      - [ ] Add a single spike in the center, result should increase
            and become large with peak value;
      - [ ] Create two images of same shape with a standard deviation
            of one, the first image made of random values (e.g. drawn
            with random.normal(0, 1, shape) and the other with an
            organized pattern (for instance left half=-1, right
            half=1). Check that l1l2 and markov return a (much) larger
            value for random noise than for the organized pattern.;
      - [~] Check gradient (compare gradient provided by l1l2 and
            markov with a numerical estimate);
      - [ ] Verify that setting scale and delta does change the
            result, coherently with expectations, including gradient
            estimate;
   1. [ ] Implement and check regularization in CubeModel.eval():
      - [ ] Reactivate lines of code in eval() that add the
            regularization term; run the test suite to validate that
            the behavior of eval() with self.regularisation set to
            None is unchanged;
      - [ ] Add lines in the existing test suite to verify that
            setting regularisation to l1l2 or markov does the right
            thing:
        - [ ] Return value of eval() is larger when regularisation
              is not None;
        - [ ] Gradient is still correctly estimated (compare with
              numerical estimate).
   1. [ ] Check that fitting work!
      - [ ] Expand examples/streamers.py to perform a second fit with
            regularisation activated (try both l1l2 and markov. Try to
            find a good set of hyper-parameters (scale and delta) so
            that the velocity map is recovered for the full field
            without blurring the flux and width maps.
      - [ ] Include tests of CubeModel.fit() with regularisation=l1l2
            and with regularisation=markov in the unit test suite; if
            possible include asserts to check that the result is as
            expected.
1. [ ] Implement and check advanced, required features:
   1. [ ] Allow user to fix some parameters; perhaps setting
          lower==upper is enough?
   1. [ ] Pythonify multiprofile:
      - [ ] Use proper capitalisation;
      - [ ] Define Pythonic API;
      - [ ] Verify and write test suite and example;
      - [ ] Write documentation.
1. [ ] Check on real data.

### P1: Packaging

1. [ ] Fill packaging files:
   1. [ ] pyproject.toml;
   1. [ ] setup.py;
   1. [ ] setup.cfg.
1. [ ] Sort out copyrighting:
   1. [ ] Add a LICENSE file;
   1. [ ] Add copyright and license statement to each file.
1. [ ] Lint the code:
   1. [ ] Run pylint and correct all issues found.;
   1. [ ] Run pydocstyle --convention=numpy  and fix all issues;
   1. [ ] Activate those checks in gitlab (as warnings).
1. [ ] Continuous integration:
   1. [ ] Activate auto-building and auto-checking in gitlab.

### P2: Nice-to-have feature (=optional)

1. [ ] Allow each parameter to be tweaked depending on the other
       parameters in ptweak (e.g. neighboring pixels, or flux
       depending on width etc.); The big part is to transform the
       gradient into a Jacobian matrix.
1. [ ] Pythonify moffat.py and merge it into lineprofiles.py:
   - [ ] Rename 1D profile to simply moffat instead of moffat1d
   - [ ] Debug and write test suite including gradient;
   - [ ] Write an example or test where moffat is used as a base
         profile for DopplerLines.
1. [ ] Verify that all 1D models (from lineprofiles, dopplerlines or
       multiprofile) work with xdata of any shape (scalar,
       n-dimensional array...).
1. [ ] Verify that all 1D models (from lineprofiles, dopplerlines or
       multiprofile) work with params of any shape (where each
       parameter could be scalar or an n-dimensional array); maybe
       difficult to code the Jacobian.
