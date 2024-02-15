# Unit tests for CubeFit

This directory contains the CubeFit test suite.

## Running the test suite

One way to run it:

* make sure the cubefit module can be found, for instance by making an
  editable install:
    python3 -m pip install -e ./

* run he unittest main:
    python3 -m unittest

To run (and debug!) a single test, one option is to follow this
example:

  ipython3
  from tests.test_cubemodel import TestCubemodel
  t=TestCubemodel()
  t.test_cubemodel_gradient()

If one test fails, an error will be raised. Enter the debugger with:
  %debug

Then move up the stack to investigate:
  up
  print(<variable_name>)
  ...

## Writing tests

Make one file test_xxx.py for each submodule cubefit.xxx.

Try to write at least one test for each function or method.

The tests should be quiet by default, not output anything on the
terminal, plot anything or produce files. Any output (graphics or text
on the console) should be protected by :
   if DEBUG:
Debugging output can be enabled by setting (to anything) an
environment variable, one per test file, for instance
TEST_CUBEMODEL_DEBUG for test_cubemodel.py.


Each test should call the various self.assertXXX() provided by
unittest to actually check the results.
