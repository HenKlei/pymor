# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymortests.base import runmodule
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(analytical_problem):
    assert_picklable(analytical_problem)


def test_pickle_without_dumps_function(picklable_analytical_problem):
    assert_picklable_without_dumps_function(picklable_analytical_problem)


if __name__ == "__main__":
    runmodule(filename=__file__)
