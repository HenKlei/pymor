# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.parameters.base import Parameters, Mu
from pymortests.base import runmodule

import pytest


num_samples = 100


@pytest.fixture(scope='module')
def space():
    return Parameters({'diffusionl': 1}).space(0.1, 1)


def test_uniform(space):
    values = space.sample_uniformly(num_samples)
    assert len(values) == num_samples
    for value in values:
        assert space.contains(value)


def test_randomly(space):
    values = space.sample_randomly(num_samples)
    assert len(values) == num_samples
    for value in values:
        assert space.contains(value)


def test_randomly_without_count(space):
    mu = space.sample_randomly()
    assert isinstance(mu, Mu)


def test_parse_parameter():
    parameters = Parameters(b=2, a=1)
    mu_as_list = [1, 2, 3]
    mu_as_parameter_and_back = list(parameters.parse(mu_as_list).to_numpy())
    assert mu_as_list == mu_as_parameter_and_back


if __name__ == "__main__":
    runmodule(filename=__file__)
