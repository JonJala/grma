"""
Testing of bedbimfam.py (mainly read_bed_file())
"""

import itertools as it
import os
import sys

import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sp
from scipy import stats

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_directory = os.path.abspath(os.path.join(main_directory, "test"))
data_directory = os.path.abspath(os.path.join(test_directory, "data"))
sys.path.append(main_directory)
pd.options.mode.copy_on_write = True #https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#

import bedbimfam as sut

import helper
import tcs


##############

np.seterr(all='print')

##############




class TestReadBedFile:


    @pytest.mark.parametrize("G",
        [
            np.array([[1.0]]),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
            np.array([[1.0, 2.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 0.0, 2.0, 1.0]]),
            np.array([[1.0, np.nan], [0.0, 1.0]])
        ]
    )
    def test__precanned_inputs__expected_results(self, G, tmp_path):
        bed_filename = os.path.join(tmp_path, "temp.bed")

        M, N = G.shape

        sut.write_bed_file(bed_filename=bed_filename, G=G)

        G_0 = sut.read_bed_file(bed_filename=bed_filename, M=M, N=N)

        assert G_0.shape == (M,N)
        assert np.allclose(G, G_0, equal_nan=True)


    @pytest.mark.parametrize("rng_seed", [390, 586, 9522])
    def test__random_inputs__expected_results(self, rng_seed, tmp_path):
        rng = np.random.default_rng(seed=rng_seed)
        bed_filename = os.path.join(tmp_path, "temp.bed")

        M = rng.integers(100, 1000)
        N = rng.integers(100, 1000)

        G = rng.choice([0.0, 1.0, 2.0, np.nan], size=(M, N), replace=True)

        sut.write_bed_file(bed_filename=bed_filename, G=G)

        G_0 = sut.read_bed_file(bed_filename=bed_filename, M=M, N=N)

        assert G_0.shape == (M,N)
        assert np.allclose(G, G_0, equal_nan=True)


    @pytest.mark.parametrize("rng_seed", [23, 54, 71])
    def test__varying_M_start__expected_results(self, rng_seed, tmp_path):
        rng = np.random.default_rng(seed=rng_seed)
        bed_filename = os.path.join(tmp_path, "temp.bed")

        M = rng.integers(100, 1000)
        N = rng.integers(100, 1000)
        M_start = rng.integers(10, M-10)

        G = rng.choice([0.0, 1.0, 2.0, np.nan], size=(M, N), replace=True)

        sut.write_bed_file(bed_filename=bed_filename, G=G)

        G_0 = sut.read_bed_file(bed_filename=bed_filename, M=M, N=N, M_start=M_start)

        assert G_0.shape == (M-M_start,N)
        assert np.allclose(G[M_start:M], G_0, equal_nan=True)
