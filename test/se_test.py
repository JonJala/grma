"""
Testing of grma_lib.py
"""

import itertools as it
import os
import sys

import numpy as np
import pytest
import pandas as pd
pd.options.mode.copy_on_write = True #https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#
import scipy.sparse as sp

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
test_directory = os.path.abspath(os.path.join(main_directory, "test"))
data_directory = os.path.abspath(os.path.join(test_directory, "data"))

import grma_lib as sut
the_func = sut.calculate_ses



# Numpy settings
def numpy_err_handler(err: str, flag: bytes):
    """
    Function that numpy should call when an error occurs.  This is used to ensure that any errors
    are also logged, as opposed to just going to stderr and not being collected in the log

    :param err: String describing the error
    :param flag: A byte describing the error (see numpy.seterrcall() docs)
    """
    print("Received Numpy error: %s (%s)", err, flag)
np.seterr(all='call')
np.seterrcall(numpy_err_handler)
np.set_printoptions(threshold=sys.maxsize)


# Random number generator
rng = np.random.default_rng(seed=143823)



def residualize(rel_info: sut.THRESHOLDED_REL_TYPE, unresidualized: np.ndarray) -> np.ndarray:
    _, num_ppl = unresidualized.shape

    results = np.zeros_like(unresidualized, dtype=float)

    for person in range(num_ppl):
        results[:, person] = unresidualized[:, person] - np.mean(unresidualized[:, rel_info[person]], axis=1)

    return results

def generate_pheno_geno(num_snps, num_ppl, allele_freq=0.5, pheno_var=1.0):
    return rng.normal(scale=pheno_var, size=(1, num_ppl)), \
           rng.binomial(n=2, p=allele_freq, size=(num_snps, num_ppl))


def convert_rel_info_to_R(rel_info):
    num_ppl = len(rel_info)

    result = np.identity(num_ppl, dtype=float)
    for person, rel_list in enumerate(rel_info):
        result[person, rel_list] -= np.reciprocal(len(rel_list), dtype=float)

    return result


def calculate_expected_ses(rel_info, residualized_phenotypes, residualized_genotypes) -> np.ndarray:
    sqrt_e_ssr = np.linalg.norm(residualized_phenotypes)

    X = residualized_genotypes.T   # In the write-up, X seems to be 1 (or N) x M
    R = convert_rel_info_to_R(rel_info)

    inv_X_t_X = np.reciprocal(np.sum(np.square(X), axis=0, keepdims=True))
    right_term = np.sum((X.T @ R @ R.T).T * X, axis=0, keepdims=True) * inv_X_t_X

    main_term_numerator = inv_X_t_X * right_term
    main_term_denominator = np.trace(R @ R.T).reshape(1, -1) - right_term

    result = sqrt_e_ssr * np.sqrt(main_term_numerator / main_term_denominator)

    return result.ravel()


REL_INFO_1 = [
    [0, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [1, 5],
    [1, 5, 6],
    [0, 7, 8],
    [0, 7, 8],
    [1, 5, 7, 8, 9],
    [10, 11],
    [10, 11],
    [12, 13, 14, 15],
    [12, 13, 14, 15],
    [12, 13, 14, 15],
    [12, 13, 14, 15]
]

DUPLICATES_1 = np.array([False, False, False, False, False, False, False, False, False, False,
                True, True, True, True, True, True])

R_MATRIX_1 = np.array(
    [
        [0.75, 0.0, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
        [-0.2, -0.2, 0.8, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.2, -0.2, -0.2, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.2, -0.2, -0.2, -0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.3333, 0.0, 0.0, 0.0, -0.3333, 0.6667, 0.0, 0.0, 0.0],
        [-0.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6667, -0.3333, 0.0],
        [-0.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3333, 0.6667, 0.0],
        [0.0, -0.2, 0.0, 0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.8]
    ]
)





REL_INFO_2 = [
    [0, 1],
    [0, 1],
    [2, 3, 4],
    [2, 3],
    [4, 5],
    [4, 5]
]

DUPLICATES_2 = np.array([True, True, False, False, False, False])

R_MATRIX_2 = np.array(
    [
        [0.6667, -0.3333, -0.3333, 0.0],
        [-0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, -0.5],
        [0.0, 0.0, -0.5, 0.5]
    ]
)

TRACE_2 = 1.0 + np.trace(R_MATRIX_2)

GENO_2 = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    ]
)

PHENO_2 = np.array(
    [
        [0.9, 0.7, 0.5, 0.5, 0.6, 0.3]
    ]
)

SES_2 = np.array(
    [
        [0.3983, 0.2230, 0.2934]
    ]
)



class TestCalculateSes:

    @pytest.mark.parametrize("num_ppl", [1, 10, 1000])
    @pytest.mark.parametrize("num_snps", [1, 10, 1000])
    def test__all_singletons__all_nan_or_inf(self, num_ppl, num_snps):
        rel_info = [[i] for i in range(num_ppl)]
        r_matrix = convert_rel_info_to_R(rel_info)
        duplicates = np.full(num_ppl, False)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)
        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl

        actual_results = the_func(R_matrix=sp.csr_matrix(r_matrix), duplicates=duplicates,
                                   trace_rr=np.trace(r_matrix),
                                   residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes.ravel())

        assert np.all(np.isnan(actual_results) | np.isinf(actual_results))


    @pytest.mark.parametrize("num_ppl", [4, 20, 100])  # Needs to be even
    @pytest.mark.parametrize("num_snps", [1, 3, 10, 100])
    def test__all_contiguous_doubles__expected_results(self, num_ppl, num_snps):
        rel_info = [[2*(pnum>>1), 2*(pnum>>1)+1] for pnum in range(num_ppl)]
        r_matrix = np.zeros((0,0))
        duplicates = np.full(num_ppl, True)
        trace_rr = num_ppl / 2.0

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)

        phenotypes = np.zeros((1, num_ppl), dtype=float)
        phenotypes[:, 1::2] = 1.0
        genotypes = np.zeros((num_snps, num_ppl), dtype=float)
        genotypes[:, 1::2] = 1.0

        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)
        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl


        expected_results = np.reciprocal(np.sqrt((num_ppl - 2.0) / 2.0))
        actual_results = the_func(R_matrix=sp.csr_matrix(r_matrix), duplicates=duplicates,
                                   trace_rr=trace_rr,
                                   residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes.ravel())

        test_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)

        assert np.allclose(actual_results, expected_results, atol=0.00001, equal_nan=True)



    @pytest.mark.parametrize("rel_info, genotypes, phenotypes, expected_results",
                             [(REL_INFO_2, GENO_2, PHENO_2, SES_2)])
    def test__precanned_inputs__expected_results(self, rel_info, genotypes, phenotypes, expected_results):
        num_ppl = len(rel_info)
        sp_r_matrix, duplicates, trace_rr = sut.calculate_R_matrix(rel_info)

        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl

        actual_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                  trace_rr=trace_rr, residualized_genotypes=residualized_genotypes,
                                  residualized_phenotypes=residualized_phenotypes.ravel())

        assert np.allclose(actual_results, expected_results, atol=0.0001, equal_nan=True)



    @pytest.mark.parametrize("rel_info", [REL_INFO_1, REL_INFO_2])
    @pytest.mark.parametrize("num_snps", [10]) #10, 100, 1000
    def test__precanned_rel_info__expected_results(self, rel_info, num_snps):
        num_ppl = len(rel_info)
        sp_r_matrix, duplicates, trace_rr = sut.calculate_R_matrix(rel_info)


        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl

        expected_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)
        actual_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                  trace_rr=trace_rr, residualized_genotypes=residualized_genotypes,
                                  residualized_phenotypes=residualized_phenotypes.ravel())

        assert np.allclose(actual_results, expected_results, atol=0.00001, equal_nan=True)




    @pytest.mark.parametrize("rel_info, duplicates, r_matrix", [(REL_INFO_1, DUPLICATES_1, R_MATRIX_1),
                                                                (REL_INFO_2, DUPLICATES_2, R_MATRIX_2)])
    @pytest.mark.parametrize("seed", [4, 7832])
    def test__reshuffle_snps__results_shuffled_similarly(self, rel_info, duplicates, r_matrix, seed):
        num_snps = 500
        num_ppl = len(rel_info)
        sp_r_matrix, duplicates, trace_rr = sut.calculate_R_matrix(rel_info)

        rgen = np.random.default_rng(seed=seed)
        permutation = rgen.permutation(num_snps)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)
        snpshuffled_genotypes = residualized_genotypes[permutation, :]

        unshuf_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                  trace_rr=trace_rr, residualized_genotypes=residualized_genotypes,
                                  residualized_phenotypes=residualized_phenotypes.ravel())

        shuf_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                trace_rr=trace_rr, residualized_genotypes=snpshuffled_genotypes,
                                residualized_phenotypes=residualized_phenotypes.ravel())

        assert np.allclose(unshuf_results[permutation], shuf_results, atol=0.00001, equal_nan=True)




    @pytest.mark.parametrize("rel_info", [REL_INFO_1, REL_INFO_2])
    @pytest.mark.parametrize("seed", [35, 6537])
    def test__reshuffle_ppl__results_unchanged(self, rel_info, seed):
        num_snps = 10
        num_ppl = len(rel_info)
        sp_r_matrix, duplicates, trace_rr = sut.calculate_R_matrix(rel_info)

        rgen = np.random.default_rng(seed=seed)
        permutation = rgen.permutation(num_ppl)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)


        pplshuffled_rel_info = [[] for _ in range(num_ppl)]
        for i, rel_list in enumerate(rel_info):
            pplshuffled_rel_info[permutation[i]] = sorted([permutation[person] for person in rel_list])
        permutation_indices = [permutation.tolist().index(i) for i in range(num_ppl)]
        pplshuffled_genotypes = residualized_genotypes[:, permutation_indices]
        pplshuffled_phenotypes = residualized_phenotypes[:, permutation_indices]

        shuf_sp_r_matrix, shuf_duplicates, shuf_trace_rr = sut.calculate_R_matrix(pplshuffled_rel_info)
        shuf_var_y = np.sum(np.square(pplshuffled_phenotypes)) / num_ppl


        unshuf_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                  trace_rr=trace_rr, residualized_genotypes=residualized_genotypes,
                                  residualized_phenotypes=residualized_phenotypes.ravel())

        shuf_results = the_func(R_matrix=shuf_sp_r_matrix, duplicates=shuf_duplicates,
                                trace_rr=shuf_trace_rr, residualized_genotypes=pplshuffled_genotypes,
                                residualized_phenotypes=residualized_phenotypes.ravel())


        assert np.isclose(trace_rr, shuf_trace_rr, atol=0.00001, equal_nan=True)
        assert np.allclose(unshuf_results, shuf_results, atol=0.00001, equal_nan=True)




    # This test makes use of the GRMA calculate_R_matrix function.  That function should be tested
    # separately (here it is assumed correct / is used for both main code and test inputs)
    @pytest.mark.parametrize("num_ppl", [10, 100, 500])
    @pytest.mark.parametrize("seed", [1, 135432, 95423, 8465233194])
    @pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.8])
    def test__random_rel_info__expected_results(self, seed, num_ppl, p):
        num_snps = 500
        rgen = np.random.default_rng(seed=seed)

        rel_info = []
        for person in range(num_ppl):
            g = rgen.choice([True, False], size=num_ppl, p=[p, 1.0-p])
            g[person] = True
            rel_info.extend([np.nonzero(g)[0].tolist()])

        sp_r_matrix, duplicates, trace_rr = sut.calculate_R_matrix(rel_info)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        var_y = np.sum(np.square(residualized_phenotypes)) / num_ppl
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        expected_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)
        actual_results = the_func(R_matrix=sp_r_matrix, duplicates=duplicates,
                                  trace_rr=trace_rr, residualized_genotypes=residualized_genotypes,
                                  residualized_phenotypes=residualized_phenotypes.ravel())

        assert np.allclose(actual_results, expected_results, atol=0.00001, equal_nan=True)
