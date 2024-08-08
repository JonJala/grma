"""
Testing of grma_lib.py
"""

import os
import sys

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import numpy as np
import pytest
import pandas as pd
pd.options.mode.copy_on_write = True #https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#
import itertools as it

import grma_lib as sut


# rng = np.random.default_rng(seed=0)
test_directory = os.path.abspath(os.path.join(main_directory, "test"))
data_directory = os.path.abspath(os.path.join(test_directory, "data"))
# testcase_name = "toy_example_1"
# testcase_dir = os.path.join(data_directory, testcase_name)
# fam_file = os.path.join(testcase_dir, f"{testcase_name}.fam")

# # TODO(jonbjala) Many more tests will need to be written

# @pytest.fixture
# def base_king_output():
#     king_output = {
#         "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
#         "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
#         "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
#     }
#     df = pd.DataFrame(king_output)
#     return df


# @pytest.fixture
# def king_output_disconnected_rels():
#     KING_DF_LENGTH = 6
    
#     king_output = {
#         "ID1": [i for i in range(1, 2 * KING_DF_LENGTH, 2)],
#         "ID2": [i for i in range(2, 2 * KING_DF_LENGTH + 1, 2)],
#         "FID1": [1] * KING_DF_LENGTH,
#         "FID2": [1] * KING_DF_LENGTH,
#         "Kinship": [0.50] * KING_DF_LENGTH,
#         "InfType": ["Dup/MZTwin", "FS", "PO", "2nd", "3rd", "4th"],
#     }
#     df = pd.DataFrame(king_output)
#     return df

# def generate_king_output_same_degree(inftype: str):
#     NUM_RELS = 7
#     NUM_ROWS = NUM_RELS * NUM_RELS

#     king_output = {
#         "ID1": [i for i in range(1, NUM_RELS + 1) for _ in range(NUM_RELS)],
#         "ID2": [i for _ in range(NUM_RELS) for i in range(1, NUM_RELS + 1)],
#         "FID1": [1] * NUM_ROWS,
#         "FID2": [1] * NUM_ROWS,
#         "Kinship": [0.50] * NUM_ROWS,
#         "InfType": [inftype] * NUM_ROWS,
#     }
#     df = pd.DataFrame(king_output)
#     df = df.loc[df["ID1"] != df["ID2"]]
#     return df    

# @pytest.fixture()
# def king_output_same_degree(request):
#     # Allowed InfTypes are ["Dup/MZTwin", "FS", "PO" "1", "2", "3", "4", "UN"]
#     return generate_king_output_same_degree(inftype=request.param)

# def fam_file_format(num_rows: int):
#     # Sets rng to simulate num_rows number of phenotypes
#     rng = np.random.default_rng(seed=0)
#     fam_file = {
#         "FID": [1] * num_rows,
#         "IID": [i for i in range(1, num_rows + 1)],
#         "IIDF": [0] * num_rows,
#         "IIDM": [0] * num_rows,
#         "SEX": [1] * num_rows,
#         "PHENO": list(rng.integers(1, 6, size=num_rows).astype(float)),
#     }
#     fam_df = pd.DataFrame(fam_file)
#     return fam_df

# @pytest.fixture
# def short_fam_file():
#     return fam_file_format(num_rows=8)

# @pytest.fixture
# def long_fam_file():
#     return fam_file_format(num_rows=20)

# # List of InfTypes that are available in the InfType column of a King output file.
# ALLOWED_INFTYPES = [key for key in sut.INF_TO_DEG_MAP.keys()]

# # Getting MAX_ID of king_output_same_degree to use in tests
# DF2 = generate_king_output_same_degree(inftype="Dup/MZTwin") # InfType doesn't matter here, as we only care about max_ID for range in iid parametrization. 
# DF2_MAX_ID = max(DF2['ID1'].max(), DF2['ID2'].max())

rng = np.random.default_rng(seed=143823)

def residualize(rel_info: sut.THRESHOLDED_REL_TYPE, unresidualized: np.ndarray) -> np.ndarray:
    _, num_ppl = unresidualized.shape

    results = np.zeros_like(unresidualized)

    for person in range(num_ppl):
        results[:, person] = results[:, person] - np.mean(unresidualized[:, rel_info[person]], axis=1)

    return results

def generate_pheno_geno(num_snps, num_ppl, allele_freq=0.5, pheno_var=1.0):
    return rng.normal(scale=pheno_var, size=(1, num_ppl), dtype=float), \
           rng.binomial(n=2, p=allele_freq, size=(num_snps, num_ppl), dtype=float)


def convert_rel_info_to_R(rel_info):
    num_ppl = len(rel_info)

    result = np.identity(num_ppl, dtype=float)
    for person in rel_info:
        result[person, rel_info[person]] - np.reciprocal(len(rel_info[person]))

    return result


def calculate_expected_ses(rel_info, residualized_phenotypes, residualized_genotypes) -> np.ndarray:
    sqrt_e_ssr = np.linalg.norm(residualized_phenotypes)

    X = residualized_genotypes
    R = convert_rel_info_to_R(rel_info)
    
    inv_X_t_X = np.linalg.inv(X.T @ X)
    right_term = (X.T @ R @ R.T @ X) @ inv_X_t_X

    main_term_numerator = inv_X_t_X @ right_term
    main_term_denominator = np.trace(R @ R.T) - right_term

    result = sqrt_e_ssr * np.sqrt(main_term_numerator / main_term_denominator)

    return result


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
    [1, 5, 7, 8, 9]
]


class TestCalculateSes:

    test_func = sut.calculate_ses

    @pytest.mark.parametrize("num_ppl", [1, 10, 1000])
    @pytest.mark.parametrize("num_snps", [1, 10, 1000])
    def test__all_singletons__all_nan_or_inf(self, num_ppl, num_snps):
        rel_info = [[i] for i in range(num_ppl)]

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        actual_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)
        assert np.all((actual_results == np.nan) | (actual_results == np.inf))


    @pytest.mark.parametrize("num_ppl", [2, 20, 2000])  # Needs to be even
    @pytest.mark.parametrize("num_snps", [20, 2000])
    @pytest.mark.parametrize("allele_freq", [0.25, 0.5, 0.75])
    def test__all_contiguous_doubles__expected_results(self, num_ppl, num_snps, allele_freq):
        rel_info = [[2*i, 2*i + 1] for i in range(num_ppl >> 1)]

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl,
                                                    allele_freq=allele_freq)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        expected_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)
        actual_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)

        assert np.allclose(actual_results, expected_results)


    @pytest.mark.parametrize("num_ppl", [10, 100, 10000])
    @pytest.mark.parametrize("seed", [1, 135432, 95423, 8465233194])
    def test__random_rel_info__expected_results(self, seed, num_ppl):
        num_snps = 5000
        rgen = np.random.default_rng(seed=seed)

        rel_info = []
        for person in num_ppl:
            g = rgen.choice([True, False], size=num_ppl)
            g[person] = True
            rel_info.extend(g.tolist())

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        expected_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)
        actual_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)

        assert np.allclose(actual_results, expected_results)    


    @pytest.mark.parametrize("rel_info", [REL_INFO_1])
    @pytest.mark.parametrize("num_snps", [10, 100, 10000])
    def test__precanned_rel_info__expected_results(self, rel_info, num_snps):
        num_ppl = len(rel_info)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)

        expected_results = calculate_expected_ses(rel_info=rel_info,
                                                  residualized_phenotypes=residualized_phenotypes,
                                                  residualized_genotypes=residualized_genotypes)
        actual_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)

        assert np.allclose(actual_results, expected_results) 


    @pytest.mark.parametrize("rel_info", [REL_INFO_1])
    @pytest.mark.parametrize("seed", [4, 7832, 5243, 68034211])
    def test__reshuffle_snps__results_shuffled_similarly(self, rel_info, seed):
        num_snps = 5000
        num_ppl = len(rel_info)

        rgen = np.random.default_rng(seed=seed)
        permutation = rgen.permutation(num_snps)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl,
                                                    allele_freq=allele_freq)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)
        snpshuffled_genotypes = residualized_genotypes[permutation, :]

        unshuf_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)
        shuf_results = test_func(rel_info=rel_info, residualized_genotypes=snpshuffled_genotypes,
                                 residualized_phenotypes=residualized_phenotypes)

        assert np.allclose(unshuf_results[permutation], shuf_results)


    @pytest.mark.parametrize("rel_info", [REL_INFO_1])
    @pytest.mark.parametrize("seed", [4, 7832, 5243, 68034211])
    def test__reshuffle_snps__results_shuffled_similarly(self, seed):
        rel_info = REL_INFO_1
        num_snps = 5000
        num_ppl = len(rel_info)

        rgen = np.random.default_rng(seed=seed)
        permutation = rgen.permutation(num_snps)

        phenotypes, genotypes = generate_pheno_geno(num_snps=num_snps, num_ppl=num_ppl,
                                                    allele_freq=allele_freq)
        residualized_phenotypes = residualize(rel_info=rel_info, unresidualized=phenotypes)
        residualized_genotypes = residualize(rel_info=rel_info, unresidualized=genotypes)
        snpshuffled_genotypes = residualized_genotypes[permutation, :]

        unshuf_results = test_func(rel_info=rel_info, residualized_genotypes=residualized_genotypes,
                                   residualized_phenotypes=residualized_phenotypes)
        shuf_results = test_func(rel_info=rel_info, residualized_genotypes=snpshuffled_genotypes,
                                 residualized_phenotypes=residualized_phenotypes)

        assert np.allclose(unshuf_results[permutation], shuf_results)