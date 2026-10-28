"""
Testing of grma_lib.py
"""

import copy
import itertools as it
import logging
import os
import re
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

import bedbimfam as bbf
import grma
import grma_lib as sut
import snipar_hdf5

import helper
import tcs  # The case registry and harness; see its docstring for the format


##############

np.seterr(all='print')

##############


def make_bim(bim_filename: str, M: int):
    rs_width = len(str(M))

    bbf.write_bim_file(bim_filename=bim_filename, chrs=np.ones(M),
                       rsid=np.array([f'RS{snp_id:0{rs_width}d}' for snp_id in range(M)]),
                       bp=np.array([10*snp_id for snp_id in range(M)]),
                       a1=['G'] * M,
                       a2=['A'] * M)

def make_bim_df(M: int):
    rs_width = len(str(M+1))

    return pd.DataFrame(
        {
            bbf.BIM_CHR_COL : 1,
            bbf.BIM_RSID_COL : [f'RS{snp_id+1:0{rs_width}d}' for snp_id in range(M)],
            bbf.BIM_CM_COL : [float(i) for i in range(M)],
            bbf.BIM_BP_COL : [10 * i for i in range(M)],
            bbf.BIM_A1_COL : 'A',
            bbf.BIM_A2_COL : 'T'
        }
    )



def make_fam(fam_filename: str, N: int, pheno: np.ndarray=None):
    iids=np.array([i for i in range(N)])
    fids=iids.copy()
    bbf.write_fam_file(fam_filename=fam_filename, fid=fids, iid=iids, pheno=pheno.ravel())



def make_fam_df(N: int):
    return pd.DataFrame(
        {
            bbf.FAM_FID_COL : [2*i for i in range(N)],
            bbf.FAM_IID_COL : [i for i in range(N)],
            bbf.FAM_IIDF_COL : 0,
            bbf.FAM_IIDM_COL : 0,
            bbf.FAM_SEX_COL : 0,
            bbf.FAM_PHENO_COL : [float((i+1) * (-1) ** i) for i in range(N)]
        }
    )


make_bed=bbf.write_bed_file


# TODO(jonbjala) Include more error cases

class TestProcessPhenotypes:

    def test__simple_fam_file__return_centered(self):
        N = 100

        fam_df = make_fam_df(N)

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df)

        # Without covariates, phenotypes are still regressed on an intercept, i.e. centered
        # (notes §13.5)
        expected_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy(dtype=float)
        expected_pheno = expected_pheno - expected_pheno.mean()

        assert actual_N == actual_N_orig
        assert actual_N == N
        assert np.array_equal(actual_indices, np.arange(N))
        assert fam_df[sut.FAM_KEY].equals(actual_id_df)
        assert np.allclose(actual_pheno, expected_pheno)


    @pytest.mark.parametrize("rng_seed", [90, 456])
    def test__randomized_phenos__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        fam_df = make_fam_df(N)
        phenotypes = 10.0 * rng.random(N) + 1.0
        fam_df[bbf.FAM_PHENO_COL] = phenotypes

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df)

        assert actual_N_orig == N
        assert actual_N == N
        assert np.array_equal(actual_indices, np.arange(N))
        assert fam_df[sut.FAM_KEY].equals(actual_id_df)
        assert np.allclose(actual_pheno, phenotypes - phenotypes.mean())


    @pytest.mark.parametrize("rng_seed", [7, 2024])
    def test__no_covariates__same_as_regressing_on_intercept_alone(self, rng_seed):
        """Centering is the covariate regression with no covariate columns.  A covariate that is
        zero for everyone contributes nothing, so only the intercept is fitted."""

        rng = np.random.default_rng(seed=rng_seed)
        N = 50 + rng.integers(1, 50)

        fam_df = make_fam_df(N)
        fam_df[bbf.FAM_PHENO_COL] = 10.0 * rng.random(N) + 5.0
        zero_covar_df = fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]].copy()
        zero_covar_df["COVAR"] = 0.0

        *_, without_covars = sut.process_phenotypes(fam_file=fam_df)
        *_, with_zero_covar = sut.process_phenotypes(fam_file=fam_df, covar_file=zero_covar_df)

        assert np.allclose(without_covars, with_zero_covar)
        assert np.isclose(without_covars.mean(), 0.0)


    @pytest.mark.parametrize("rng_seed", [37, 268])
    def test__fam_file_with_nans__nan_rows_dropped(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        num_nan = rng.integers(1, N // 10 + 1)
        nan_indices = rng.choice(N, size=num_nan, replace=False)

        fam_df = make_fam_df(N)
        fam_df.loc[nan_indices, bbf.FAM_PHENO_COL] = np.nan

        expected_dropped = fam_df.iloc[nan_indices]

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df)

        assert actual_N_orig == N
        assert actual_N == N - num_nan
        assert not expected_dropped[bbf.FAM_FID_COL].isin(actual_id_df[bbf.FAM_FID_COL]).any()
        assert not expected_dropped[bbf.FAM_IID_COL].isin(actual_id_df[bbf.FAM_IID_COL]).any()
        assert not np.any(np.isnan(actual_pheno))


    @pytest.mark.parametrize("rng_seed", [6, 8923])
    def test__sample_id_file__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        num_samples = rng.integers(N // 10 + 1, 9 * (N // 10))
        sample_indices = rng.choice(N, size=num_samples, replace=False)

        fam_df = make_fam_df(N)
        sample_df = fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]].iloc[sample_indices].copy()


        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df, sample_id_file=sample_df)

        assert actual_N_orig == N
        assert actual_N == num_samples
        assert sample_df[bbf.FAM_FID_COL].isin(actual_id_df[bbf.FAM_FID_COL]).all()
        assert sample_df[bbf.FAM_IID_COL].isin(actual_id_df[bbf.FAM_IID_COL]).all()
        assert set(zip(actual_id_df[bbf.FAM_FID_COL], actual_id_df[bbf.FAM_IID_COL])) == \
               set(zip(sample_df[bbf.FAM_FID_COL], sample_df[bbf.FAM_IID_COL]))
        assert len(actual_id_df) == num_samples


    @pytest.mark.parametrize("rng_seed", [16, 18923])
    def test__pheno_file__expected_pheno_values(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        num_phenos = rng.integers(N // 10 + 1, 9 * (N // 10))
        pheno_indices = rng.choice(N, size=num_phenos, replace=False)

        fam_df = make_fam_df(N)
        phenotypes = 10.0 * rng.random(num_phenos) + 1.0
        pheno_df = fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL, bbf.FAM_PHENO_COL]].iloc[pheno_indices].copy()

        pheno_df[bbf.FAM_PHENO_COL] = phenotypes

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df, pheno_file=pheno_df)

        assert actual_N_orig == N
        assert actual_N == num_phenos
        # Centered on the mean of the samples kept, not of the whole fam file (notes §13.5)
        expected_pheno = pheno_df[bbf.FAM_PHENO_COL].sort_index().to_numpy()
        assert np.allclose(actual_pheno, expected_pheno - expected_pheno.mean())
        assert pheno_df[bbf.FAM_FID_COL].isin(actual_id_df[bbf.FAM_FID_COL]).all()
        assert pheno_df[bbf.FAM_IID_COL].isin(actual_id_df[bbf.FAM_IID_COL]).all()
        assert len(actual_id_df) == num_phenos


    @pytest.mark.parametrize("rng_seed", [215, 887])
    def test__covar_file__expected_residualization(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100
        num_covariates = rng.integers(4,10)
        covar_names = [f"COVAR_{covar_num+1}" for covar_num in range(num_covariates)]

        fam_df = make_fam_df(N)
        orig_phenotypes = fam_df[bbf.FAM_PHENO_COL].to_numpy().copy()
        covariates = np.hstack([(10.0 * rng.random(size=(N, num_covariates)) - 5.0), np.ones((N,1))])
        covar_df = pd.DataFrame({bbf.FAM_FID_COL : fam_df[bbf.FAM_FID_COL],
                                 bbf.FAM_IID_COL : fam_df[bbf.FAM_IID_COL]} |
                                {covar_names[i] : covariates[:, i].ravel()
                                    for i in range(num_covariates)
                                })

        x, _, _, _ = np.linalg.lstsq(a = covariates, b = orig_phenotypes, rcond = None)
        expected_pheno = orig_phenotypes - covariates @ x

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df, covar_file=covar_df)

        assert np.allclose(actual_pheno, expected_pheno, equal_nan=True)


    def test__covar_file_with_nans__nan_rows_dropped(self):
        pass


    def test__invalid_covar_file__expected_error(self):
        N = 100

        invalid_covar_df = pd.DataFrame(data={bbf.FAM_FID_COL : list(range(N)),
                                              bbf.FAM_IID_COL : list(range(N))})

        fam_df = make_fam_df(N)

        with pytest.raises(ValueError) as ex_info:
            actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
                sut.process_phenotypes(fam_file=fam_df, covar_file=invalid_covar_df)

    def test__all_inputs__expected_results(self):
        pass

    @pytest.mark.parametrize("rng_seed", [316, 765765])
    def test__permute_fam_file_rows__permuted_results(self, rng_seed):
        pass

    @pytest.mark.parametrize("rng_seed", [25, 908])
    def test__permute_pheno_file_rows__unchanged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        fam_df = make_fam_df(N)
        phenotypes = 10.0 * rng.random(N) + 1.0
        pheno_df_1 = fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL, bbf.FAM_PHENO_COL]].copy()
        pheno_df_1[bbf.FAM_PHENO_COL] = phenotypes

        actual_N_orig_1, actual_N_1, actual_indices_1, actual_id_df_1, actual_pheno_1 = \
            sut.process_phenotypes(fam_file=fam_df, pheno_file=pheno_df_1)

        pheno_df_2 = pheno_df_1.sample(frac=1, random_state=rng).reset_index(drop=True).copy()

        actual_N_orig_2, actual_N_2, actual_indices_2, actual_id_df_2, actual_pheno_2 = \
            sut.process_phenotypes(fam_file=fam_df, pheno_file=pheno_df_2)

        assert actual_N_orig_1 == actual_N_orig_2
        assert actual_N_1 == actual_N_2
        assert np.array_equal(actual_indices_1, actual_indices_2)
        assert actual_id_df_1.equals(actual_id_df_2)
        assert np.allclose(actual_pheno_1, actual_pheno_2)


    @pytest.mark.parametrize("rng_seed", [382, 99903])
    def test__permute_covar_file_rows__unchanged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100
        fam_df = make_fam_df(N)

        covar_df_1 = pd.DataFrame({bbf.FAM_FID_COL : fam_df[bbf.FAM_FID_COL],
                                 bbf.FAM_IID_COL : fam_df[bbf.FAM_IID_COL],
                                 'COVAR_1' : 10.0 * rng.random(N),
                                 'COVAR_2' : -7.0 * rng.random(N)})

        covar_df_2 = covar_df_1.sample(frac=1, random_state=rng).reset_index(drop=True).copy()

        N_orig_1, N_1, ind_1, fam_cols_1, pheno_1 = \
            sut.process_phenotypes(fam_file=fam_df, covar_file=covar_df_1)


        N_orig_2, N_2, ind_2, fam_cols_2, pheno_2 = \
            sut.process_phenotypes(fam_file=fam_df, covar_file=covar_df_2)

        assert N_orig_1 == N_orig_2
        assert N_1 == N_2
        assert np.array_equal(ind_1, ind_2)
        assert fam_cols_1.equals(fam_cols_2)
        assert np.allclose(pheno_1, pheno_2)


    @pytest.mark.parametrize("rng_seed", [17, 85])
    def test__permute_covar_file_cols__unchanged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100
        fam_df = make_fam_df(N)

        covar_df_1 = pd.DataFrame({bbf.FAM_FID_COL : fam_df[bbf.FAM_FID_COL],
                                 bbf.FAM_IID_COL : fam_df[bbf.FAM_IID_COL],
                                 'COVAR_1' : 10.0 * rng.random(N),
                                 'COVAR_2' : -7.0 * rng.random(N),
                                 'COVAR_3' : 4.5 * rng.random(N)})

        covar_df_2 = covar_df_1[[bbf.FAM_FID_COL, bbf.FAM_IID_COL,
                                 'COVAR_2', 'COVAR_3', 'COVAR_1']].copy()

        N_orig_1, N_1, ind_1, fam_cols_1, pheno_1 = \
            sut.process_phenotypes(fam_file=fam_df, covar_file=covar_df_1)
        

        N_orig_2, N_2, ind_2, fam_cols_2, pheno_2 = \
            sut.process_phenotypes(fam_file=fam_df, covar_file=covar_df_2)

        assert N_orig_1 == N_orig_2
        assert N_1 == N_2
        assert np.array_equal(ind_1, ind_2)
        assert fam_cols_1.equals(fam_cols_2)
        assert np.allclose(pheno_1, pheno_2)



    @pytest.mark.parametrize("rng_seed", [125, 1908])
    def test__permute_sample_file_rows__unchanged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100 + rng.integers(1, 100)

        fam_df = make_fam_df(N)
        sample_df_1 = fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]].copy()

        actual_N_orig_1, actual_N_1, actual_indices_1, actual_id_df_1, actual_pheno_1 = \
            sut.process_phenotypes(fam_file=fam_df, sample_id_file=sample_df_1)

        sample_df_2 = sample_df_1.sample(frac=1, random_state=rng).reset_index(drop=True).copy()

        actual_N_orig_2, actual_N_2, actual_indices_2, actual_id_df_2, actual_pheno_2 = \
            sut.process_phenotypes(fam_file=fam_df, sample_id_file=sample_df_2)

        assert actual_N_orig_1 == actual_N_orig_2
        assert actual_N_1 == actual_N_2
        assert np.array_equal(actual_indices_1, actual_indices_2)
        assert actual_id_df_1.equals(actual_id_df_2)
        assert np.allclose(actual_pheno_1, actual_pheno_2)

    def test__zero__intersection__expected_error(self):
        pass

class TestCalculateOmega:

    def test__already_PSD__no_change(self):
        # 1 pair (0,1) at degree 1; pheno chosen so pheno_variance >> |pair_cov|.
        # For a single pair, pair_cov = -1.0 exactly (analytically), while pheno_var = 24.4.
        # Since lambda_min(off_diag) = -1.0 > -24.4 = -pheno_var, the full
        # omega = pheno_var*I + off_diag is already PSD, so alpha = 1.0 (no shrinkage).
        N = 5
        pheno = np.array([1.0, -1.0, 10.0, 10.0, 10.0])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0],
            sut.FAM_INDEX_2: [1],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING],
        })

        actual_omega = sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno,
            labels=sut.get_relatedness_components(rel_df, N))

        # Expected: pheno_var * I + 1.0 * off_diag  (alpha = 1.0, no shrinkage applied)
        pheno_var = np.var(pheno)
        pair_cov = np.cov(pheno[[0, 1]], pheno[[1, 0]], ddof=0)[0, 1]  # = -1.0
        expected_off_diag = sp.coo_array(
            (np.array([pair_cov, pair_cov]), ([0, 1], [1, 0])), shape=(N, N)
        ).tocsr()
        expected_omega = pheno_var * sp.eye(N) + expected_off_diag

        assert np.allclose(actual_omega.toarray(), expected_omega.toarray())

    def test__not_PSD__shrink_as_expected(self):
        # 1 pair (0,1) at degree 1; pheno chosen so pheno_variance < |pair_cov|.
        # For a single pair, pair_cov = -1.0 exactly, while pheno_var approx = 0.40.
        # Since lambda_min(off_diag) = -1.0 < -0.40 = -pheno_var, the full
        # omega = pheno_var*I + off_diag is NOT PSD, so shrinkage is applied (alpha < 1.0).
        # Cross-validates the closed-form shrinkage formula against the mock binary search.
        N = 5
        pheno = np.array([1.0, -1.0, 0.1, 0.1, 0.1])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0],
            sut.FAM_INDEX_2: [1],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING],
        })

        # Build KING-format data for mock cross-validation
        fam_df = pd.DataFrame({
            bbf.FAM_FID_COL: list(range(N)),
            bbf.FAM_IID_COL: list(range(N)),
            bbf.FAM_IIDF_COL: 0,
            bbf.FAM_IIDM_COL: 0,
            bbf.FAM_SEX_COL: 0,
            bbf.FAM_PHENO_COL: pheno,
        })
        king_rel_df = pd.DataFrame({
            sut.KING_FID1_COL: [0],
            sut.KING_IID1_COL: [0],
            sut.KING_FID2_COL: [1],
            sut.KING_IID2_COL: [1],
            sut.KING_KINSHIP_COL: [0.0],
            sut.KING_REL_COL: [sut.INF_PARENT_OFFSPRING],
        }, columns=sut.NEEDED_KING_COLS)

        actual_omega = sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno,
            labels=sut.get_relatedness_components(rel_df, N))

        # Verify result is PSD
        actual_eigenvalues = np.linalg.eigvalsh(actual_omega.toarray())
        assert np.all(actual_eigenvalues >= -1e-8)

        # Cross-validate: closed-form formula (production) vs. binary search (mock).
        # Both methods should find the same alpha; atol accounts for binary search precision.
        expected_omega = helper.mock_calc_omega(
            fam_df=fam_df, N=N, rel_df=king_rel_df, unresidualized_phenotypes=pheno
        )
        assert np.allclose(actual_omega.toarray(), expected_omega.toarray(), atol=1e-3)


    def test__multiple_components__only_guilty_block_shrunk(self):
        # Two disjoint components carrying the SAME relatedness degree, so both get the same
        # covariance rho and any difference in their shrinkage is due to STRUCTURE alone:
        #
        #   block A, a star 0-1 / 0-2 / 0-3   lambda_min = -rho*sqrt(3)
        #   block B, a single pair 4-5        lambda_min = -rho
        #
        # The phenotypes below put var(y) between the two, so the star breaks
        # positive-definiteness and the pair does not.  A single global alpha would shrink the
        # innocent pair anyway; per-block leaves it untouched.
        N = 10
        pheno = np.array([0.680193, -0.845163, -0.010330, 0.699210, -1.368729,
                          -1.756903, 0.215909, 0.036958, -0.204671, 0.032086])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0, 0, 0, 4],
            sut.FAM_INDEX_2: [1, 2, 3, 5],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * 4,
        })

        rel_to_cov = sut.get_rel_covariances(rel_df=rel_df, unresidualized_phenotypes=pheno)
        rho = rel_to_cov[sut.DEG_PARENT_OFFSPRING]
        pheno_var = float(np.var(pheno))

        # The premise of the test: the star breaks PD, the pair does not
        assert rho < pheno_var < rho * np.sqrt(3.0)

        omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                    unresidualized_phenotypes=pheno,
                                    labels=sut.get_relatedness_components(rel_df, N)).toarray()

        # The pair keeps its unshrunk covariance; the star does not
        assert np.isclose(omega[4, 5], rho)
        assert abs(omega[0, 1]) < abs(rho)
        assert not np.isclose(omega[0, 1], omega[4, 5])

        # What a single global alpha would have done to the innocent pair, for contrast
        global_alpha = min((sut.DEFAULT_OMEGA_EPSILON - pheno_var) / (-rho * np.sqrt(3.0)), 1.0)
        assert not np.isclose(omega[4, 5], global_alpha * rho)

        # And the whole matrix is still positive semi-definite
        assert np.all(np.linalg.eigvalsh(omega) >= -1e-8)


    def test__per_block_shrinkage__result_is_psd(self):
        # The guarantee that matters: shrinking per block must still leave the whole matrix
        # positive semi-definite.
        rng = np.random.default_rng(20260831)
        N = 40
        pheno = rng.normal(size=N)
        pairs = [(i, i + 1) for i in range(0, N - 1, 2)] + [(0, 2), (4, 6), (8, 10)]
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [p[0] for p in pairs],
            sut.FAM_INDEX_2: [p[1] for p in pairs],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * len(pairs),
        })

        omega = sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno,
            labels=sut.get_relatedness_components(rel_df, N))

        assert np.all(np.linalg.eigvalsh(omega.toarray()) >= -1e-8)


    def test__no_related_pairs__omega_is_diagonal(self):
        # With no pairs at all every sample is its own component, off_diag is empty, and omega
        # should come back as pheno_variance on the diagonal and nothing else.
        N = 6
        pheno = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: pd.Series(dtype=np.int64),
            sut.FAM_INDEX_2: pd.Series(dtype=np.int64),
            sut.KING_REL_COL: pd.Series(dtype=np.int64),
        })

        omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                    unresidualized_phenotypes=pheno,
                                    labels=sut.get_relatedness_components(rel_df, N)).toarray()

        assert np.allclose(omega, np.var(pheno) * np.eye(N))


    def test__all_blocks_pass_gershgorin_screen__no_shrinkage(self):
        # When every block's largest absolute row sum is below pheno_variance, the screen proves
        # all of them definite and no eigendecomposition happens at all.  Exercises the
        # early-return path in get_block_alphas().
        N = 8
        pheno = np.array([12.0, -11.0, 9.0, -13.0, 10.0, -9.0, 11.0, -12.0])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0, 2, 4],
            sut.FAM_INDEX_2: [1, 3, 5],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * 3,
        })

        rel_to_cov = sut.get_rel_covariances(rel_df=rel_df, unresidualized_phenotypes=pheno)
        rho = rel_to_cov[sut.DEG_PARENT_OFFSPRING]

        # Premise: each block is a bare pair, so its row sum is |rho|, comfortably under var(y)
        assert abs(rho) < float(np.var(pheno))

        omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                    unresidualized_phenotypes=pheno,
                                    labels=sut.get_relatedness_components(rel_df, N)).toarray()

        # No shrinkage anywhere -- every off-diagonal entry is the raw covariance
        assert np.isclose(omega[0, 1], rho)
        assert np.isclose(omega[2, 3], rho)
        assert np.isclose(omega[4, 5], rho)


    def test__degree_without_covariance__expected_error(self):
        # get_rel_covariances() keys rel_to_cov over range(SE_RELATEDNESS + 1), so a degree above
        # that maps to NaN.  process_relatedness() filters those out before calling this function,
        # so it is unreachable in normal use -- but a NaN must not pass silently, because the
        # Gershgorin screen compares against a row sum and every comparison against NaN is False.
        # The block would keep alpha=1, skip its eigendecomposition, and poison omega.
        N = 4
        pheno = np.array([1.0, 2.0, 3.0, 4.0])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0, 2],
            sut.FAM_INDEX_2: [1, 3],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING, sut.DEG_4TH],
        })

        with pytest.raises(ValueError) as ex_info:
            sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno,
                labels=sut.get_relatedness_components(rel_df, N))

        assert str(sut.DEG_4TH) in str(ex_info.value)


class TestProcessRelatedness:

    @pytest.mark.parametrize("tc_key, rel_degree", tcs.params("process_relatedness"))
    def test__precanned_tcs__expected_results(self, tc_key, rel_degree, tmp_path):
        rel_df, full_fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")

        # The analysis sample as the case states or the mocks derive it, in the form
        # process_phenotypes() hands on -- rather than produced by production's process_phenotypes()
        analysis = tcs.get(tc_key, "ANALYSIS_IIDS")
        fam_df = full_fam_df[full_fam_df[bbf.FAM_IID_COL].astype(str).isin(analysis)].reset_index(
            drop=True)
        unres_pheno = tcs.get(tc_key, "PHENOTYPES")

        expected_R = tcs.get(tc_key, "R_MATRICES", rel_degree)
        expected_R = expected_R.toarray() if sp.issparse(expected_R) else expected_R
        expected_res_pheno = tcs.get(tc_key, "RESID_PHENOS", rel_degree)
        expected_affected = sorted(a for a in tcs.get(tc_key, "AFFECTED") if a in analysis)

        # gls_weights is the pseudo-inverse of R @ omega @ R.T.  Production inverts it block by
        # block over the relatedness components; the mocks take one pinv over the whole matrix, a
        # separate route to the same answer.  Every input here is well-conditioned (worst condition
        # number about 25), so the values can be compared directly.  If a case ever makes these
        # matrices near-singular, fix the case rather than weakening this check -- see notes §7.4j
        # for what happened last time.
        expected_gls_weights = tcs.get(tc_key, "GLS_WEIGHTS", rel_degree)


        actual_R, actual_gls_weights, actual_res_pheno, actual_parental = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=full_fam_df,
                                    imputed_parental_file=tcs.write_imputed_file(tc_key, tmp_path))


        # affected_samples are positions in the full, unfiltered fam file
        all_iids = full_fam_df[bbf.FAM_IID_COL].astype(str).to_numpy()
        assert sorted(all_iids[actual_parental.affected_samples]) == expected_affected
        assert np.allclose(expected_R, actual_R.toarray(), atol=0.001)
        assert np.allclose(expected_res_pheno, actual_res_pheno, atol=0.001)
        assert np.allclose(expected_gls_weights, actual_gls_weights.toarray(), atol=1e-6)


    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("rng_seed", [1, 34])
    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED])
    def test__randomized_phenos__expected_results(self, rel_degree, rng_seed, tc_key):
        rng = np.random.default_rng(seed=rng_seed)

        rel_df, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        N = len(fam_df)

        expected_R = tcs.get(tc_key, "R_MATRICES", rel_degree)

        unres_pheno = 10.0 * rng.random(N)
        expected_omega = helper.mock_calc_omega(fam_df=fam_df, N=N, rel_df=rel_df,
                                                unresidualized_phenotypes=unres_pheno)
        expected_se_matrix = expected_R @ expected_omega @ expected_R.T
        
        expected_res_pheno = expected_R @ unres_pheno


        actual_R, actual_gls_weights, actual_res_pheno, actual_parental = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert np.allclose(expected_R.toarray(), actual_R.toarray(), atol=0.001)
        assert np.allclose(expected_res_pheno, actual_res_pheno, atol=0.001)

        # gls_weights is the pseudo-inverse of R @ omega @ R.T.  Production inverts it block by
        # block over the relatedness components; one pinv over the whole matrix is a separate
        # route to the same answer.  Every input here is well-conditioned (worst condition number
        # about 25), so the values can be compared directly.  If a fixture ever makes these
        # matrices near-singular, fix the fixture rather than weakening this check -- see notes
        # §7.4j for what happened last time.
        expected_gls_weights = np.linalg.pinv(expected_se_matrix.toarray())
        assert np.allclose(expected_gls_weights, actual_gls_weights.toarray(), atol=1e-6)


    @pytest.mark.parametrize("rel_degree", ["FS", "AbCd", ""])
    def test__rel_degree_nonint_str__expected_error(self, rel_degree):

        tc_key = tcs.KEY_SIMPLE
        rel_df, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        unres_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy()

        with pytest.raises(ValueError) as ex_info:
            actual_R, actual_gls_weights, actual_res_pheno, actual_parental = \
                sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                        unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert str(rel_degree) in str(ex_info.value)
        assert str(0) in str(ex_info.value)
        assert str(sut.MAX_GRMA_RELATEDNESS) in str(ex_info.value)


    @pytest.mark.parametrize("rel_degree", [1.0, {1, 2, 3}])
    def test__rel_degree_wrong_type__expected_error(self, rel_degree):

        tc_key = tcs.KEY_SIMPLE
        rel_df, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        unres_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy()

        with pytest.raises(TypeError) as ex_info:
            actual_R, actual_gls_weights, actual_res_pheno, actual_parental = \
                sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                        unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert "str" in str(ex_info.value)
        assert "int" in str(ex_info.value)
        assert str(type(rel_degree)) in str(ex_info.value)


    @pytest.mark.parametrize("rel_degree", [-1, sut.MAX_GRMA_RELATEDNESS + 1])
    def test__rel_degree_out_of_range__expected_error(self, rel_degree):

        tc_key = tcs.KEY_SIMPLE
        rel_df, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        unres_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy()

        with pytest.raises(ValueError) as ex_info:
            actual_R, actual_gls_weights, actual_res_pheno, actual_parental = \
                sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                        unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert str(rel_degree) in str(ex_info.value)
        assert str(0) in str(ex_info.value)
        assert str(sut.MAX_GRMA_RELATEDNESS) in str(ex_info.value)


    @pytest.mark.parametrize("rng_seed", [1, 34])
    def test__permute_rel_file__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        rel_df_2 = rel_df_1.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
        N = len(fam_df)


        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_gls_weights_1, actual_res_pheno_1, actual_parental_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        actual_R_2, actual_gls_weights_2, actual_res_pheno_2, actual_parental_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_gls_weights_1.toarray(), actual_gls_weights_2.toarray(), atol=0.001)



    @pytest.mark.parametrize("rng_seed", [5, 74])
    def test__swap_id_cols_in_rel_file__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        cols = sut.NEEDED_KING_COLS.copy()
        fid1_index, iid1_index, fid2_index, iid2_index = \
            cols.index(sut.KING_FID1_COL), cols.index(sut.KING_IID1_COL), \
            cols.index(sut.KING_FID2_COL), cols.index(sut.KING_IID2_COL)
        cols[fid1_index], cols[iid1_index], cols[fid2_index], cols[iid2_index] = \
            cols[fid2_index], cols[iid2_index], cols[fid1_index], cols[iid1_index]
        rel_df_2 = rel_df_1.copy()
        rel_df_2.columns = cols
 
        N = len(fam_df)
        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_gls_weights_1, actual_res_pheno_1, actual_parental_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        actual_R_2, actual_gls_weights_2, actual_res_pheno_2, actual_parental_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df)

        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_gls_weights_1.toarray(), actual_gls_weights_2.toarray(), atol=0.001)


    @pytest.mark.parametrize("rng_seed", [44, 90233])
    def test__rearrange_fam_file__rearranged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_2FAM

        rel_df, fam_df_1 = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")

        N = len(fam_df_1)
        permutation = rng.permutation(N)
        fam_df_2 = fam_df_1.iloc[permutation].reset_index(drop=True)

        
        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_gls_weights_1, actual_res_pheno_1, actual_parental_1 = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df_1, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df_1)

        actual_R_2, actual_gls_weights_2, actual_res_pheno_2, actual_parental_2 = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df_2, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno[permutation], fam_file=fam_df_2)


        assert np.allclose(actual_R_1.toarray()[permutation][:, permutation], actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1[permutation], actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_gls_weights_1.toarray()[permutation][:, permutation], actual_gls_weights_2.toarray(), atol=0.001)


    @pytest.mark.parametrize("rng_seed", [64, 738])
    def test__permute_ids__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df_1 = tcs.get(tc_key, "REL_DF"), tcs.get(tc_key, "FAM_DF")
        N = len(fam_df_1)
        permutation = rng.permutation(N)
        mapping = {i : permutation[i] for i in range(N)}
        
        fam_df_2 = fam_df_1.copy()
        fam_df_2[bbf.FAM_FID_COL] = fam_df_1[bbf.FAM_FID_COL].map(mapping)
        fam_df_2[bbf.FAM_IID_COL] = fam_df_1[bbf.FAM_IID_COL].map(mapping)

        rel_df_2 = rel_df_1.copy()
        rel_df_2[sut.KING_FID1_COL] = rel_df_1[sut.KING_FID1_COL].map(mapping)
        rel_df_2[sut.KING_IID1_COL] = rel_df_1[sut.KING_IID1_COL].map(mapping)
        rel_df_2[sut.KING_FID2_COL] = rel_df_1[sut.KING_FID2_COL].map(mapping)
        rel_df_2[sut.KING_IID2_COL] = rel_df_1[sut.KING_IID2_COL].map(mapping)


        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_gls_weights_1, actual_res_pheno_1, actual_parental_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df_1, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df_1)

        actual_R_2, actual_gls_weights_2, actual_res_pheno_2, actual_parental_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df_2, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno, fam_file=fam_df_2)


        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_gls_weights_1.toarray(), actual_gls_weights_2.toarray(), atol=0.001)


    def test__fully_randomized__expected_results(self):
        pass


class TestGetBlockAlphas:

    def test__gershgorin_screen_is_safe(self):
        # The Gershgorin screen skips blocks it can prove are definite.  Verify it never skips one
        # that needed shrinking, by checking every returned alpha against a direct per-block
        # eigendecomposition.
        rng = np.random.default_rng(1234)
        N = 60
        pheno = rng.normal(size=N)
        pairs = [(i, i + 1) for i in range(0, N - 1, 3)] + [(0, 3), (6, 9)]
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [p[0] for p in pairs],
            sut.FAM_INDEX_2: [p[1] for p in pairs],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * len(pairs),
        })
        rel_to_cov = sut.get_rel_covariances(rel_df=rel_df, unresidualized_phenotypes=pheno)
        i1 = rel_df[sut.FAM_INDEX_1].to_numpy(np.int64)
        i2 = rel_df[sut.FAM_INDEX_2].to_numpy(np.int64)
        data = np.tile(rel_df[sut.KING_REL_COL].map(rel_to_cov).to_numpy(), 2)
        off_diag = sp.coo_array((data, (np.concatenate([i1, i2]),
                                        np.concatenate([i2, i1]))), shape=(N, N)).tocsr()
        pheno_var = float(np.var(pheno))

        labels = sut.get_relatedness_components(rel_df, N)
        alphas = sut.get_block_alphas(off_diag, pheno_var, labels)

        for c in range(labels.max() + 1):
            idx = np.flatnonzero(labels == c)
            if len(idx) < 2:
                continue
            block = off_diag[idx][:, idx].toarray()
            lmin = float(np.linalg.eigvalsh(block).min())
            expected = (min((sut.DEFAULT_OMEGA_EPSILON - pheno_var) / lmin, 1.0)
                        if lmin < 0.0 else 1.0)
            assert np.isclose(alphas[c], expected, atol=1e-6)


class TestGetLambdaMin:

    @staticmethod
    def _banded_matrix(N: int, seed: int):
        rng = np.random.default_rng(seed)
        vals = rng.normal(size=N - 1)
        rows = np.concatenate([np.arange(N - 1), np.arange(1, N)])
        cols = np.concatenate([np.arange(1, N), np.arange(N - 1)])
        return sp.coo_array((np.concatenate([vals, vals]), (rows, cols)), shape=(N, N)).tocsr()

    @pytest.mark.parametrize("N", [sut.DENSE_LAMBDA_MIN_MAX_SIZE,
                                   sut.DENSE_LAMBDA_MIN_MAX_SIZE + 1,
                                   sut.DENSE_LAMBDA_MIN_MAX_SIZE + 100])
    def test__both_paths_agree_with_dense_truth(self, N):
        # get_lambda_min routes small matrices to a dense eigendecomposition and larger ones to
        # ARPACK.  Both sides of that boundary must agree with the dense answer, otherwise a
        # mistaken comparison would silently send everything down one path.
        B = self._banded_matrix(N, seed=99)

        expected = float(np.linalg.eigvalsh(B.toarray()).min())

        assert np.isclose(sut.get_lambda_min(B), expected, atol=1e-6)


    def test__repeated_calls_are_deterministic(self):
        # ARPACK starts from a random vector unless given one, which made repeated runs differ at
        # the ULP level.  get_lambda_min fixes the start vector; this pins that down.
        B = self._banded_matrix(sut.DENSE_LAMBDA_MIN_MAX_SIZE + 200, seed=7)

        results = [sut.get_lambda_min(B) for _ in range(5)]

        assert len(set(results)) == 1



class TestProcessBimFile:

    @pytest.mark.parametrize("tc_key", tcs.key_params("process_bim_file"))
    def test__precanned_tcs__expected_results(self, tc_key, tmp_path):
        bim_df = tcs.get(tc_key, "BIM_DF")

        # In imputed mode the SNPs are aligned against the imputed file, which is the only thing
        # process_bim_file() reads from the parental sources -- so only its metadata is filled in
        parental_sources = None
        imputed_file = tcs.write_imputed_file(tc_key, tmp_path)
        if imputed_file is not None:
            parental_sources = sut.ParentalSources(
                affected_samples=np.array([], dtype=int),
                observed_parent_cols=np.empty((0, 2), dtype=int),
                imputed_rows=np.array([], dtype=int), imputed_file=imputed_file,
                snipar_metadata=snipar_hdf5.read_snipar_metadata(imputed_file))

        expected_M_orig = len(bim_df)
        expected_snp_filter = tcs.snp_filter(tc_key)
        expected_M = len(expected_snp_filter) if expected_snp_filter is not None else expected_M_orig

        actual_M_orig, actual_M, actual_snp_filter, _ = sut.process_bim_file(
            bim_file=bim_df, snp_list=tcs.get(tc_key, "SNP_LIST"), parental_sources=parental_sources)

        assert expected_M_orig == actual_M_orig
        assert expected_M == actual_M
        assert (expected_snp_filter is None and actual_snp_filter is None) or \
               np.allclose(expected_snp_filter, actual_snp_filter)


    @pytest.mark.parametrize("rng_seed", [95, 301, 486])
    def test__snp_list_no_missing__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 100

        bim_df = make_bim_df(M)

        snp_mask = rng.choice([True, False], size=M)

        expected_M = np.sum(snp_mask)
        expected_snp_filter = np.where(snp_mask)[0]
        snp_list = bim_df[[bbf.BIM_RSID_COL]].loc[snp_mask]

        actual_M_orig, actual_M, actual_snp_filter, _ = sut.process_bim_file(
            bim_file=bim_df, snp_list=snp_list)

        assert actual_M_orig == M
        assert actual_M == expected_M
        assert np.allclose(actual_snp_filter, expected_snp_filter)


    @pytest.mark.parametrize("rng_seed", [1, 10])
    def test__shuffle_snp_list__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        tc_key = tcs.KEY_VARIED
        bim_df = tcs.get(tc_key, "BIM_DF")

        expected_M_orig = len(bim_df)

        expected_snp_filter = tcs.snp_filter(tc_key)
        expected_M = len(expected_snp_filter) if expected_snp_filter is not None else expected_M_orig

        actual_M_orig, actual_M, actual_snp_filter, _ = sut.process_bim_file(
            bim_file=bim_df, snp_list=tcs.get(tc_key, "SNP_LIST").sample(frac=1, random_state=rng))

        assert expected_M_orig == actual_M_orig
        assert expected_M == actual_M
        assert (expected_snp_filter is None and actual_snp_filter is None) or \
               np.allclose(expected_snp_filter, actual_snp_filter)


class TestGetResidualizedGenotypeData:

    # TODO(jonbjala)  Test some with an actual bed file?
    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED])
    def test__precanned_tcs__expected_results(self, tc_key, rel_degree):

        snp_filter = tcs.snp_filter(tc_key)
        sample_filter = tcs.sample_filter(tc_key)
        expected_resid_G = tcs.get(tc_key, "RESID_GENOS", rel_degree)

        orig_G = tcs.get(tc_key, "G")

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=orig_G,
            M_orig=orig_G.shape[0], N_orig=orig_G.shape[1], snp_filter=snp_filter,
            sample_filter=sample_filter, R=tcs.get(tc_key, "R_MATRICES", rel_degree),
            M_start=0, num_snps=orig_G.shape[0])

        assert actual_resid_G.shape == expected_resid_G.shape
        assert np.allclose(actual_resid_G, expected_resid_G)


    @pytest.mark.parametrize("rng_seed", [2, 34])
    def test__simple_bed_file_with_identity_R__return_unchanged(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=0, num_snps=M)


        assert actual_M == M
        assert actual_resid_G.shape == (M, N)
        assert np.allclose(actual_resid_G, G)


    @pytest.mark.parametrize("rng_seed", [33, 904])
    def test__request_contiguous_subset_of_snps__return_expected(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        M_start = rng.integers(1, M >> 1)
        num_snps = rng.integers(1, M >> 1)

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, 
            M_start=M_start, num_snps=num_snps)


        assert actual_M == num_snps
        assert actual_resid_G.shape == (num_snps, N)
        assert np.allclose(actual_resid_G, G[M_start:M_start+num_snps])


    @pytest.mark.parametrize("rng_seed", [2, 4])
    def test__request_snps_past_end_of_array__return_remainder(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        M_start = rng.integers(M >> 1, M)
        num_snps = 10 * M
        expected_M = M - M_start

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, 
            M_start=M_start, num_snps=num_snps)

        assert actual_M == expected_M
        assert actual_resid_G.shape == (expected_M, N)
        assert np.allclose(actual_resid_G, G[M_start:M_start+expected_M])


    @pytest.mark.parametrize("rng_seed", [4, 86])
    def test__simple_bed_file__expected_residualization(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(rng.random(size=(N, N)))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=0, num_snps=M)


        assert actual_M == M
        assert actual_resid_G.shape == (M, N)
        assert np.allclose(actual_resid_G, G @ R.T)


    @pytest.mark.parametrize("rng_seed", [55, 76])
    def test__include_nans__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(rng.random(size=(N, N)))

        x, y = rng.integers(0, M), rng.integers(0, N)

        G[x, y] = np.nan

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=0, num_snps=M)

        expected_resid_G = np.nan_to_num(G @ R.T)

        assert actual_M == M
        assert actual_resid_G.shape == (M, N)
        assert np.allclose(actual_resid_G, expected_resid_G)


    # The three simple filtering cases differ only in which filters are supplied, so they are one
    # test over the combinations.  Filtering that interacts with a BLOCK -- where the filter's
    # indices are offsets into the whole file rather than into what was read -- is a separate
    # case below, since that is where the arithmetic actually gets interesting.
    @pytest.mark.parametrize("use_snp_filter, use_sample_filter", [
        pytest.param(True, False, id="snps_only"),
        pytest.param(False, True, id="samples_only"),
        pytest.param(True, True, id="snps_and_samples"),
    ])
    def test__filters__expected_filtering(self, use_snp_filter, use_sample_filter):
        rng = np.random.default_rng(seed=7)
        M = 20
        N = 100

        snp_filter = (np.sort(rng.choice(M, size=rng.integers(M // 3, 2 * M // 3), replace=False))
                      if use_snp_filter else None)
        sample_filter = (np.sort(rng.choice(N, size=rng.integers(N // 3, 2 * N // 3),
                                            replace=False)) if use_sample_filter else None)

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N if sample_filter is None else len(sample_filter)))

        expected = G if snp_filter is None else G[snp_filter]
        expected = expected if sample_filter is None else expected[:, sample_filter]

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=M, N_orig=N, snp_filter=snp_filter, sample_filter=sample_filter,
            R=R, M_start=0, num_snps=M)

        assert actual_M == expected.shape[0]
        assert actual_resid_G.shape == expected.shape
        assert np.allclose(actual_resid_G, expected)


    @pytest.mark.parametrize("rng_seed", [26, 47, 880])
    def test__snp_filter_some_snps__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        M_start = 2 * rng.integers(0, M >> 3) + 1
        num_snps = M >> 1
        snp_filter = np.array([2*i for i in range(M >> 1)])
        snp_mask = np.zeros(M, dtype=bool); snp_mask[snp_filter] = True

        expected_M = num_snps >> 1

        G = rng.choice(M, size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=snp_filter, sample_filter=None, R=R, M_start=M_start, num_snps=num_snps)

        assert actual_M == expected_M
        assert actual_resid_G.shape == (expected_M, N)
        assert np.allclose(actual_resid_G, G[M_start:M_start+num_snps][snp_mask[M_start:M_start+num_snps]])


    @pytest.mark.parametrize("rng_seed", [8])
    def test__invalid_M_start__expected_error(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = rng.integers(20, 40)
        N = 100

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        M_start = rng.integers(-2* M, -M)
        with pytest.raises(ValueError) as ex_info:
            actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
                M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=M_start, num_snps=N)


        assert str(M_start) in str(ex_info.value)
        assert str(M-1) in str(ex_info.value)


        with pytest.raises(ValueError) as ex_info:
            actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
                M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=M, num_snps=N)

        assert str(M) in str(ex_info.value)


    @pytest.mark.parametrize("rng_seed", [102])
    def test__invalid_num_snps__expected_error(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = rng.integers(20, 40)
        N = 100

        G = rng.choice([0.0, 1.0, 2.0], size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        num_snps = rng.integers(-2*M, -M)
        with pytest.raises(ValueError) as ex_info:
            actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
                M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=0, num_snps=num_snps)


        assert str(num_snps) in str(ex_info.value)

        with pytest.raises(ValueError) as ex_info:
            actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
                M_orig=M, N_orig=N, snp_filter=None, sample_filter=None, R=R, M_start=0, num_snps=0)

        assert "(0)" in str(ex_info.value)


    # ---- Parental residualization --------------------------------------------------------

    @staticmethod
    def trio_sources(affected, parents, imputed_rows=None):
        """Builds a ParentalSources from (affected sample, (father, mother)) positions"""

        return sut.ParentalSources(
            affected_samples=np.array(affected, dtype=np.int64),
            observed_parent_cols=np.array(parents, dtype=np.int64).reshape(len(affected), 2),
            imputed_rows=imputed_rows,
            imputed_file=None if imputed_rows is None else "unused in these tests",
            snipar_metadata=None if imputed_rows is None else {})


    def test__affected_sample__differenced_against_its_parents(self):
        # Sample 2 is the child of samples 0 and 1, and the only one residualized against parents.
        # With an identity R, what comes back is the raw genotype for everyone else and the
        # Mendelian residual for the child.
        G = np.array([[2.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 2.0, 0.0]])
        R = sp.csr_array(np.identity(4))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=2, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=2, parental_sources=self.trio_sources([2], [[0, 1]]))

        # SNP 0: parents 2 and 0 -> mean 1.0, child 1.0 -> residual 0.0
        # SNP 1: parents 1 and 1 -> mean 1.0, child 2.0 -> residual 1.0
        assert np.allclose(actual[:, 2], [0.0, 1.0])

        # Nobody else is touched
        assert np.allclose(actual[:, [0, 1, 3]], G[:, [0, 1, 3]])


    def test__relatives_of_an_affected_sample__center_against_the_differenced_value(self):
        # The ordering that matters.  Subtracting the parental means BEFORE multiplying by R, or
        # after, makes no difference to the affected individual -- their identity row passes them
        # through either way.  It changes the answer for everyone centering against them.
        #
        # Samples 2 and 3 are siblings; 2 is a trio and so is differenced, 3 is not.  Sample 3's
        # row centers the pair, so what it subtracts depends on whether sample 2 has already had
        # its parents removed.
        G = np.array([[2.0, 0.0, 1.0, 2.0]])
        parental_sources = self.trio_sources([2], [[0, 1]])

        # Sample 2 gets an identity row; sample 3 centers against both of them
        R = sut.calculate_R(
            pd.DataFrame({sut.FAM_INDEX_1 : [2], sut.FAM_INDEX_2 : [3],
                          sut.KING_REL_COL : [sut.DEG_FULLSIB]}),
            N=4, affected_samples=np.array([2]))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=1, parental_sources=parental_sources)

        # Parents are 2 and 0, so the mean is 1.0 and sample 2 differences to 0.0.  Sample 3 then
        # centers against that: 2.0 - (0.0 + 2.0) / 2 = 1.0.
        #
        # Had R been applied first, sample 3 would have centered against the RAW 1.0, giving
        # 2.0 - (1.0 + 2.0) / 2 = 0.5, and its own parental mean would then have been subtracted
        # from a value that had already been centered.
        assert np.isclose(actual[0, 2], 0.0)
        assert np.isclose(actual[0, 3], 1.0)


    def test__parent_outside_the_analysis_sample__still_used(self):
        # A parent dropped for a missing phenotype still has perfectly good genotypes, and the
        # whole reason the parental means are taken before the sample filter is applied is so
        # that they remain reachable.  Here sample 0 is the father and is NOT in sample_filter.
        G = np.array([[2.0, 0.0, 1.0, 1.0]])
        sample_filter = np.array([1, 2, 3])
        R = sp.csr_array(np.identity(len(sample_filter)))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=4, snp_filter=None, sample_filter=sample_filter, R=R,
            M_start=0, num_snps=1, parental_sources=self.trio_sources([2], [[0, 1]]))

        # Sample 2 is at position 1 of the filtered array.  Its parents are 2 and 0, so the mean
        # is 1.0 and the residual is 0.0 -- which needs the father's genotype, which the filter
        # would have removed had it been applied first.
        assert actual.shape == (1, 3)
        assert np.isclose(actual[0, 1], 0.0)


    def test__genotypes_passed_in_as_an_array__not_modified(self):
        # With no SNP filter the block is a VIEW of the caller's array, so subtracting in place
        # would reach back into it.  process_genotypes() reuses that array for every block, so the
        # second block would be working from genotypes the first had already differenced.
        G = np.array([[2.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 2.0, 0.0]])
        before = G.copy()
        R = sp.csr_array(np.identity(4))

        sut.get_residualized_genotype_data(
            bed_file=G, M_orig=2, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=2, parental_sources=self.trio_sources([2], [[0, 1]]))

        assert np.array_equal(before, G)


    @pytest.mark.parametrize("parental_sources", [
        pytest.param(None, id="no_parental_processing"),
        pytest.param(sut.empty_parental_sources(), id="nobody_affected"),
    ])
    def test__no_parental_residualization__genotypes_untouched(self, parental_sources):
        # Two separate guards: no imputed file and no trios at all, versus trios looked for and
        # none found.  Both mean the same thing here.
        G = np.array([[2.0, 0.0, 1.0, 1.0]])
        R = sp.csr_array(np.identity(4))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=1, parental_sources=parental_sources)

        assert np.allclose(actual, G)


    def test__imputed_block_supplied__used_for_the_parent_that_was_not_genotyped(self):
        # One parent genotyped and the other imputed.  The imputed arrays arrive already sliced to
        # this block and already flipped, and are handed straight to the mean calculation.
        G = np.array([[2.0, 9.0, 1.0, 9.0]])
        R = sp.csr_array(np.identity(4))

        imputed_block = {"imputed_gts" : np.array([[1.25]]),
                         "num_obs_par_alleles" : np.array([[3.0]]),
                         "standard_f" : np.array([0.25])}

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=1,
            parental_sources=self.trio_sources([2], [[0, -1]], imputed_rows=np.array([0])),
            imputed_block=imputed_block)

        # corrected = 1.25 - 0.25 * (4 - 3) = 1.0; mean = (2.0 + 1.0) / 2 = 1.5
        assert np.isclose(actual[0, 2], 1.0 - 1.5)


    def test__affected_sample_who_is_also_a_parent__differenced_against_raw_genotypes(self):
        # Three generations.  Sample 2 is the child of 0 and 1, and the parent of 4 along with 3.
        # Every mean is computed before any subtraction is applied, so sample 4 is differenced
        # against sample 2's RAW genotype rather than against one already adjusted.
        G = np.array([[2.0, 0.0, 1.0, 1.0, 1.0]])
        R = sp.csr_array(np.identity(5))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=5, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=1, parental_sources=self.trio_sources([2, 4], [[0, 1], [2, 3]]))

        # Sample 2: parents 2 and 0 -> mean 1.0 -> residual 0.0
        # Sample 4: parents 1 (RAW sample 2) and 1 -> mean 1.0 -> residual 0.0
        #
        # Had sample 2 been differenced first, sample 4's mean would be (0.0 + 1.0) / 2 = 0.5 and
        # its residual 0.5 instead.
        assert np.allclose(actual[0, [2, 4]], [0.0, 0.0])


    def test__missing_parental_genotype__knocks_out_the_relatedness_group(self):
        # A parent missing at a SNP makes that SNP missing for the child, and the NaN then spreads
        # through R to everyone centering against them before being zeroed.  The whole group loses
        # the SNP rather than just the child.
        #
        # This is conservative rather than wrong: zeroing before R instead would treat the missing
        # genotype as homozygous reference and bias the group's mean, where zeroing after discards
        # a contrast that cannot be computed.
        G = np.array([[np.nan, 0.0, 1.0, 2.0]])
        R = sut.calculate_R(
            pd.DataFrame({sut.FAM_INDEX_1 : [2], sut.FAM_INDEX_2 : [3],
                          sut.KING_REL_COL : [sut.DEG_FULLSIB]}),
            N=4, affected_samples=np.array([2]))

        actual, _ = sut.get_residualized_genotype_data(
            bed_file=G, M_orig=1, N_orig=4, snp_filter=None, sample_filter=None, R=R,
            M_start=0, num_snps=1, parental_sources=self.trio_sources([2], [[0, 1]]))

        # Sample 2 is missing, and sample 3 centers against it, so both come back as zero
        assert np.allclose(actual[0, [2, 3]], [0.0, 0.0])


class TestCalculateParentalMeans:

    # Both parents genotyped means there is nothing to impute, and that arises in BOTH modes:
    # in fam file trio mode there is no imputed file at all (imputed_rows is None), and in imputed
    # parent mode the family simply has no row in it (imputed_rows is -1).  The two take different
    # routes through calculate_parental_means() -- None short-circuits before the imputed branch
    # is considered, -1 reaches the guard and is turned away by it -- and must agree, which is
    # what running the same expectations through both asserts.
    @pytest.mark.parametrize("imputed_rows, imputed_file, snipar_metadata", [
        pytest.param(None, None, None, id="fam_file_trio_mode"),
        pytest.param(np.array([-1]), "unused -- nobody has an imputed row", {},
                     id="imputed_mode_family_has_no_imputed_row"),
    ])
    def test__both_parents_observed__expected_results(self, imputed_rows, imputed_file,
                                                      snipar_metadata):
        # Dimensions are all distinct -- 3 SNPs, 5 samples, 1 of them affected -- so that a
        # transposition cannot pass unnoticed.
        #
        # Sample 0 is the father, 1 the mother, 2 their child; 3 and 4 are unrelated and should be
        # untouched.  The three SNPs give three different parental means, so a bug returning a
        # constant cannot pass:
        #
        #     SNP 0: father 2, mother 0 -> mean 1.0
        #     SNP 1: father 1, mother 0 -> mean 0.5
        #     SNP 2: father 2, mother 2 -> mean 2.0
        #
        # SNP 0 is worth checking by hand.  A parent with genotype 2 always transmits that allele
        # and a parent with genotype 0 never does, so the child's genotype is forced to 1 and the
        # residual must be exactly 0.  Subtracting half the TRANSMITTED ALLELE instead of half the
        # GENOTYPE would give 0.5 here, which is the error this case exists to catch.
        #
        # The imputed arrays are deliberately NOT passed in either run: neither route needs them,
        # and the function must not require them.
        G = np.array([[2.0, 0.0, 1.0, 1.0, 0.0],
                      [1.0, 0.0, 1.0, 0.0, 1.0],
                      [2.0, 2.0, 2.0, 1.0, 2.0]])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([2]),
            observed_parent_cols=np.array([[0, 1]]),
            imputed_rows=imputed_rows,
            imputed_file=imputed_file,
            snipar_metadata=snipar_metadata)

        expected = np.array([[1.0],
                             [0.5],
                             [2.0]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

        # The homozygous SNP leaves no Mendelian variation, so the child residualizes to zero
        assert np.isclose(G[0, 2] - actual[0, 0], 0.0)


    # Which of the two parents was genotyped makes no difference to the result -- the parental
    # mean is symmetric in them -- so the same expected values serve both.  Running it both ways
    # catches a bug that masks only one of the two columns.
    @pytest.mark.parametrize("observed_parent_cols", [
        pytest.param(np.array([[0, -1]]), id="father_observed"),
        pytest.param(np.array([[-1, 0]]), id="mother_observed"),
    ])
    def test__one_parent_observed__expected_results(self, observed_parent_cols):
        # 3 SNPs, 5 samples, 1 affected.  Sample 0 is the genotyped parent, sample 1 the child;
        # the other parent was not genotyped, so their contribution comes from the imputed file.
        #
        # The two parents are handled DIFFERENTLY and deliberately so.  For the observed parent
        # the genotype is known, so half of it is the expected transmitted allele.  For the
        # unobserved parent only the transmitted allele itself is known, and it is subtracted
        # whole.  Per SNP:
        #
        #     corrected = imputed - standard_f * (4 - k)     <- the transmitted allele
        #     mean      = (g_observed + corrected) / 2
        #
        #     SNP 0: f=0.25 k=3 imputed=1.25 -> corrected 1.0, observed 2.0 -> mean 1.5
        #     SNP 1: f=0.50 k=3 imputed=0.50 -> corrected 0.0, observed 0.0 -> mean 0.0
        #     SNP 2: f=0.10 k=2 imputed=1.20 -> corrected 1.0, observed 1.0 -> mean 1.0
        #
        # standard_f differs per SNP so that a bug using one SNP's frequency for all of them
        # cannot pass, and k is not constant so that the (4 - k) multiplier is tested rather than
        # merely the presence of a subtraction.
        G = np.array([[2.0, 1.0, 0.0, 1.0, 2.0],
                      [0.0, 0.0, 1.0, 2.0, 1.0],
                      [1.0, 2.0, 2.0, 0.0, 0.0]])

        # Two families in the imputed file, with the affected individual's in the SECOND row, so
        # that a bug reading row 0 regardless cannot pass.  Shape is (families, SNPs).
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [1.25, 0.5, 1.2]])
        num_obs_par_alleles = np.array([[4.0, 4.0, 4.0],
                                        [3.0, 3.0, 2.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([1]),
            observed_parent_cols=observed_parent_cols,
            imputed_rows=np.array([1]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[1.5],
                             [0.0],
                             [1.0]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources,
                                              imputed_gts=imputed_gts,
                                              num_obs_par_alleles=num_obs_par_alleles,
                                              standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__neither_parent_observed__expected_results(self):
        # 3 SNPs, 5 samples, 1 affected -- individual 2, who has no genotyped parent at all.  G is
        # therefore never read for them; it is filled with a sentinel to make that explicit, and
        # the test would fail loudly if any of it leaked into the result.
        #
        # This is the branch where imputed_par_gts holds the AVERAGE of two parents rather than
        # one parent's genotype.  An average carries half the frequency contribution of a sum, so
        # the correction is halved -- and the corrected value then stands for both parents, so it
        # counts twice towards the sum:
        #
        #     corrected = imputed - standard_f * (4 - k) / 2
        #     mean      = (corrected * 2) / 2 = corrected
        #
        # The doubling and the halving cancel, which is worth knowing before anyone "simplifies"
        # them away: they do not cancel in the one-parent-observed branch, where only one applies.
        # Without the divisor this case over-subtracts and can produce a NEGATIVE observed-allele
        # count, which is impossible by construction and is how two earlier errors were caught.
        #
        #     k=[2,3,2]  imputed=[0.75, 1.25, 1.6]  ->  mean [0.5, 1.0, 1.5]
        G = np.full((3, 5), 9.0)

        # Two families, with the affected individual's at the SECOND row, so that a bug reading
        # row 0 regardless cannot pass.  Indexing across SEVERAL live rows is covered by the
        # mixed-configuration case rather than here.
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [0.75, 1.25, 1.6]])
        num_obs_par_alleles = np.array([[4.0, 4.0, 4.0],
                                        [2.0, 3.0, 2.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([2]),
            observed_parent_cols=np.array([[-1, -1]]),
            imputed_rows=np.array([1]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[0.5],
                             [1.0],
                             [1.5]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources,
                                              imputed_gts=imputed_gts,
                                              num_obs_par_alleles=num_obs_par_alleles,
                                              standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

    def test__missing_parental_genotype__snp_missing_for_the_child(self):
        # A genotype missing for either parent at a SNP makes that SNP missing for the child as
        # well: there is no way to say what the child inherited when half of what they inherited
        # from is unknown, and GRMA has no fallback for it.  (The transmitted-allele rule used
        # when one parent is ungenotyped does not help here -- that needs phasing information,
        # which only an imputed parental genotype file provides.)
        #
        # The propagation is currently a consequence of the arithmetic rather than an explicit
        # branch: a NaN enters the parental sum and survives the division.  That is the correct
        # behaviour, but it is fragile -- replacing the sum with something NaN-tolerant, say
        # np.nansum, would silently turn a missing parent into a parent with genotype 0 and make
        # the mean wrong rather than absent.  Hence this test.
        #
        # Sample 0 is the father, 1 the mother, 2 their child.  The father is missing at SNP 1
        # only, so SNPs 0 and 2 must still produce ordinary means:
        #
        #     SNP 0: father 2, mother 0   -> 1.0
        #     SNP 1: father NaN, mother 1 -> NaN
        #     SNP 2: father 1, mother 1   -> 1.0
        G = np.array([[2.0, 0.0, 1.0, 1.0, 0.0],
                      [np.nan, 1.0, 1.0, 0.0, 1.0],
                      [1.0, 1.0, 2.0, 1.0, 2.0]])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([2]),
            observed_parent_cols=np.array([[0, 1]]),
            imputed_rows=None,
            imputed_file=None,
            snipar_metadata=None)

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == (G.shape[0], len(parental_sources.affected_samples))

        # Missing at the SNP where a parent is missing, and ONLY there -- a NaN that spread to
        # the other SNPs would mean one missing call knocked out the whole individual
        assert np.isnan(actual[1, 0])
        assert np.allclose([1.0, 1.0], [actual[0, 0], actual[2, 0]])


    def test__no_affected_samples__empty_result(self):
        # Nobody is residualized against their parents, which happens whenever the fam file lists
        # no parents, or lists none that are usable.  The caller currently skips this function
        # entirely in that case, so the guard lives outside it -- but a function that falls over
        # on an empty input is one bad refactor away from a crash, and an empty result costs
        # nothing to support.
        #
        # empty_parental_sources() is used rather than hand-built arrays so that whatever it
        # produces is what gets tested; the two could otherwise drift apart.
        G = np.array([[2.0, 0.0, 1.0, 1.0, 0.0],
                      [1.0, 0.0, 1.0, 0.0, 1.0],
                      [2.0, 2.0, 2.0, 1.0, 2.0]])

        parental_sources = sut.empty_parental_sources()

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        # One column per affected individual, of which there are none.  The SNP dimension still
        # has to be right: a (0, 0) result would subtract cleanly today and break the moment
        # somebody is affected.
        assert actual.shape == (G.shape[0], 0)

        # Subtracting an empty result from an empty column selection is what the caller does, and
        # it must leave the genotypes untouched
        unchanged = G.copy()
        unchanged[:, parental_sources.affected_samples] -= actual
        assert np.array_equal(G, unchanged)


    def test__siblings_share_both_parents__each_gets_its_own_column(self):
        # Siblings are the ordinary case, not an edge case -- a couple with three children gives
        # three affected individuals all pointing at the same two parent columns.  The parental
        # sum is indexed by affected individual while the genotypes are indexed by parent, so
        # repeated parent indices must not make two siblings collide into one column.
        #
        # Sample 0 is the father, 1 the mother, 2 through 4 their children.  All three have the
        # same parents and so the same expected means, which is the point:
        #
        #     SNP 0: father 2, mother 0 -> 1.0
        #     SNP 1: father 1, mother 0 -> 0.5
        #     SNP 2: father 2, mother 2 -> 2.0
        #
        # Note what this canNOT catch: with every sibling expecting the same value, a bug that
        # computes one column and broadcasts it to the rest passes.  Distinguishing those needs
        # affected individuals whose expected values DIFFER, which the mixed-configuration case
        # provides.
        G = np.array([[2.0, 0.0, 1.0, 1.0, 0.0],
                      [1.0, 0.0, 1.0, 0.0, 1.0],
                      [2.0, 2.0, 2.0, 1.0, 2.0]])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([2, 3, 4]),
            observed_parent_cols=np.array([[0, 1], [0, 1], [0, 1]]),
            imputed_rows=None,
            imputed_file=None,
            snipar_metadata=None)

        expected = np.array([[1.0, 1.0, 1.0],
                             [0.5, 0.5, 0.5],
                             [2.0, 2.0, 2.0]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__affected_samples_out_of_order__columns_follow_that_order(self):
        # Nothing requires affected_samples to be sorted.  It comes out of whichever parental
        # processing branch ran, and in imputed parent mode that is the order of snipar's pedigree
        # records, which bears no relation to the fam file's ordering.
        #
        # So the result's columns correspond to affected_samples POSITIONALLY, not to sample
        # order.  Column 0 belongs to whoever affected_samples names first, whatever their sample
        # index.  Getting this wrong would pair every affected individual with somebody else's
        # parents -- plausible-looking numbers attached to the wrong people.
        #
        # Sample 4 is listed first and sample 1 second, deliberately reversing sample order.
        # Their parents differ, so the two columns must differ too:
        #
        #     individual 4, parents 0 and 2: (2+1)/2, (1+0)/2, (2+0)/2 -> 1.5, 0.5, 1.0
        #     individual 1, parents 3 and 0: (0+2)/2, (2+1)/2, (1+2)/2 -> 1.0, 1.5, 1.5
        G = np.array([[2.0, 9.0, 1.0, 0.0, 9.0],
                      [1.0, 9.0, 0.0, 2.0, 9.0],
                      [2.0, 9.0, 0.0, 1.0, 9.0]])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([4, 1]),
            observed_parent_cols=np.array([[0, 2], [3, 0]]),
            imputed_rows=None,
            imputed_file=None,
            snipar_metadata=None)

        expected = np.array([[1.5, 1.0],
                             [0.5, 1.5],
                             [1.0, 1.5]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__affected_individual_is_also_a_parent__genotypes_not_modified(self):
        # affected_samples and the parent columns can overlap: in a pedigree spanning three
        # generations, somebody is both a child with parents of their own and a parent in turn.
        # That individual's own parental mean is subtracted later, by the caller, but the mean
        # computed for THEIR child must use their raw genotype, not an already-differenced one.
        #
        # What makes that work is that this function only ever reads G.  It computes every mean
        # into a fresh array and leaves the genotypes untouched, so the caller can subtract them
        # all in one go afterwards.  Were G used as scratch space -- an easy optimisation to
        # reach for -- the order in which individuals happened to be processed would start
        # changing the answers.
        #
        # Samples 0 and 1 are grandparents, 2 their child, 3 an unrelated spouse, 4 the grandchild.
        # Individuals 2 and 4 are affected, and 2 appears in both roles:
        #
        #     individual 2, parents 0 and 1: 1.0, 1.0, 1.0
        #     individual 4, parents 2 and 3: 1.0, 1.0, 1.5   <- from the RAW genotype of 2
        #
        # Had G been written to first, individual 4 would come out at 0.5, 0.5, 1.0 instead.
        G = np.array([[2.0, 0.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 2.0, 0.0, 1.0],
                      [0.0, 2.0, 1.0, 2.0, 2.0]])
        G_before = G.copy()

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([2, 4]),
            observed_parent_cols=np.array([[0, 1], [2, 3]]),
            imputed_rows=None,
            imputed_file=None,
            snipar_metadata=None)

        expected = np.array([[1.0, 1.0],
                             [1.0, 1.0],
                             [1.0, 1.5]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

        # The genotypes the caller still needs must come back exactly as they went in
        assert np.array_equal(G_before, G)


    def test__no_parental_information__subtracts_nothing(self):
        # An affected individual with neither a genotyped parent nor an imputed row has nothing
        # to be residualized against, and gets a parental mean of zero.
        #
        # Nothing is raised and nothing is logged.  GRMA code excludes such
        # individuals before they ever reach here -- both of its branches report how many were
        # dropped -- so a check in this function would be a second place enforcing one rule, and
        # would run once per block rather than once per run.  The behaviour is pinned anyway,
        # because "zero" is what the arithmetic gives and somebody should have to change this test
        # deliberately before changing that.
        #
        # G is filled with a sentinel rather than genotypes.  A -1 that escaped the mask would be
        # read by numpy as the LAST sample rather than rejected, so the only way to see that go
        # wrong is for the value it would pick up to be obviously not a parental mean.
        G = np.full((3, 5), 9.0)

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([4]),
            observed_parent_cols=np.array([[-1, -1]]),
            imputed_rows=None,
            imputed_file=None,
            snipar_metadata=None)

        expected = np.zeros((3, 1))

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__mixed_configurations__each_individual_handled_independently(self):
        # Real data has every configuration mixed together in one block, and the masking is what
        # keeps them apart.  Each affected individual is handled by a boolean mask over the
        # parent columns and another over the imputed rows, and the two have to stay aligned with
        # each other and with the output columns.  The cases above each isolate one configuration;
        # this one checks that none of them disturbs its neighbours.
        #
        # Every individual expects a DIFFERENT value, which matters: with identical expectations
        # a bug that computes one column and broadcasts it to the rest would pass.
        #
        # Samples 0-3 are parents, 4 is unused, 5-9 are the affected individuals:
        #
        #   5  both parents observed (0 and 1), no imputed row
        #   6  father observed (2), imputed row 1
        #   7  mother observed (3), imputed row 3
        #   8  neither observed, imputed row 4
        #   9  nothing at all
        #
        # The imputed rows are non-contiguous, so a bug walking them in order rather than reading
        # the row each individual names cannot pass.  k varies by individual AND by SNP, so the
        # (4 - k) multiplier has to be applied per cell rather than per row or per SNP.
        #
        #   5: (2+0)/2, (1+1)/2, (0+2)/2                           -> 1.0,  1.0,   1.0
        #   6: corrected 1.0, 0.5, 0.5; father 1, 2, 1             -> 1.0,  1.25,  0.75
        #   7: corrected 0.5, 0.25, 2.0; mother 2, 0, 1            -> 1.25, 0.125, 1.5
        #   8: corrected 0.5, 0.5, 1.5, halved correction, doubled -> 0.5,  0.5,   1.5
        #   9: nothing                                             -> 0.0,  0.0,   0.0
        G = np.array([[2.0, 0.0, 1.0, 2.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                      [1.0, 1.0, 2.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                      [0.0, 2.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]])

        # Five families.  Rows 0 and 2 belong to nobody here and hold sentinels, so a bug reading
        # the wrong row produces something obviously not a parental mean.
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [1.25, 1.5, 0.6],
                                [9.0, 9.0, 9.0],
                                [1.0, 0.75, 2.0],
                                [0.625, 1.0, 1.6]])
        num_obs_par_alleles = np.array([[4.0, 4.0, 4.0],
                                        [3.0, 2.0, 3.0],
                                        [4.0, 4.0, 4.0],
                                        [2.0, 3.0, 4.0],
                                        [3.0, 2.0, 2.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([5, 6, 7, 8, 9]),
            observed_parent_cols=np.array([[0, 1], [2, -1], [-1, 3], [-1, -1], [-1, -1]]),
            imputed_rows=np.array([-1, 1, 3, 4, -1]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[1.0, 1.0,   1.25,  0.5, 0.0],
                             [1.0, 1.25,  0.125, 0.5, 0.0],
                             [1.0, 0.75,  1.5,   1.5, 0.0]])

        actual = sut.calculate_parental_means(G=G, parental_sources=parental_sources,
                                              imputed_gts=imputed_gts,
                                              num_obs_par_alleles=num_obs_par_alleles,
                                              standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)



class TestCalculateImputedContribution:

    # This function returns the imputed parents' contribution to the PARENTAL SUM, not a mean and
    # not a corrected genotype.  Its caller halves the sum afterwards; nothing here does.
    #
    # Flipping happens before this function sees its inputs, in read_imputed_block(), so no case
    # here involves a flip.

    def test__one_parent_observed__frequency_removed_without_a_divisor(self):
        # snipar fills the parental alleles it could not observe with the population allele
        # frequency.  That term carries population structure, so GRMA takes it back out and keeps
        # only genuinely observed alleles.
        #
        # Where ONE parent was genotyped, the imputed value is that single missing parent's
        # genotype, so the whole frequency contribution is subtracted and the result counts once:
        #
        #     contribution = imputed - standard_f * (4 - k)
        #
        # 3 SNPs, 5 families, 1 affected individual -- all different, so a transposition cannot
        # pass.  k differs between SNPs so that the (4 - k) multiplier is tested rather than
        # merely the presence of a subtraction, and standard_f differs too so that a bug reusing
        # one SNP's frequency fails.  k=4 is deliberately absent: it makes the correction zero,
        # so that SNP would exercise nothing here.  The no-op case has its own test.
        #
        #     SNP 0: f=0.25 k=2 imputed=1.00 -> 1.00 - 0.50 = 0.50
        #     SNP 1: f=0.50 k=3 imputed=0.75 -> 0.75 - 0.50 = 0.25
        #     SNP 2: f=0.10 k=2 imputed=1.30 -> 1.30 - 0.20 = 1.10
        #
        # The affected individual reads family 3, with sentinels in the rows either side, so a bug
        # reading the wrong row produces something obviously not an allele count.
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0]])
        num_obs_par_alleles = np.array([[-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [2.0, 3.0, 2.0],
                                        [-1.0, -1.0, -1.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([7]),
            observed_parent_cols=np.array([[0, -1]]),
            imputed_rows=np.array([3]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[0.5],
                             [0.25],
                             [1.1]])

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__neither_parent_observed__correction_halved_and_result_doubled(self):
        # Where NEITHER parent was genotyped, imputed_par_gts holds the AVERAGE of the two rather
        # than one parent's genotype.  An average carries half the frequency contribution of a
        # sum, so the correction is halved -- and the corrected value then stands for both
        # parents, so it counts twice towards the parental sum:
        #
        #     contribution = (imputed - standard_f * (4 - k) / 2) * 2
        #
        # Both operations are needed and they do NOT cancel, however much they look as though
        # they might: the halving applies only here while the doubling applies only here too, and
        # removing one leaves the other.  Dropping the divisor over-subtracts and can drive the
        # recovered allele count below zero, which is impossible by construction.
        #
        # The inputs are deliberately identical to the one-parent-observed case, so the two tests
        # differ only in which branch runs.  Same imputed values, same k, same frequencies:
        #
        #     one parent observed : 0.50, 0.25, 1.10
        #     neither observed    : 1.50, 1.00, 2.40
        #
        #     SNP 0: f=0.25 k=2 imputed=1.00 -> (1.00 - 0.25) * 2 = 1.50
        #     SNP 1: f=0.50 k=3 imputed=0.75 -> (0.75 - 0.25) * 2 = 1.00
        #     SNP 2: f=0.10 k=2 imputed=1.30 -> (1.30 - 0.10) * 2 = 2.40
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0]])
        num_obs_par_alleles = np.array([[-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [2.0, 3.0, 2.0],
                                        [-1.0, -1.0, -1.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        # Both parent columns are -1, which is what selects this branch
        parental_sources = sut.ParentalSources(
            affected_samples=np.array([7]),
            observed_parent_cols=np.array([[-1, -1]]),
            imputed_rows=np.array([3]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[1.5],
                             [1.0],
                             [2.4]])

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    # k = 4 means snipar observed all four parental alleles and filled in nothing, so there is no
    # frequency contribution to remove.  The correction vanishes -- but the DOUBLING does not,
    # and only one of the two branches has it.  Running both makes that visible: the same imputed
    # values come back unchanged for one observed parent and doubled for none.
    @pytest.mark.parametrize("observed_parent_cols, expected_multiplier", [
        pytest.param(np.array([[0, -1]]), 1.0, id="one_parent_observed"),
        pytest.param(np.array([[-1, -1]]), 2.0, id="neither_parent_observed"),
    ])
    def test__all_parental_alleles_observed__no_frequency_removed(self, observed_parent_cols,
                                                                  expected_multiplier):
        # standard_f is deliberately large and varied.  Nothing should be subtracted whatever it
        # holds, so a frequency that leaked through would be obvious rather than marginal.
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0]])
        num_obs_par_alleles = np.full((5, 3), 4.0)
        standard_f = np.array([0.4, 0.45, 0.5])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([7]),
            observed_parent_cols=observed_parent_cols,
            imputed_rows=np.array([3]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[1.0], [0.75], [1.3]]) * expected_multiplier

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__non_contiguous_imputed_rows__each_individual_reads_its_own(self):
        # imputed_rows holds, for each affected individual, which row of the imputed arrays
        # belongs to their family.  Those rows are not consecutive and bear no relation to the
        # order of the individuals: families are stored sorted by ID, and which of them end up
        # with affected offspring in a given analysis is arbitrary.
        #
        # Three individuals reading rows 0, 2 and 5 of a seven-row file, with sentinel rows
        # between them.  A bug walking the rows in order, or reusing one row, or offsetting by
        # the individual's position would all pick up a sentinel and fail obviously.
        #
        # The branches are mixed as well, so the row lookup is tested under both:
        #
        #     individual at row 0, one parent observed : 0.50, 0.25, 1.10
        #     individual at row 2, neither observed    : 0.75, 1.50, 0.70
        #     individual at row 5, one parent observed : 1.00, 0.25, 2.00
        #
        # Every imputed value is at least as large as the frequency contribution it contains,
        # which is a property of real data: snipar's imputed genotype INCLUDES the filled-in
        # frequency, so removing it cannot drive the recovered allele count below zero.
        imputed_gts = np.array([[1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0],
                                [0.5, 1.25, 0.4],
                                [9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [1.5, 1.25, 2.0],
                                [9.0, 9.0, 9.0]])
        num_obs_par_alleles = np.array([[2.0, 3.0, 2.0],
                                        [-1.0, -1.0, -1.0],
                                        [3.0, 2.0, 3.0],
                                        [-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [2.0, 2.0, 4.0],
                                        [-1.0, -1.0, -1.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([11, 12, 13]),
            observed_parent_cols=np.array([[0, -1], [-1, -1], [-1, 1]]),
            imputed_rows=np.array([0, 2, 5]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[0.5,  0.75, 1.0],
                             [0.25, 1.5,  0.25],
                             [1.1,  0.7,  2.0]])

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)



    def test__some_individuals_have_no_imputed_row__those_columns_are_zero(self):
        # A -1 in imputed_rows means that family had both parents genotyped, so snipar imputed
        # nothing for it and there is no row to read.  Those individuals contribute zero from
        # this function; their parental mean comes entirely from the bed file.
        #
        # The zero has to land in that individual's OWN column.  The imputed arrays are indexed
        # by the live rows only, so the results come back shorter than the list of individuals and
        # have to be scattered back into the right positions -- dropping the -1 individuals
        # instead, or packing the results from the left, would shift everybody after them onto
        # somebody else's values.
        #
        # Four individuals, with the -1 ones second and last so that a shift is visible at both
        # ends:
        #
        #     individual 0: row 1, one parent observed -> 0.50, 0.25, 1.10
        #     individual 1: no imputed row             -> 0.00, 0.00, 0.00
        #     individual 2: row 4, neither observed    -> 0.75, 1.50, 0.70
        #     individual 3: no imputed row             -> 0.00, 0.00, 0.00
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0],
                                [9.0, 9.0, 9.0],
                                [0.5, 1.25, 0.4]])
        num_obs_par_alleles = np.array([[-1.0, -1.0, -1.0],
                                        [2.0, 3.0, 2.0],
                                        [-1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0],
                                        [3.0, 2.0, 3.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([11, 12, 13, 14]),
            observed_parent_cols=np.array([[0, -1], [0, 1], [-1, -1], [0, 1]]),
            imputed_rows=np.array([1, -1, 4, -1]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        expected = np.array([[0.5,  0.0, 0.75, 0.0],
                             [0.25, 0.0, 1.5,  0.0],
                             [1.1,  0.0, 0.7,  0.0]])

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)



    def test__no_individual_has_an_imputed_row__zero_contribution(self):
        # Every affected individual in this block had both parents genotyped, so none of them has
        # a row in the imputed file.  The whole contribution is zero.
        #
        # The pipeline never takes this path, since calculate_parental_means() guards against it,
        # but the shape still has to be right for the addition the caller performs.
        #
        # Deliberately large frequencies and low allele counts, so that anything computed instead
        # of returning zeros would come out visibly wrong rather than near zero.
        imputed_gts = np.full((5, 3), 9.0)
        num_obs_par_alleles = np.full((5, 3), -1.0)
        standard_f = np.array([0.4, 0.45, 0.5])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([11, 12]),
            observed_parent_cols=np.array([[0, 1], [2, 3]]),
            imputed_rows=np.array([-1, -1]),
            imputed_file="unused -- nobody has an imputed row",
            snipar_metadata={})

        expected = np.zeros((3, 2))

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)


    def test__unknown_allele_count__contribution_missing_for_that_cell_only(self):
        # snipar records NaN in num_observed_parental_alleles wherever its backup imputation path
        # fired and it could not establish how many parental alleles it had actually seen.  That
        # is not rare -- it was 13.3% of the cells in a file generated from snipar's own example
        # data.
        #
        # Without the count there is no way to know how much frequency was folded in, so the
        # correction cannot be computed and the contribution is missing for that family at that
        # SNP.  It flows on to a missing parental mean and a SNP knocked out for that individual,
        # which is how a missing parental genotype behaves too.  snipar's own robust estimator
        # masks these entries rather than guessing at them.
        #
        # Two individuals, so that one family's missing count cannot take its neighbour with it,
        # and the NaN sits at a single SNP so the other two must still come out as numbers:
        #
        #     individual 0 (row 1): k = 2, NaN, 3  ->  0.5, NaN, 1.2
        #     individual 1 (row 3): no missing counts  ->  0.75, 1.5, 0.7
        imputed_gts = np.array([[9.0, 9.0, 9.0],
                                [1.0, 0.75, 1.3],
                                [9.0, 9.0, 9.0],
                                [0.5, 1.25, 0.4],
                                [9.0, 9.0, 9.0]])
        num_obs_par_alleles = np.array([[-1.0, -1.0, -1.0],
                                        [2.0, np.nan, 3.0],
                                        [-1.0, -1.0, -1.0],
                                        [3.0, 2.0, 3.0],
                                        [-1.0, -1.0, -1.0]])
        standard_f = np.array([0.25, 0.5, 0.1])

        parental_sources = sut.ParentalSources(
            affected_samples=np.array([11, 12]),
            observed_parent_cols=np.array([[0, -1], [-1, -1]]),
            imputed_rows=np.array([1, 3]),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == (len(standard_f), len(parental_sources.affected_samples))

        # Missing where the count is missing, and nowhere else
        assert np.isnan(actual[1, 0])
        assert not np.isnan(np.delete(actual, 1, axis=0)).any()
        assert not np.isnan(actual[:, 1]).any()

        # The cells either side of the missing one, and the whole of the neighbouring individual,
        # are unaffected
        assert np.allclose([0.5, 1.2], [actual[0, 0], actual[2, 0]])
        assert np.allclose([0.75, 1.5, 0.7], actual[:, 1])


    # Several seeds rather than one.  Each draws a different mix of allele counts, frequencies and
    # branches, and a failure names the seed that produced it.
    @pytest.mark.parametrize("seed", [11, 404, 7919, 90210])
    def test__randomized_snipar_inputs__observed_alleles_recovered(self, seed):
        # The strongest statement that can be made about this function: whatever snipar folded in,
        # taking it back out must leave exactly the parental alleles snipar actually observed.
        # The helper builds those inputs forwards, from alleles to imputed value; this checks that
        # the function runs the same thing backwards.
        #
        # Two errors found during development both showed up here first, and both produced
        # NEGATIVE recovered allele counts -- impossible, since an allele count cannot be below
        # zero.  Hand-written cases pin the arithmetic that was expected; this one catches
        # rearrangements nobody thought to expect.
        rng = np.random.default_rng(seed)

        # Six families, mixed between the two branches, over seven SNPs.  Family count, SNP count
        # and affected count all differ, so a transposition cannot pass.
        neither_observed = rng.random(6) < 0.5
        imputed_gts, num_obs_par_alleles, standard_f, expected = helper.mock_snipar_imputation(
            rng=rng, num_snps=7, neither_observed=neither_observed)

        # Every family has an affected individual here, in order, so that the recovered values
        # line up with the helper's output.  Which parent is genotyped does not matter beyond
        # whether one is, so a single arbitrary column index serves.
        observed_parent_cols = np.where(neither_observed[:, None],
                                        np.array([-1, -1]), np.array([0, -1]))

        parental_sources = sut.ParentalSources(
            affected_samples=np.arange(len(neither_observed)),
            observed_parent_cols=observed_parent_cols,
            imputed_rows=np.arange(len(neither_observed)),
            imputed_file="unused -- the arrays are passed in directly",
            snipar_metadata={})

        actual = sut.calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual)

        # Stated separately because it is the property that caught the real errors, and it fails
        # more legibly than a mismatch against an array of expectations
        assert (actual >= 0.0).all()


class TestClassifyAlleleOrientation:

    # The three masks are NOT mutually exclusive, and the palindromic cases are why.  A caller
    # that treated them as a choice of one would get the ambiguous SNPs wrong.
    @pytest.mark.parametrize("grma_alleles, snipar_alleles, agrees, swapped, strand_differs", [
        # The two files record the same variant the same way round
        pytest.param(("A", "G"), ("A", "G"), True, False, False, id="identical"),
        # The same variant, counted from the other allele -- the imputed genotypes need flipping
        pytest.param(("A", "G"), ("G", "A"), False, True, False, id="swapped"),
        # Read from opposite strands: the letters differ but describe the same two bases
        pytest.param(("A", "G"), ("T", "C"), False, False, True, id="opposite_strand"),
        pytest.param(("A", "G"), ("C", "T"), False, False, True, id="opposite_strand_and_swapped"),
        # Not the same variant at all
        pytest.param(("A", "G"), ("A", "C"), False, False, False, id="incompatible"),
        # A/T and C/G are their own complements, so a strand difference is INDISTINGUISHABLE from
        # the letters agreeing.  Both masks come back true, and nothing here can separate them.
        pytest.param(("A", "T"), ("A", "T"), True, False, True, id="palindromic_at_identical"),
        pytest.param(("A", "T"), ("T", "A"), False, True, True, id="palindromic_at_swapped"),
        pytest.param(("C", "G"), ("C", "G"), True, False, True, id="palindromic_cg_identical"),
        pytest.param(("C", "G"), ("G", "C"), False, True, True, id="palindromic_cg_swapped"),
    ])
    def test__allele_pairs__expected_classification(self, grma_alleles, snipar_alleles, agrees,
                                                    swapped, strand_differs):
        actual = sut.classify_allele_orientation(np.array([grma_alleles]),
                                                 np.array([snipar_alleles]))

        assert [bool(mask[0]) for mask in actual] == [agrees, swapped, strand_differs]


    def test__many_snps_at_once__classified_independently(self):
        # The masks are computed over whole columns, so a SNP's classification must not depend on
        # what its neighbours look like.  Every category appears, in an order that puts unlike
        # cases next to each other.
        grma_alleles = np.array([("A", "T"), ("A", "G"), ("C", "G"), ("A", "G"),
                                 ("A", "G"), ("A", "T"), ("A", "G")])
        snipar_alleles = np.array([("T", "A"), ("A", "G"), ("C", "G"), ("A", "C"),
                                   ("G", "A"), ("A", "T"), ("T", "C")])

        agrees, swapped, strand_differs = sut.classify_allele_orientation(grma_alleles,
                                                                          snipar_alleles)

        assert agrees.shape == swapped.shape == strand_differs.shape == (len(grma_alleles),)
        assert list(agrees) == [False, True, True, False, False, True, False]
        assert list(swapped) == [True, False, False, False, True, False, False]
        assert list(strand_differs) == [True, False, True, False, False, True, True]



class TestGetBlockSnpBounds:

    # A SNP filter retaining 10, 11, 12, then 20, 21, then 30.  The gaps matter: a block can fall
    # entirely inside one, and the ends of the retained range have to be clamped rather than run
    # past.
    SNP_FILTER = np.array([10, 11, 12, 20, 21, 30])

    @pytest.mark.parametrize("M_start, num_to_read, expected", [
        # Ordinary case -- the block covers SNPs 15 to 24, catching the middle pair
        pytest.param(15, 10, (3, 5), id="retained_snps_inside_the_block"),
        # Nothing retained before SNP 10, so a block ending at 9 selects nothing from the front
        pytest.param(0, 10, (0, 0), id="block_entirely_before_any_retained_snp"),
        # Nothing retained after SNP 30 either
        pytest.param(40, 10, (6, 6), id="block_entirely_after_every_retained_snp"),
        # SNPs 15 to 19 fall in the gap between 12 and 20 -- a real case whenever a SNP list
        # skips a region, and the one that makes the caller's lower == upper check worth having
        pytest.param(15, 5, (3, 3), id="block_falls_in_a_gap"),
        # The block starts below the first retained SNP, so the lower bound clamps at 0
        pytest.param(5, 10, (0, 3), id="block_starts_before_the_retained_range"),
        # ...and above the last, so the upper bound clamps at the end
        pytest.param(25, 10, (5, 6), id="block_ends_after_the_retained_range"),
        # Both bounds land exactly on a retained SNP.  The one at M_start belongs to this block;
        # the one at M_start + num_to_read belongs to the NEXT.  A right-sided search on the
        # upper bound would hand SNP 20 to both blocks, and since these bounds slice the
        # genotypes and the imputed data alike, it would be counted twice rather than rejected.
        pytest.param(10, 10, (0, 3), id="retained_snps_exactly_on_both_boundaries"),
    ])
    def test__various_blocks__expected_bounds(self, M_start, num_to_read, expected):
        actual = sut.get_block_snp_bounds(self.SNP_FILTER, M_start, num_to_read)

        assert actual == expected


    # The positions are run separately rather than in a loop so that a failure at one does not
    # hide the others: an off-by-one only at the start of the file looks very different from one
    # everywhere.
    @pytest.mark.parametrize("M_start", [
        pytest.param(0, id="first_block"),
        pytest.param(17, id="middle_block"),
        pytest.param(40, id="last_full_block"),
        pytest.param(45, id="short_final_block"),
    ])
    def test__nothing_filtered_out__bounds_are_the_block_itself(self, M_start):
        # With every SNP retained, the positions into the filter and the SNP indices are the same
        # thing, so the bounds come back as the block's own range.  Worth pinning because it is
        # the common case -- most runs have no SNP list -- and because an off-by-one that the
        # gapped filter above absorbs would show up here plainly.
        M_orig = 50
        snp_filter = np.arange(M_orig)
        num_to_read = min(10, M_orig - M_start)

        assert sut.get_block_snp_bounds(snp_filter, M_start, num_to_read) == (
            M_start, M_start + num_to_read)



class TestGetRelatednessComponents:

    @staticmethod
    def partition(labels):
        """Groups sample indices by their component label

        The labels themselves are whatever the graph library chose to number things with, and
        nothing downstream depends on the numbering -- only on which samples share a component.
        Comparing partitions rather than label arrays keeps these tests from breaking over a
        renumbering that changes nothing.
        """

        groups = {}
        for sample, label in enumerate(labels):
            groups.setdefault(label, set()).add(sample)

        return {frozenset(group) for group in groups.values()}


    @staticmethod
    def relatedness(pairs, degrees=None):
        """Builds a relatedness frame of the shape this function expects"""

        degrees = degrees if degrees is not None else [sut.DEG_FULLSIB] * len(pairs)

        return pd.DataFrame({sut.FAM_INDEX_1 : [pair[0] for pair in pairs],
                             sut.FAM_INDEX_2 : [pair[1] for pair in pairs],
                             sut.KING_REL_COL : degrees})


    def test__chain_of_relatives__all_one_component(self):
        # Components are transitive, and over any distance.  The chain here runs 0-1-2-3, so the
        # two ends are two hops apart and appear in no row together -- nothing pairs 0 with 3, or
        # even with anyone 3 is paired with.  They still share a component.
        #
        # That is the whole reason for computing components rather than grouping pairs.  Omega is
        # block diagonal over components, not over pairs, and a block has to hold everyone
        # reachable within it or the shrinkage is applied to the wrong thing.
        actual = sut.get_relatedness_components(
            self.relatedness([(0, 1), (1, 2), (2, 3)]), N=6)

        assert self.partition(actual) == {frozenset({0, 1, 2, 3}), frozenset({4}), frozenset({5})}


    def test__disjoint_families__separate_components(self):
        # Two families sharing nobody stay apart.  Were they merged, per-block shrinkage would
        # hand one family's worst eigenvalue to the other, which is precisely what shrinking per
        # block rather than globally exists to avoid.
        actual = sut.get_relatedness_components(self.relatedness([(0, 1), (2, 3)]), N=4)

        assert self.partition(actual) == {frozenset({0, 1}), frozenset({2, 3})}


    def test__samples_with_no_relatives__each_alone(self):
        # Samples 2, 3 and 4 appear in no pair.  This is not a degenerate case to be guarded
        # against but the ordinary one: most samples in a large cohort have no relatives in it.
        # They still need labels, since everything indexed by label has to line up with the full
        # sample space.
        actual = sut.get_relatedness_components(self.relatedness([(0, 1)]), N=5)

        assert self.partition(actual) == {frozenset({0, 1}), frozenset({2}),
                                          frozenset({3}), frozenset({4})}


    def test__no_related_pairs_at_all__every_sample_alone(self):
        # An empty relatedness frame, which is a different path from merely having isolated
        # samples: the adjacency is assembled from zero-length arrays rather than from some.
        actual = sut.get_relatedness_components(self.relatedness([]), N=4)

        assert self.partition(actual) == {frozenset({0}), frozenset({1}),
                                          frozenset({2}), frozenset({3})}
        assert len(actual) == 4


    # The partition depends only on WHICH pairs exist, never on how closely related they are.
    # That is what lets the components be worked out before any phenotype has been read, and it is
    # why this is a function of its own rather than something calculate_omega() derives.
    #
    # Three components, one of them with three members, so a bug sensitive to the degrees would
    # have somewhere to show up rather than collapsing everything either way.
    @pytest.mark.parametrize("degrees", [
        pytest.param([sut.DEG_FULLSIB] * 4, id="all_full_siblings"),
        pytest.param([sut.DEG_PARENT_OFFSPRING, sut.DEG_3RD, sut.DEG_FULLSIB, sut.DEG_2ND],
                     id="mixed_degrees"),
        pytest.param([sut.DEG_3RD] * 4, id="all_third_degree"),
    ])
    def test__same_pairs_different_degrees__same_partition(self, degrees):
        actual = sut.get_relatedness_components(
            self.relatedness([(0, 1), (1, 2), (4, 5), (6, 7)], degrees), N=8)

        assert self.partition(actual) == {frozenset({0, 1, 2}), frozenset({3}),
                                          frozenset({4, 5}), frozenset({6, 7})}


    def test__duplicated_and_reversed_pairs__same_partition(self):
        # The relatedness file may list a pair either way round, and may list it more than once;
        # nothing upstream deduplicates.  The adjacency stacks both directions regardless, so a
        # duplicate becomes a repeated entry, and it is worth knowing that changes nothing here.
        #
        # It does NOT follow that duplicates are harmless everywhere: calculate_omega() sums
        # entries that land in the same cell, so a pair listed twice contributes its covariance
        # twice there.  This function is immune because a graph does not care how many times an
        # edge is mentioned.
        messy = self.relatedness([(0, 1), (1, 0), (0, 1), (2, 3)])
        tidy = self.relatedness([(0, 1), (2, 3)])

        assert (self.partition(sut.get_relatedness_components(messy, N=5)) ==
                self.partition(sut.get_relatedness_components(tidy, N=5)))


    @pytest.mark.parametrize("seed", [3, 58, 1024])
    def test__rows_in_any_order__same_partition(self, seed):
        # Nothing orders the relatedness file, and nothing should need to.
        pairs = [(0, 1), (1, 2), (2, 3), (5, 6), (7, 8)]
        expected = self.partition(sut.get_relatedness_components(self.relatedness(pairs), N=9))

        shuffled = list(pairs)
        np.random.default_rng(seed).shuffle(shuffled)

        actual = sut.get_relatedness_components(self.relatedness(shuffled), N=9)

        assert self.partition(actual) == expected


class TestCalculateGLSWeights:

    @staticmethod
    def sibling_pair(variance=2.0, covariance=0.5):
        """Returns (R, omega) for two full siblings and nobody else

        R centers them against each other; omega gives them a shared phenotypic covariance.  Small
        enough that the result can be worked out by hand.
        """

        R = sp.csr_array(np.array([[0.5, -0.5], [-0.5, 0.5]]))
        omega = sp.csr_array(np.array([[variance, covariance], [covariance, variance]]))

        return R, omega


    def test__one_sibling_pair__expected_values(self):
        # Worth following through by hand, since everything else here checks properties rather
        # than numbers.  With variance 2 and covariance 0.5:
        #
        #     R omega R' = (variance - covariance) * R = 1.5 * [[0.5, -0.5], [-0.5, 0.5]]
        #
        # which is rank 1 -- the siblings' contrast is the only direction with any variance, and
        # their shared family mean has none.  Its one non-zero eigenvalue is 1.5, so the
        # pseudo-inverse is that same contrast scaled by 1/1.5, giving 1/3 on the diagonal.
        R, omega = self.sibling_pair(variance=2.0, covariance=0.5)

        expected = np.array([[1.0 / 3.0, -1.0 / 3.0],
                             [-1.0 / 3.0, 1.0 / 3.0]])

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.array([0, 0]))

        assert actual.shape == expected.shape
        assert np.allclose(expected, actual.toarray())


    def test__block_diagonal_inverse__matches_a_single_pinv(self):
        # Inverting block by block is not an approximation: the pseudo-inverse of a block diagonal
        # matrix IS the block diagonal matrix of pseudo-inverses.  Checking that against one pinv
        # over the whole thing is a genuinely separate route to the same answer -- a different
        # algorithm on different sized inputs -- rather than a restatement of what the code does.
        #
        # Two families of three, so the blocks are large enough for the two routes to diverge if
        # the assembly were wrong.
        R = sp.csr_array(np.kron(np.eye(2), np.eye(3) - np.ones((3, 3)) / 3.0))
        omega = sp.csr_array(np.kron(np.eye(2), 2.0 * np.eye(3) + 0.5 * (np.ones((3, 3))
                                                                         - np.eye(3))))
        labels = np.array([0, 0, 0, 1, 1, 1])

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=labels)
        expected = np.linalg.pinv((R @ omega @ R.T).toarray())

        assert np.allclose(expected, actual.toarray())


    def test__several_components__no_weight_crosses_between_them(self):
        # The result has to stay as sparse as the input.  A dense inverse of a biobank-sized
        # matrix is not something that can be held, let alone multiplied, so a weight linking two
        # samples who are not related at all is a correctness problem and a scaling one at once.
        R = sp.csr_array(np.kron(np.eye(3), np.eye(2) - np.ones((2, 2)) / 2.0))
        omega = sp.csr_array(np.kron(np.eye(3), np.array([[2.0, 0.5], [0.5, 2.0]])))
        labels = np.array([0, 0, 1, 1, 2, 2])

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=labels).toarray()

        different_components = labels[:, None] != labels[None, :]
        assert np.allclose(actual[different_components], 0.0)


    def test__sample_with_no_relatives__contributes_nothing(self):
        # Somebody with no relatives and no parental data has a zero row in R, so their row and
        # column of R omega R' are zero as well.  The pseudo-inverse of a zero block is zero, not
        # an infinity: they carry no information and are weighted accordingly.
        R = sp.csr_array(np.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]))
        omega = sp.csr_array(np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]]))

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.array([0, 0, 1])).toarray()

        assert np.isclose(actual[2, 2], 0.0)
        assert np.isfinite(actual).all()


    def test__affected_sample_with_no_relatives__weighted_by_its_own_variance(self):
        # The same person, but residualized against their parents instead.  Their row of R is the
        # identity rather than zero, so their block is their own phenotypic variance and the
        # weight is its reciprocal.  This is how parental residualization recovers somebody who
        # would otherwise have contributed nothing -- the contrast between this case and the one
        # above is the whole point of the feature.
        R = sp.csr_array(np.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]))
        omega = sp.csr_array(np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]]))

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.array([0, 0, 1])).toarray()

        assert np.isclose(actual[2, 2], 1.0 / 2.0)


    def test__every_sample_unrelated__diagonal_of_reciprocals(self):
        # The ordinary case for most of a cohort: no relatives, so every component holds one
        # sample and every block is a single number.  Nothing about the block machinery should
        # misbehave at that size.
        variances = np.array([2.0, 0.5, 4.0])
        R = sp.identity(3, format="csr")
        omega = sp.csr_array(np.diag(variances))

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.arange(3))

        assert np.allclose(np.diag(1.0 / variances), actual.toarray())


    def test__badly_conditioned_block__still_a_pseudo_inverse(self):
        # A block can be badly conditioned and still be entirely real.  Here the two directions
        # differ by a factor of 10^9, and both have to survive: the smaller one is a genuine
        # phenotypic variance, not numerical noise, and discarding it would silently drop that
        # sample from the regression.
        #
        # What this does NOT do is pin the cutoff that separates real directions from noise.  The
        # explicit rcond and numpy's default give the same answer at this ratio, so the test
        # cannot tell them apart, and the right threshold has not been settled in any case -- see
        # the note on it in calculate_gls_weights().  The assertions are therefore about the
        # result being a pseudo-inverse of whatever was kept, which holds under either cutoff.
        R = sp.identity(2, format="csr")
        omega = sp.csr_array(np.diag([1.0, 1e-9]))

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.array([0, 0])).toarray()

        V = (R @ omega @ R.T).toarray()
        assert np.allclose(V @ actual @ V, V)
        assert np.isclose(actual[1, 1], 1e9)


    def test__no_samples__empty_matrix(self):
        # Nothing to invert.  grma() rejects an empty sample filter long before this, so the
        # pipeline never arrives here -- but the assembly concatenates a list of blocks, and an
        # empty list cannot be concatenated, so without the guard this is a crash rather than an
        # empty result.
        empty = sp.csr_array((0, 0))

        actual = sut.calculate_gls_weights(R=empty, omega=empty, labels=np.array([], dtype=int))

        assert actual.shape == (0, 0)
        assert actual.nnz == 0


    def test__components_of_R_rather_than_omega__no_longer_a_pseudo_inverse(self):
        # The partition must be omega's, not R's.  R only ever holds pairs within the user's
        # relatedness threshold while omega holds everything out to third degree, so R's
        # components are a refinement of omega's -- and a refinement splits blocks that R omega R'
        # genuinely connects.
        #
        # Four samples: 0-1 and 2-3 are sibling pairs, and 1-2 are third-degree relatives.  That
        # is one omega component but two R components, and R omega R' has entries linking them.
        # Splitting on R's partition discards those entries, and what comes back is no longer a
        # pseudo-inverse of anything.
        #
        # Nothing here guards against it: the check is that the caller passes the right labels.
        R = sp.csr_array(np.array([[0.5, -0.5, 0.0, 0.0],
                                   [-0.5, 0.5, 0.0, 0.0],
                                   [0.0, 0.0, 0.5, -0.5],
                                   [0.0, 0.0, -0.5, 0.5]]))
        omega = sp.csr_array(np.array([[2.0, 0.5, 0.0, 0.0],
                                       [0.5, 2.0, 0.2, 0.0],
                                       [0.0, 0.2, 2.0, 0.5],
                                       [0.0, 0.0, 0.5, 2.0]]))
        V = (R @ omega @ R.T).toarray()

        from_omega = sut.calculate_gls_weights(R=R, omega=omega,
                                               labels=np.array([0, 0, 0, 0])).toarray()
        from_R = sut.calculate_gls_weights(R=R, omega=omega,
                                           labels=np.array([0, 0, 1, 1])).toarray()

        assert np.allclose(V @ from_omega @ V, V)
        assert not np.allclose(V @ from_R @ V, V)


    def test__affected_sample_among_relatives__R_not_treated_as_symmetric(self):
        # R is generally NOT symmetric.  An affected sample's ROW is the identity, since their
        # genotypes were already residualized against their parents -- but their COLUMN keeps the
        # entries their relatives use to center against them.  Three siblings with one of them
        # affected is the smallest case that shows it.
        #
        # The covariance being inverted is R omega R', and with an asymmetric R the transposes
        # cannot be swapped: R' omega R is a different matrix.  Every other case here happens to
        # use a symmetric R, so this is the one that pins the order.
        relatedness = pd.DataFrame({sut.FAM_INDEX_1 : [0, 0, 1],
                                    sut.FAM_INDEX_2 : [1, 2, 2],
                                    sut.KING_REL_COL : [sut.DEG_FULLSIB] * 3})
        R = sut.calculate_R(relatedness, 3, affected_samples=np.array([2]))
        omega = sp.csr_array(2.0 * np.eye(3) + 0.5 * (np.ones((3, 3)) - np.eye(3)))

        assert not np.allclose(R.toarray(), R.toarray().T), "fixture is meant to be asymmetric"

        actual = sut.calculate_gls_weights(R=R, omega=omega, labels=np.zeros(3, dtype=int)
                                           ).toarray()

        V = (R @ omega @ R.T).toarray()
        assert np.allclose(V @ actual @ V, V)
        assert np.allclose(actual @ V @ actual, actual)


class TestCalculateGLSEstimates:

    # Every case asserts XtWX first, then the standard errors, then the betas.  The three are a
    # chain -- XtWX is shared, the standard errors are its reciprocal square root, and the betas
    # divide a separate numerator by it -- so a failure on the first assertion to break says which
    # link went, where checking the betas first would blame the numerator for everything.

    def test__diagonal_weights__expected_values(self):
        # Hand-workable: three samples weighted 1, 2 and 4, and two SNPs with different answers so
        # that a result computed once and broadcast cannot pass.
        #
        #     SNP 0, g = [1, 0, 2]:  XtWX = 1*1 + 0*2 + 4*4  = 17
        #                            g'Wy = 1*1 + 0*6 + 2*2  = 5     -> beta 5/17
        #     SNP 1, g = [2, 1, 0]:  XtWX = 4*1 + 1*2 + 0*4  = 6
        #                            g'Wy = 2*1 + 1*6 + 0*2  = 8     -> beta 8/6
        #
        # with Wy = [1, 6, 2] from phenotypes [1, 3, 0.5] under those weights.
        genotypes = np.array([[1.0, 0.0, 2.0],
                              [2.0, 1.0, 0.0]])
        gls_weights = sp.csr_array(np.diag([1.0, 2.0, 4.0]))
        weighted_phenotypes = np.array([1.0, 6.0, 2.0])

        betas, ses, XtWX = sut.calculate_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights,
            weighted_phenotypes=weighted_phenotypes)

        assert np.allclose([17.0, 6.0], XtWX)
        assert np.allclose([1.0 / np.sqrt(17.0), 1.0 / np.sqrt(6.0)], ses)
        assert np.allclose([5.0 / 17.0, 8.0 / 6.0], betas)


    def test__correlated_weights__matches_a_per_snp_reference(self):
        # A weight matrix with off-diagonal entries, which is the real case: relatives are
        # correlated, and that is the whole reason for weighting at all.  Five SNPs over four
        # samples, with the two counts differing so a transposition cannot pass.
        rng = np.random.default_rng(176)
        genotypes = rng.integers(0, 3, (5, 4)).astype(float)
        weighted_phenotypes = rng.normal(size=4)

        # Two families of two, each with a correlated block
        block = np.array([[2.0, -0.7], [-0.7, 2.0]])
        gls_weights = sp.csr_array(np.kron(np.eye(2), block))

        expected_betas, expected_ses, expected_XtWX = helper.mock_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights,
            weighted_phenotypes=weighted_phenotypes)

        betas, ses, XtWX = sut.calculate_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights,
            weighted_phenotypes=weighted_phenotypes)

        assert np.allclose(expected_XtWX, XtWX)
        assert np.allclose(expected_ses, ses)
        assert np.allclose(expected_betas, betas)


    def test__identity_weights__reduces_to_ordinary_least_squares(self):
        # Weighting every observation equally is what ordinary least squares does, so the
        # estimates have to come out as the familiar (g'y) / (g'g).  A recognisable special case
        # is worth having: it would catch the weights being ignored, applied on the wrong side, or
        # applied twice.
        genotypes = np.array([[1.0, 0.0, 2.0, 1.0],
                              [2.0, 1.0, 0.0, 1.0]])
        phenotypes = np.array([1.0, 3.0, 0.5, 2.0])
        gls_weights = sp.identity(4, format="csr")

        betas, ses, XtWX = sut.calculate_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights, weighted_phenotypes=phenotypes)

        expected_XtX = np.einsum("ij,ij->i", genotypes, genotypes)

        assert np.allclose(expected_XtX, XtWX)
        assert np.allclose(1.0 / np.sqrt(expected_XtX), ses)
        assert np.allclose(np.einsum("ij,j->i", genotypes, phenotypes) / expected_XtX, betas)


    def test__snp_with_no_information__missing_beta_and_infinite_standard_error(self):
        # A SNP that survives filtering can still carry nothing -- monomorphic in this sample,
        # missing for everybody, or residualized away to zero -- and then XtWX is zero.
        #
        # Nothing is substituted for the results.  An infinite standard error means ZERO weight in
        # any inverse-variance meta-analysis downstream, which is the safe direction to fail in; a
        # small positive number in its place would mean near-infinite confidence instead.  The
        # beta is left as NaN rather than zero for the same reason: no estimate, rather than an
        # estimate of no effect.
        #
        # The second SNP carries real information and must be unaffected, so that one dead SNP
        # cannot take the rest of the block with it.
        genotypes = np.array([[0.0, 0.0, 0.0],
                              [2.0, 1.0, 0.0]])
        gls_weights = sp.csr_array(np.diag([1.0, 2.0, 4.0]))
        weighted_phenotypes = np.array([1.0, 6.0, 2.0])

        betas, ses, XtWX = sut.calculate_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights,
            weighted_phenotypes=weighted_phenotypes)

        assert np.allclose([0.0, 6.0], XtWX)
        assert np.isinf(ses[0]) and np.isclose(ses[1], 1.0 / np.sqrt(6.0))
        assert np.isnan(betas[0]) and np.isclose(betas[1], 8.0 / 6.0)


    def test__returns_one_value_per_snp(self):
        # Genotypes arrive as (SNPs, samples) and the phenotypes as (samples,); everything coming
        # back is per SNP.  The two counts differ here so that a transposition shows up as a shape
        # error rather than as plausible numbers.
        num_snps, num_samples = 6, 4
        rng = np.random.default_rng(4242)
        genotypes = rng.integers(0, 3, (num_snps, num_samples)).astype(float)
        gls_weights = sp.identity(num_samples, format="csr")
        weighted_phenotypes = rng.normal(size=num_samples)

        betas, ses, XtWX = sut.calculate_gls_estimates(
            genotypes=genotypes, gls_weights=gls_weights,
            weighted_phenotypes=weighted_phenotypes)

        assert XtWX.shape == (num_snps,)
        assert ses.shape == (num_snps,)
        assert betas.shape == (num_snps,)


class TestAlignImputedFileToBim:

    # snipar stores the NAMES of its bim columns alongside the values, because the spellings
    # differ between its own code paths -- a file written from real data gives Chr / id /
    # coordinate where the simulation path gives chrom / rsid / position.  These are the real
    # file's spellings; one case below uses the others.
    REAL_FILE_COLUMNS = ["Chr", "id", "morgans", "coordinate", "allele1", "allele2"]

    @staticmethod
    def bim(variants):
        """Builds a bim frame from (rsid, position, A1, A2) tuples, all on chromosome 1"""

        return pd.DataFrame({bbf.BIM_CHR_COL : ["1"] * len(variants),
                             bbf.BIM_RSID_COL : [variant[0] for variant in variants],
                             bbf.BIM_CM_COL : ["0"] * len(variants),
                             bbf.BIM_BP_COL : [str(variant[1]) for variant in variants],
                             bbf.BIM_A1_COL : [variant[2] for variant in variants],
                             bbf.BIM_A2_COL : [variant[3] for variant in variants]})

    @classmethod
    def metadata(cls, variants, columns=None):
        """Builds the imputed file's view of the same variants, as read_snipar_metadata returns it

        Everything in bim_values is a string, including the position: snipar converts the whole
        table at once when it writes the file.

        Columns are laid out to match whatever spellings are asked for, by name rather than by
        position, so that a caller can reorder them to check that the code does the same.
        """

        columns = columns if columns is not None else cls.REAL_FILE_COLUMNS

        by_name = {}
        for spellings, values in (
                (snipar_hdf5.SNIPAR_BIM_CHR_NAMES, ["1"] * len(variants)),
                (snipar_hdf5.SNIPAR_BIM_ID_NAMES, [variant[0] for variant in variants]),
                (snipar_hdf5.SNIPAR_BIM_BP_NAMES, [str(variant[1]) for variant in variants]),
                (snipar_hdf5.SNIPAR_BIM_A1_NAMES, [variant[2] for variant in variants]),
                (snipar_hdf5.SNIPAR_BIM_A2_NAMES, [variant[3] for variant in variants])):
            for spelling in spellings:
                by_name[spelling] = values

        # Anything the aligner does not look up -- the genetic map column, say -- is filler
        filler = ["0"] * len(variants)

        return {
            snipar_hdf5.SNIPAR_BIM_VALUES_KEY : np.array(
                [by_name.get(column, filler) for column in columns]).T,
            snipar_hdf5.SNIPAR_BIM_COLUMNS_KEY : np.array(columns)
        }

    def test__every_variant_matches__all_kept_and_nothing_logged(self, caplog):
        # The ordinary case by a long way: snipar was run on the same bfile GRMA is being run on,
        # so the two bim tables are the same table.
        variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T"), ("rs3", 300, "A", "C")]

        with caplog.at_level(logging.DEBUG):
            keep, hdf5_indices, flip = sut.align_imputed_file_to_bim(
                bim_df=self.bim(variants), metadata=self.metadata(variants))

        assert list(keep) == [True, True, True]
        assert list(hdf5_indices) == [0, 1, 2]
        assert not flip.any()

        # Nothing was dropped, so nothing is said.  A line reporting zero drops on every run
        # would be noise.
        assert caplog.records == []


    def test__alleles_the_other_way_round__kept_and_flagged_for_flipping(self, caplog):
        # The same variant counted from the other allele.  The imputed genotypes are usable, but
        # only after being subtracted from 2, so the SNP is kept and marked rather than dropped.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T"), ("rs3", 300, "A", "C")]
        imputed_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "T", "C"),
                            ("rs3", 300, "C", "A")]

        with caplog.at_level(logging.DEBUG):
            keep, _, flip = sut.align_imputed_file_to_bim(
                bim_df=self.bim(bim_variants), metadata=self.metadata(imputed_variants))

        assert list(keep) == [True, True, True]
        assert list(flip) == [False, True, True]
        assert caplog.records == []


    def test__variants_missing_from_the_imputed_file__dropped_and_counted(self, caplog):
        # snipar imputes only the variants it was given, and restricts to unique ones at that, so
        # its table can be a subset of GRMA's.  Those SNPs simply cannot be residualized against
        # parents, and are dropped rather than half-processed.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T"), ("rs3", 300, "A", "C"),
                        ("rs4", 400, "G", "T")]
        imputed_variants = [("rs1", 100, "A", "G"), ("rs3", 300, "A", "C")]

        with caplog.at_level(logging.INFO):
            keep, _, _ = sut.align_imputed_file_to_bim(
                bim_df=self.bim(bim_variants), metadata=self.metadata(imputed_variants))

        assert list(keep) == [True, False, True, False]

        logged = "\n".join(record.message for record in caplog.records)
        assert "2 SNP(s) (50.0%) dropped" in logged
        assert "2 not present in the imputed parental genotype file" in logged


    def test__different_alleles_entirely__dropped_as_a_different_variant(self, caplog):
        # Same ID, same position, but the alleles do not correspond under any reading -- not
        # swapped, not complemented.  The two files disagree about what the variant is, and
        # nothing can be salvaged from that.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T")]
        imputed_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "A")]

        with caplog.at_level(logging.INFO):
            keep, _, _ = sut.align_imputed_file_to_bim(
                bim_df=self.bim(bim_variants), metadata=self.metadata(imputed_variants))

        assert list(keep) == [True, False]

        logged = "\n".join(record.message for record in caplog.records)
        assert "1 SNP(s) (50.0%) dropped" in logged
        assert "1 incompatible alleles, so not the same variant" in logged


    def test__palindromic_variants_with_no_strand_evidence__kept(self, caplog):
        # A/T and C/G cannot have their strand read off the letters: on the opposite strand an
        # A/T variant still presents as A/T, with only the two alleles exchanged.  So their
        # orientation is ambiguous in isolation.
        #
        # It is resolved by looking at the UNAMBIGUOUS variants instead.  None of them shows a
        # strand difference here, so the two files share a strand, and the palindromic ones can
        # be read at face value.
        variants = [("rs1", 100, "A", "G"), ("rs2", 200, "A", "T"), ("rs3", 300, "C", "G")]

        with caplog.at_level(logging.DEBUG):
            keep, _, flip = sut.align_imputed_file_to_bim(
                bim_df=self.bim(variants), metadata=self.metadata(variants))

        assert list(keep) == [True, True, True]
        assert not flip.any()
        assert caplog.records == []


    def test__strand_difference_somewhere__palindromic_variants_dropped_too(self, caplog):
        # One unambiguous variant recorded on the opposite strand is enough to settle the
        # question the other way: the two files do NOT share a strand, so no palindromic variant
        # can be trusted, including the ones whose letters happen to agree.
        #
        # rs1 matches, rs2 is complemented, rs3 and rs4 are palindromic and agree letter for
        # letter.  Only rs1 survives.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "A", "G"), ("rs3", 300, "A", "T"),
                        ("rs4", 400, "C", "G")]
        imputed_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "T", "C"),
                            ("rs3", 300, "A", "T"), ("rs4", 400, "C", "G")]

        with caplog.at_level(logging.INFO):
            keep, _, _ = sut.align_imputed_file_to_bim(
                bim_df=self.bim(bim_variants), metadata=self.metadata(imputed_variants))

        assert list(keep) == [True, False, False, False]

        logged = "\n".join(record.message for record in caplog.records)
        assert "1 SNP(s) are recorded on opposite strands" in logged
        assert "3 SNP(s) (75.0%) dropped" in logged
        assert "2 palindromic, and the two files disagree about strand" in logged
        assert "1 recorded on the opposite strand" in logged


    def test__same_id_at_a_different_position__raises(self, caplog):
        # An ID matching while the position does not means the two files are using one name for
        # two different variants.  That is not something to drop quietly: every other match is
        # then suspect, and the user needs to know before any of it is used.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T")]
        imputed_variants = [("rs1", 100, "A", "G"), ("rs2", 999, "C", "T")]

        with pytest.raises(ValueError) as error:
            sut.align_imputed_file_to_bim(bim_df=self.bim(bim_variants),
                                          metadata=self.metadata(imputed_variants))

        assert "rs2" in str(error.value)


    def test__alternate_column_spellings__aligned_the_same_way(self, caplog):
        # The names are looked up rather than assumed, so a file from snipar's simulation path
        # aligns exactly as one from real data does.
        #
        # Both the spellings AND the order differ, since the order is not fixed either -- storing
        # the names in the file would be pointless if it were.
        variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T")]
        simulate_path_columns = ["allele2", "position", "rsid", "map", "chrom", "allele1"]

        from_real = sut.align_imputed_file_to_bim(
            bim_df=self.bim(variants), metadata=self.metadata(variants))
        from_simulated = sut.align_imputed_file_to_bim(
            bim_df=self.bim(variants),
            metadata=self.metadata(variants, columns=simulate_path_columns))

        for real_mask, simulated_mask in zip(from_real, from_simulated):
            assert np.array_equal(real_mask, simulated_mask)


    def test__dropped_variant_in_the_middle__later_rows_not_shifted(self, caplog):
        # The three returned arrays are positional against the bim's rows, and the caller narrows
        # all of them with the same mask.  A dropped variant must therefore leave a hole rather
        # than closing the gap: packing the survivors leftwards here would hand rs4 the imputed
        # row belonging to rs3.
        #
        # The imputed file also holds the variants in a different ORDER, so the indices are not
        # merely a range and a bug producing one would be visible.
        bim_variants = [("rs1", 100, "A", "G"), ("rs2", 200, "C", "T"), ("rs3", 300, "A", "C"),
                        ("rs4", 400, "G", "T")]
        imputed_variants = [("rs4", 400, "G", "T"), ("rs3", 300, "A", "C"),
                            ("rs1", 100, "A", "G")]

        with caplog.at_level(logging.INFO):
            keep, hdf5_indices, flip = sut.align_imputed_file_to_bim(
                bim_df=self.bim(bim_variants), metadata=self.metadata(imputed_variants))

        assert list(keep) == [True, False, True, True]
        assert len(hdf5_indices) == len(bim_variants)
        assert len(flip) == len(bim_variants)

        # Each surviving variant points at its own row of the imputed file
        assert hdf5_indices[0] == 2
        assert hdf5_indices[2] == 1
        assert hdf5_indices[3] == 0


class TestLogDroppedSnps:

    # Testing this through align_imputed_file_to_bim() can only reach some of it: a palindromic
    # SNP is dropped only when the two files disagree about strand, so that reason never arrives
    # on its own there.  Calling it directly allows each reason to be isolated.

    @staticmethod
    def bim(num_snps):
        return pd.DataFrame({bbf.BIM_RSID_COL : [f"rs{index}" for index in range(num_snps)]})

    @classmethod
    def log(cls, caplog, num_snps, keep, matched=None, palindromic=None, strand_differs=None,
            strands_agree=True, level=logging.INFO):
        """Calls the function with masks given as lists of booleans, and returns what it logged

        Everything but keep defaults to all-False, so a case naming one mask is describing one
        reason and nothing else.
        """

        def mask(values):
            return np.array(values if values is not None else [False] * num_snps, dtype=bool)

        caplog.clear()
        with caplog.at_level(level):
            sut.log_dropped_snps(bim_df=cls.bim(num_snps),
                                 matched=mask(matched),
                                 keep_mask=np.array(keep, dtype=bool),
                                 palindromic=mask(palindromic),
                                 strand_differs=mask(strand_differs),
                                 strands_agree=strands_agree)

        return "\n".join(record.message for record in caplog.records)


    def test__nothing_dropped__says_nothing(self, caplog):
        # A line reporting zero drops would appear on every ordinary run, where snipar was given
        # the same bfile and everything matches.  Silence is the right output.
        logged = self.log(caplog, num_snps=4, keep=[True] * 4, matched=[True] * 4)

        assert logged == ""
        assert caplog.records == []


    # Each reason in isolation.  The last two are not reachable through the aligner: a
    # palindromic SNP is kept whenever the strands agree, and dropped alongside the strand
    # mismatches when they do not, so neither wording ever appears alone there.
    @pytest.mark.parametrize("masks, expected_phrase", [
        pytest.param({"matched" : [True, False, True, True]},
                     "1 not present in the imputed parental genotype file",
                     id="absent_from_the_imputed_file"),
        pytest.param({"matched" : [True] * 4, "strand_differs" : [False, True, False, False]},
                     "1 recorded on the opposite strand",
                     id="opposite_strand"),
        pytest.param({"matched" : [True] * 4},
                     "1 incompatible alleles, so not the same variant",
                     id="incompatible_alleles"),
        pytest.param({"matched" : [True] * 4, "palindromic" : [False, True, False, False],
                      "strands_agree" : True},
                     "1 palindromic, so their orientation is ambiguous",
                     id="palindromic_while_strands_agree"),
        pytest.param({"matched" : [True] * 4, "palindromic" : [False, True, False, False],
                      "strands_agree" : False},
                     "1 palindromic, and the two files disagree about strand",
                     id="palindromic_while_strands_disagree"),
    ])
    def test__one_reason_at_a_time__reported_with_its_own_phrase(self, caplog, masks,
                                                                expected_phrase):
        logged = self.log(caplog, num_snps=4, keep=[True, False, True, True], **masks)

        assert "1 SNP(s) (25.0%) dropped" in logged
        assert expected_phrase in logged


    def test__several_reasons_at_once__each_counted_separately(self, caplog):
        # Eight SNPs, four of them dropped, one for each reason.  A single total with no breakdown
        # would leave the user unable to tell a version mismatch from a strand problem.
        logged = self.log(
            caplog, num_snps=8,
            keep=[False, True, False, True, False, True, False, True],
            matched=[False, True, True, True, True, True, True, True],
            palindromic=[False, False, True, False, False, False, False, False],
            strand_differs=[False, False, False, False, True, False, False, False],
            strands_agree=False)

        assert "4 SNP(s) (50.0%) dropped" in logged
        assert "1 not present in the imputed parental genotype file" in logged
        assert "1 palindromic, and the two files disagree about strand" in logged
        assert "1 recorded on the opposite strand" in logged
        assert "1 incompatible alleles, so not the same variant" in logged


    def test__percentage__taken_against_the_whole_bim(self, caplog):
        # 2 of 8, which is 25% -- and distinguishable from the alternatives.  Against the number
        # dropped it would be 100%, and against the number matched 28.6%.
        logged = self.log(caplog, num_snps=8,
                          keep=[True, False, True, True, False, True, True, True],
                          matched=[True, False, True, True, True, True, True, True])

        assert "2 SNP(s) (25.0%) dropped" in logged


    def test__ordinary_logging__counts_without_the_identities(self, caplog):
        # The convention is counts at the normal level and per-item detail only when asked for.
        # A run dropping tens of thousands of SNPs would otherwise be unreadable.
        logged = self.log(caplog, num_snps=4, keep=[True, False, True, False],
                          matched=[True, False, True, False], level=logging.INFO)

        assert "2 not present in the imputed parental genotype file" in logged
        assert "rs1" not in logged
        assert "rs3" not in logged


    def test__verbose_logging__identities_as_well(self, caplog):
        # Asked for the detail, the user gets the IDs, so that a handful of unexpected drops can
        # actually be chased down.
        logged = self.log(caplog, num_snps=4, keep=[True, False, True, False],
                          matched=[True, False, True, False], level=logging.DEBUG)

        assert "2 not present in the imputed parental genotype file" in logged
        assert "rs1" in logged
        assert "rs3" in logged



class TestReadImputedBlock:

    # The chunk cache exists because HDF5 cannot serve part of a chunk: asking for a block's worth
    # of SNPs reads the whole chunk containing them and throws the rest away.  Chunk widths run
    # from a few hundred SNPs to several thousand while a block is typically a hundred, so reading
    # per block costs an order of magnitude more than it needs to.

    CHUNK_WIDTH = 8
    NUM_SNPS = 40
    NUM_FAMILIES = 5

    @classmethod
    def imputed_file(cls, tmp_path, seed=99):
        """Writes a small imputed parental genotype file and returns its path and contents

        Written to disk rather than faked, because the chunking is the subject: a stand-in handle
        would not have any.  The width is forced narrow so that a block can span several chunks at
        sizes small enough to read in a test.
        """

        rng = np.random.default_rng(seed)
        neither_observed = rng.random(cls.NUM_FAMILIES) < 0.5
        imputed_gts, num_obs_par_alleles, standard_f, _ = helper.mock_snipar_imputation(
            rng=rng, num_snps=cls.NUM_SNPS, neither_observed=neither_observed)

        path = str(tmp_path / "chr_1.hdf5")
        snipar_hdf5.write_snipar_file(
            imputed_filename=path,
            imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles,
            standard_f=standard_f,
            parental_status=np.zeros((cls.NUM_FAMILIES, 3), dtype=bool),
            pedigree=np.array([[f"F{family}", f"I{family}", "P", "M", "False", "False"]
                               for family in range(cls.NUM_FAMILIES)]),
            families=np.array([f"F{family}" for family in range(cls.NUM_FAMILIES)]),
            bim_values=np.array([["1", f"rs{snp}", "0", str(snp * 100), "A", "G"]
                                 for snp in range(cls.NUM_SNPS)]),
            chunk_snps=cls.CHUNK_WIDTH)

        return path, imputed_gts, num_obs_par_alleles, standard_f

    @staticmethod
    def new_cache():
        return {"cached_start" : -1, "cached_stop" : -1, "gts" : None, "counts" : None}

    @staticmethod
    def reads(caplog):
        """Returns the (start, stop) of every chunk actually read, from what the function logged

        The alternative would be to patch the reading function, which reaches further into the
        module than this does.  It depends on the bracketed range in that debug line staying as it
        is, which is the price of observing the cache without opening it up.
        """

        return [tuple(int(bound) for bound in match.groups())
                for record in caplog.records
                for match in [re.search(r"\[(\d+), (\d+)\)", record.message)] if match]


    @pytest.mark.parametrize("block_size", [
        pytest.param(3, id="several_blocks_per_chunk"),
        pytest.param(8, id="block_exactly_one_chunk"),
        pytest.param(20, id="block_spans_several_chunks"),
        pytest.param(40, id="one_block_for_the_whole_file"),
    ])
    def test__consecutive_blocks__same_data_as_a_direct_read(self, tmp_path, block_size):
        # The property the cache has to preserve: what comes back is what is in the file,
        # whatever size the blocks happen to be relative to the chunks.  Block sizes above and
        # below the chunk width both matter, since the two take different paths through the
        # advance loop.
        path, imputed_gts, num_obs_par_alleles, _ = self.imputed_file(tmp_path)

        with snipar_hdf5.open_snipar_file(path) as handle:
            cache = self.new_cache()
            for start in range(0, self.NUM_SNPS, block_size):
                indices = np.arange(start, min(start + block_size, self.NUM_SNPS))

                block = sut.read_imputed_block(
                    imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                    block_hdf5_indices=indices,
                    block_flip_mask=np.zeros(len(indices), dtype=bool), chunk=cache)

                assert np.allclose(imputed_gts[:, indices], block["imputed_gts"],
                                   equal_nan=True, atol=1e-3)
                assert np.allclose(num_obs_par_alleles[:, indices],
                                   block["num_obs_par_alleles"], equal_nan=True, atol=1e-3)


    def test__non_contiguous_snps__each_one_read_from_its_own_position(self, tmp_path):
        # What a SNP list, or alignment dropping variants, leaves behind: the block's SNPs are
        # scattered through the file rather than being a run of them.  The gaps here are wide
        # enough to skip whole chunks, so a bug treating the indices as a range would be reading
        # from entirely the wrong place.
        path, imputed_gts, _, _ = self.imputed_file(tmp_path)
        indices = np.array([0, 1, 9, 17, 18, 33, 39])

        with snipar_hdf5.open_snipar_file(path) as handle:
            block = sut.read_imputed_block(
                imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                block_hdf5_indices=indices,
                block_flip_mask=np.zeros(len(indices), dtype=bool), chunk=self.new_cache())

        assert np.allclose(imputed_gts[:, indices], block["imputed_gts"],
                           equal_nan=True, atol=1e-3)


    def test__some_snps_flipped__only_those_are_turned_round(self, tmp_path):
        # A SNP the two files count from opposite alleles has its imputed genotypes subtracted
        # from 2 and its frequency from 1.  Both are applied HERE, to the raw values, rather than
        # to the corrected ones downstream -- a genotype always flips as 2 - g and a frequency as
        # 1 - f, where flipping a corrected value needs a constant that varies with how many
        # parental alleles were observed.
        #
        # Allele counts are NOT flipped: how many alleles were seen does not depend on which one
        # is being counted.
        path, imputed_gts, num_obs_par_alleles, standard_f = self.imputed_file(tmp_path)
        indices = np.arange(10, 18)
        flip = np.array([False, True, False, False, True, True, False, False])

        with snipar_hdf5.open_snipar_file(path) as handle:
            block = sut.read_imputed_block(
                imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                block_hdf5_indices=indices, block_flip_mask=flip, chunk=self.new_cache())

        expected_gts = imputed_gts[:, indices].astype(float)
        expected_gts[:, flip] = sut.SNIPAR_ALLELES_PER_PARENT - expected_gts[:, flip]
        expected_f = standard_f[indices].copy()
        expected_f[flip] = 1.0 - expected_f[flip]

        assert np.allclose(expected_gts, block["imputed_gts"], equal_nan=True, atol=1e-3)
        assert np.allclose(expected_f, block["standard_f"])
        assert np.allclose(num_obs_par_alleles[:, indices], block["num_obs_par_alleles"],
                           equal_nan=True, atol=1e-3)


    def test__frequencies__indexed_by_the_file_not_by_the_chunk(self, tmp_path):
        # standard_f comes from the metadata read once at setup, so it is indexed by position in
        # the whole file -- unlike the two genotype arrays, which are indexed into the cached
        # chunk.  Mixing the two up is easy and would go unnoticed for any block starting at a
        # chunk boundary, where the two indices coincide.
        #
        # This block starts at 21, which is 5 into its chunk, so they do not.
        path, _, _, standard_f = self.imputed_file(tmp_path)
        indices = np.arange(21, 27)

        with snipar_hdf5.open_snipar_file(path) as handle:
            block = sut.read_imputed_block(
                imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                block_hdf5_indices=indices,
                block_flip_mask=np.zeros(len(indices), dtype=bool), chunk=self.new_cache())

        assert np.allclose(standard_f[indices], block["standard_f"])


    def test__blocks_within_one_chunk__read_once_and_reused(self, tmp_path, caplog):
        # Four blocks of two SNPs each, all inside the first chunk.  Only the first should cause
        # a read; the rest are the cache earning its keep.
        path, _, _, _ = self.imputed_file(tmp_path)

        with snipar_hdf5.open_snipar_file(path) as handle:
            metadata = snipar_hdf5.read_snipar_metadata(path)
            cache = self.new_cache()
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                for start in (0, 2, 4, 6):
                    indices = np.arange(start, start + 2)
                    sut.read_imputed_block(
                        imputed_hdf5=handle, metadata=metadata, block_hdf5_indices=indices,
                        block_flip_mask=np.zeros(2, dtype=bool), chunk=cache)

        assert self.reads(caplog) == [(0, self.CHUNK_WIDTH)]


    def test__block_spanning_chunks__one_chunk_held_at_a_time(self, tmp_path, caplog):
        # A block covering 20 SNPs crosses three chunks of 8.  Each is read on its own and
        # replaces the last, so the memory held stays bounded at one chunk however the block size
        # and the chunk width line up.  Reading the span in one go would be simpler and would
        # keep all three resident -- which at a real file's width is gigabytes rather than
        # kilobytes.
        path, _, _, _ = self.imputed_file(tmp_path)

        with snipar_hdf5.open_snipar_file(path) as handle:
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                sut.read_imputed_block(
                    imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                    block_hdf5_indices=np.arange(0, 20),
                    block_flip_mask=np.zeros(20, dtype=bool), chunk=self.new_cache())

        reads = self.reads(caplog)
        assert reads == [(0, 8), (8, 16), (16, 24)]
        assert all(stop - start == self.CHUNK_WIDTH for start, stop in reads)


    def test__last_chunk__read_stops_at_the_end_of_the_file(self, tmp_path, caplog):
        # 40 SNPs over chunks of 8 divides evenly, so the final chunk ends exactly at the end of
        # the file.  The read range is clamped rather than running past it.
        path, _, _, _ = self.imputed_file(tmp_path)

        with snipar_hdf5.open_snipar_file(path) as handle:
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                sut.read_imputed_block(
                    imputed_hdf5=handle, metadata=snipar_hdf5.read_snipar_metadata(path),
                    block_hdf5_indices=np.array([36, 39]),
                    block_flip_mask=np.zeros(2, dtype=bool), chunk=self.new_cache())

        assert self.reads(caplog) == [(32, 40)]


class TestDescribeOffenders:

    # A message formatter, so the tests pin what the caller relies on -- the count being right and
    # the list being capped -- rather than the punctuation, which is free to change.
    #
    # The cases below pass an explicit limit so that they describe the BEHAVIOUR rather than
    # whatever MAX_REPORTED_OFFENDERS currently happens to be; two cases at the end cover the
    # default separately.
    LIMIT = 5

    def test__fewer_than_the_limit__all_of_them_named(self):
        ids = [f"IND{index}" for index in range(self.LIMIT - 2)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith(f"{self.LIMIT - 2} individual(s):")
        assert all(identifier in described for identifier in ids)
        assert "..." not in described


    def test__exactly_at_the_limit__still_all_of_them(self):
        # The boundary, since an off-by-one here would truncate a list that fits
        ids = [f"IND{index}" for index in range(self.LIMIT)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith(f"{self.LIMIT} individual(s):")
        assert all(identifier in described for identifier in ids)
        assert "..." not in described


    def test__over_the_limit__true_count_with_a_capped_sample(self):
        # The case this function exists for.  A fam file using something other than "0" for an
        # absent parent makes every individual an offender, and naming all of them turns one
        # error message into hundreds of thousands of lines.
        #
        # WHICH ones are named is deliberately not asserted: the docstring says that is arbitrary,
        # so pinning it here would contradict the contract.
        ids = [f"IND{index:06d}" for index in range(5000)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith("5000 individual(s), including:")
        assert described.count("IND") == self.LIMIT
        assert described.endswith(", ...]")


    def test__noun__used_in_the_phrase(self):
        # Callers name what they are reporting, so that the result reads as part of a sentence
        described = sut.describe_offenders(["rs1", "rs2"], noun="SNP(s)", limit=self.LIMIT)

        assert described.startswith("2 SNP(s):")


    def test__numpy_array__reads_the_same_as_a_list(self):
        # Several callers hold their IDs as numpy arrays, whose elements print their type
        # alongside their value unless they are converted first
        ids = [f"IND{index}" for index in range(self.LIMIT - 2)]

        assert (sut.describe_offenders(np.array(ids), limit=self.LIMIT) ==
                sut.describe_offenders(ids, limit=self.LIMIT))


    def test__nothing_to_report__says_so_without_failing(self):
        # Not reachable through the validators, which only call this once they have offenders,
        # but a formatter that falls over on an empty list is a poor one
        assert sut.describe_offenders([], limit=self.LIMIT).startswith("0 individual(s):")


    def test__no_limit_given__capped_at_the_module_default(self):
        # The callers all rely on the default, so it is worth knowing it is the constant rather
        # than something the function decided for itself
        ids = [f"IND{index:06d}" for index in range(sut.MAX_REPORTED_OFFENDERS + 1)]

        described = sut.describe_offenders(ids)

        assert described.count("IND") == sut.MAX_REPORTED_OFFENDERS
        assert described.endswith(", ...]")


    def test__no_limit_given__a_list_that_fits_is_not_capped(self):
        # The other side of the same boundary, at the default
        ids = [f"IND{index:06d}" for index in range(sut.MAX_REPORTED_OFFENDERS)]

        described = sut.describe_offenders(ids)

        assert described.count("IND") == sut.MAX_REPORTED_OFFENDERS
        assert "..." not in described


class TestValidatePedigreeSanity:

    # Two conditions kill the run here.  Both mean the fam file is WRONG rather than merely
    # incomplete: an individual cannot be their own parent, and one person cannot be both of
    # somebody's parents.  Neither has a reading under which the analysis would be meaningful.

    @staticmethod
    def candidates(rows):
        """Builds the frame determine_fam_file_trios() passes in, from (iid, father, mother)"""

        return pd.DataFrame({bbf.FAM_FID_COL : ["FAM"] * len(rows),
                             bbf.FAM_IID_COL : [row[0] for row in rows],
                             bbf.FAM_IIDF_COL : [row[1] for row in rows],
                             bbf.FAM_IIDM_COL : [row[2] for row in rows]})


    def test__ordinary_pedigree__accepted(self):
        sut.validate_pedigree_sanity(self.candidates([("C1", "DAD", "MUM"),
                                                      ("C2", "DAD", "MUM"),
                                                      ("C3", "OTHER_DAD", "OTHER_MUM")]))


    def test__nobody_to_check__accepted(self):
        # determine_fam_file_trios() returns early on an empty frame, so this is unreachable
        # through it -- but a validator that falls over on nothing to validate is a poor one
        sut.validate_pedigree_sanity(self.candidates([]))


    @pytest.mark.parametrize("rows, expected_in_message", [
        pytest.param([("C1", "C1", "MUM")], "C1", id="listed_as_their_own_father"),
        pytest.param([("C1", "DAD", "C1")], "C1", id="listed_as_their_own_mother"),
    ])
    def test__individual_is_their_own_parent__raises(self, rows, expected_in_message):
        with pytest.raises(ValueError) as error:
            sut.validate_pedigree_sanity(self.candidates(rows))

        assert "own parent" in str(error.value)
        assert expected_in_message in str(error.value)


    def test__one_person_as_both_parents__raises(self):
        with pytest.raises(ValueError) as error:
            sut.validate_pedigree_sanity(self.candidates([("C1", "DAD", "MUM"),
                                                          ("C2", "SAME", "SAME")]))

        assert "both parents" in str(error.value)
        assert "C2" in str(error.value)


    def test__both_parents_absent__accepted(self):
        # PLINK's marker for "no parent here" in both columns is not one person listed twice, it
        # is nobody listed at all.  The caller filters these rows out before getting here, but
        # that is the caller's business rather than something this function should depend on.
        sut.validate_pedigree_sanity(self.candidates(
            [("C1", sut.FAM_MISSING_PARENT, sut.FAM_MISSING_PARENT)]))


    def test__several_offenders__all_of_them_reported(self):
        # The user will want to fix them in one pass, so the error names everyone it found rather
        # than stopping at the first
        with pytest.raises(ValueError) as error:
            sut.validate_pedigree_sanity(self.candidates([("C1", "SAME", "SAME"),
                                                          ("C2", "DAD", "MUM"),
                                                          ("C3", "ALSO", "ALSO")]))

        assert "C1" in str(error.value)
        assert "C3" in str(error.value)


    def test__many_offenders__the_list_is_capped(self):
        # A fam file using something other than PLINK's marker for an absent parent makes EVERY
        # individual look like one person listed twice.  Naming them all turns a single error
        # into one line per sample, so the message gives the count and a sample.
        offenders = sut.MAX_REPORTED_OFFENDERS + 20

        with pytest.raises(ValueError) as error:
            sut.validate_pedigree_sanity(self.candidates(
                [(f"C{index:04d}", "-9", "-9") for index in range(offenders)]))

        assert f"{offenders} individual(s), including:" in str(error.value)
        assert str(error.value).count("'C0") == sut.MAX_REPORTED_OFFENDERS


class TestValidateParentSexes:

    # Two same-sex genetic parents warn rather than stopping the run.  The parental mean is
    # symmetric in the two parents, so which is which changes no result -- but a pedigree or a sex
    # column wrong in that way is likely wrong in others, which the user should hear about.
    #
    # A trio where either parent's sex is unknown cannot be checked at all, and that is reported
    # too: silence would otherwise mean both "checked and fine" and "could not be checked".

    @staticmethod
    def fam(sexes):
        """Builds an unfiltered fam frame whose rows are the parents, in the given order"""

        num_samples = len(sexes)

        return pd.DataFrame({bbf.FAM_FID_COL : ["FAM"] * num_samples,
                             bbf.FAM_IID_COL : [f"P{index}" for index in range(num_samples)],
                             bbf.FAM_IIDF_COL : [sut.FAM_MISSING_PARENT] * num_samples,
                             bbf.FAM_IIDM_COL : [sut.FAM_MISSING_PARENT] * num_samples,
                             bbf.FAM_SEX_COL : list(sexes),
                             bbf.FAM_PHENO_COL : [1.0] * num_samples})

    @staticmethod
    def children(iids):
        return pd.DataFrame({bbf.FAM_IID_COL : list(iids)})

    @classmethod
    def check(cls, caplog, sexes, father_pos, mother_pos, iids=None, level=logging.INFO):
        """Runs the validator and returns everything it logged"""

        iids = iids if iids is not None else [f"C{index}" for index in range(len(father_pos))]

        caplog.clear()
        with caplog.at_level(level):
            # dtype is given explicitly because an empty list would otherwise come out as
            # float64, which cannot be used as an index.  The positions production passes come
            # from Index.get_indexer(), which is integer-typed however few there are.
            sut.validate_parent_sexes(cls.fam(sexes), cls.children(iids),
                                      np.array(father_pos, dtype=np.int64),
                                      np.array(mother_pos, dtype=np.int64))

        return "\n".join(record.message for record in caplog.records)


    # Which parent is which is not something GRMA knows or needs to: the fam file's father column
    # may hold either, and the result is the same regardless.  What matters is only that the two
    # differ.
    @pytest.mark.parametrize("sexes", [
        pytest.param([sut.FAM_MALE_SEX, sut.FAM_FEMALE_SEX], id="father_male_mother_female"),
        pytest.param([sut.FAM_FEMALE_SEX, sut.FAM_MALE_SEX], id="father_female_mother_male"),
    ])
    def test__parents_of_different_sexes__nothing_said(self, caplog, sexes):
        assert self.check(caplog, sexes, father_pos=[0], mother_pos=[1]) == ""


    @pytest.mark.parametrize("sex", [
        pytest.param(sut.FAM_MALE_SEX, id="both_male"),
        pytest.param(sut.FAM_FEMALE_SEX, id="both_female"),
    ])
    def test__parents_of_the_same_sex__warns(self, caplog, sex):
        logged = self.check(caplog, [sex, sex], father_pos=[0], mother_pos=[1], iids=["KID"])

        assert "same recorded sex" in logged
        assert "KID" in logged


    # Anything outside the two known codes means the sex was not recorded.  PLINK writes 0, but
    # -9 turns up as a missing marker too, and neither can be checked.
    @pytest.mark.parametrize("unknown", [
        pytest.param(sut.FAM_UNKNOWN_SEX, id="plink_unknown_code"),
        pytest.param("-9", id="another_missing_marker"),
    ])
    def test__a_parent_whose_sex_is_unrecorded__reported_as_unchecked(self, caplog, unknown):
        logged = self.check(caplog, [sut.FAM_MALE_SEX, unknown],
                            father_pos=[0], mother_pos=[1], iids=["KID"])

        assert "could not be checked" in logged
        assert "KID" in logged
        assert "same recorded sex" not in logged


    def test__a_parent_who_is_not_genotyped__nothing_said(self, caplog):
        # -1 means that parent is not in the fam file at all, so there is no sex to compare and
        # nothing was skipped either -- this is not the same as a sex that is simply unrecorded.
        #
        # The LAST sample is given the same sex as the father deliberately.  A -1 used as an index
        # rather than as a sentinel reads the last row, so without that the two sexes would differ
        # by luck and this would pass whether the sentinel was honoured or not.
        sexes = [sut.FAM_MALE_SEX, sut.FAM_FEMALE_SEX, sut.FAM_MALE_SEX]

        assert self.check(caplog, sexes, father_pos=[0], mother_pos=[-1]) == ""


    def test__several_offenders__all_of_them_named(self, caplog):
        # Three trios, the first and last with same-sex parents
        sexes = [sut.FAM_MALE_SEX, sut.FAM_MALE_SEX,
                 sut.FAM_MALE_SEX, sut.FAM_FEMALE_SEX,
                 sut.FAM_FEMALE_SEX, sut.FAM_FEMALE_SEX]

        logged = self.check(caplog, sexes, father_pos=[0, 2, 4], mother_pos=[1, 3, 5],
                            iids=["KID_A", "KID_OK", "KID_B"])

        assert "KID_A" in logged
        assert "KID_B" in logged
        assert "KID_OK" not in logged


    def test__many_offenders__the_list_is_capped(self, caplog):
        # A sex column filled in wrongly across the whole file would otherwise name every trio
        offenders = sut.MAX_REPORTED_OFFENDERS + 20
        sexes = [sut.FAM_MALE_SEX] * (2 * offenders)

        logged = self.check(caplog, sexes,
                            father_pos=list(range(0, 2 * offenders, 2)),
                            mother_pos=list(range(1, 2 * offenders, 2)),
                            iids=[f"C{index:04d}" for index in range(offenders)])

        assert f"{offenders} individual(s), including:" in logged
        assert logged.count("'C0") == sut.MAX_REPORTED_OFFENDERS


    def test__nobody_to_check__nothing_said(self, caplog):
        assert self.check(caplog, [sut.FAM_MALE_SEX, sut.FAM_FEMALE_SEX],
                          father_pos=[], mother_pos=[], iids=[]) == ""


class TestValidateParentsAgainstKing:

    # A stated parent-offspring relationship that KING does not confirm kills the run.  The fam
    # file and snipar's pedigree are potentially independent of KING -- a user may have been given
    # one and produced the other -- so agreement between them is real evidence rather than a
    # restatement, and disagreement means at least one of the two is wrong about this sample.
    #
    # A pair KING does not mention at all counts as contradiction too: KING reports the
    # relationships it detects, so the absence of a parent-offspring pair means it did not detect
    # one.

    @staticmethod
    def king(pairs, degree=None):
        """Builds a relatedness frame from (index1, index2) pairs, parent-offspring by default"""

        degree = degree if degree is not None else sut.DEG_PARENT_OFFSPRING

        return pd.DataFrame({sut.FAM_INDEX_1 : [pair[0] for pair in pairs],
                             sut.FAM_INDEX_2 : [pair[1] for pair in pairs],
                             sut.KING_REL_COL : [degree] * len(pairs)})

    @staticmethod
    def validate(rel_df, child_pos, father_pos, mother_pos, usable=None, iids=None,
                 source="The fam file"):
        """Runs the validator over positions given as plain lists"""

        count = len(child_pos)
        iids = iids if iids is not None else [f"C{index}" for index in range(count)]
        usable = usable if usable is not None else [True] * count

        sut.validate_parents_against_king(
            iids=np.array(iids), child_pos=np.array(child_pos, dtype=np.int64),
            father_pos=np.array(father_pos, dtype=np.int64),
            mother_pos=np.array(mother_pos, dtype=np.int64),
            usable=np.array(usable, dtype=bool), rel_df=rel_df, source=source)


    def test__every_relationship_confirmed__accepted(self):
        # Child 0 with parents 1 and 2, both reported by KING as parent-offspring
        self.validate(self.king([(0, 1), (0, 2)]),
                      child_pos=[0], father_pos=[1], mother_pos=[2])


    def test__pairs_listed_the_other_way_round__accepted(self):
        # KING writes each pair once, in whichever order it happened to consider them, so the
        # lookup is keyed both ways.  Requiring the stated direction would reject half of a
        # perfectly good file, and which half would depend on how KING happened to sort its input.
        self.validate(self.king([(1, 0), (2, 0)]),
                      child_pos=[0], father_pos=[1], mother_pos=[2])


    @pytest.mark.parametrize("degree", [
        pytest.param(sut.DEG_FULLSIB, id="king_says_full_siblings"),
        pytest.param(sut.DEG_2ND, id="king_says_second_degree"),
        pytest.param(sut.DEG_UNRELATED, id="king_says_unrelated"),
    ])
    def test__king_classifies_the_pair_as_something_else__raises(self, degree):
        with pytest.raises(ValueError) as error:
            self.validate(self.king([(0, 1), (0, 2)], degree=degree),
                          child_pos=[0], father_pos=[1], mother_pos=[2], iids=["KID"])

        assert "KID" in str(error.value)


    def test__king_does_not_mention_the_pair__raises(self):
        # The mother is confirmed; the father appears nowhere in the KING data.  Absence is a
        # contradiction rather than a gap, since KING would have reported the pair had it found
        # one.
        with pytest.raises(ValueError) as error:
            self.validate(self.king([(0, 2)]),
                          child_pos=[0], father_pos=[1], mother_pos=[2], iids=["KID"])

        assert "KID (father)" in str(error.value)
        assert "mother" not in str(error.value)


    def test__a_parent_who_is_not_genotyped__nothing_to_confirm(self):
        # -1 means that parent is not a sample, so the fam file has asserted nothing about them
        # that KING could either confirm or contradict.  The mother is confirmed and the father
        # is simply absent.
        self.validate(self.king([(0, 2)]),
                      child_pos=[0], father_pos=[-1], mother_pos=[2])


    def test__individuals_who_will_not_be_used__not_checked(self):
        # Both sources describe people beyond those contributing to this analysis.  Failing over a
        # pedigree error involving somebody whose genotypes affect no result would be unhelpful,
        # so only the relationships that will actually be used are checked.
        #
        # Child 0 is contradicted but unusable; child 3 is fine.  Nothing is raised.
        self.validate(self.king([(3, 4)]),
                      child_pos=[0, 3], father_pos=[1, 4], mother_pos=[-1, -1],
                      usable=[False, True])


    def test__several_contradictions__all_of_them_named_with_their_roles(self):
        # The user will want to fix them in one pass, and which parent of which child is what
        # they need to know -- an individual named without the role leaves them checking both.
        with pytest.raises(ValueError) as error:
            self.validate(self.king([(0, 1)]),
                          child_pos=[0, 3], father_pos=[1, 4], mother_pos=[2, 5],
                          iids=["KID_A", "KID_B"])

        message = str(error.value)
        assert "KID_A (mother)" in message
        assert "KID_B (father)" in message
        assert "KID_B (mother)" in message
        assert "KID_A (father)" not in message


    def test__many_contradictions__the_list_is_capped(self):
        # A KING file covering a different sample set contradicts every stated relationship at
        # once, which would otherwise name two entries per trio for the whole cohort.
        trios = sut.MAX_REPORTED_OFFENDERS + 20

        with pytest.raises(ValueError) as error:
            self.validate(self.king([]),
                          child_pos=list(range(trios)),
                          father_pos=[trios + index for index in range(trios)],
                          mother_pos=[-1] * trios,
                          iids=[f"C{index:04d}" for index in range(trios)])

        assert f"{trios} relationship(s), including:" in str(error.value)
        assert str(error.value).count("(father)") == sut.MAX_REPORTED_OFFENDERS


    def test__nobody_to_check__accepted(self):
        self.validate(self.king([]), child_pos=[], father_pos=[], mother_pos=[], iids=[])


    def test__the_source_is_named__so_the_user_knows_which_file_to_fix(self):
        # Both parental processing branches call this, and they are told apart only by what they
        # pass here.  A message blaming the fam file for snipar's pedigree would send the user to
        # the wrong place entirely.
        with pytest.raises(ValueError) as error:
            self.validate(self.king([]), child_pos=[0], father_pos=[1], mother_pos=[-1],
                          source="the imputed parental genotype file's pedigree")

        assert str(error.value).startswith("the imputed parental genotype file's pedigree")


class TestMapSamplesToImputedFamilies:

    # Three lookups: the child and each parent into the fam file, and the family into the imputed
    # file's own list.  All three are keyed by value rather than by position, which matters --
    # snipar writes families sorted while the pedigree keeps whatever order it was built in.

    @staticmethod
    def pedigree(records):
        """Builds pedigree rows from (fid, iid, father, mother), with the header already dropped"""

        return np.array([[record[0], record[1], record[2], record[3], "False", "False"]
                         for record in records])

    @staticmethod
    def fam(samples):
        """Builds the unfiltered fam ID columns from (fid, iid) pairs"""

        return pd.DataFrame({bbf.FAM_FID_COL : [sample[0] for sample in samples],
                             bbf.FAM_IID_COL : [sample[1] for sample in samples]})


    def test__everything_resolves__positions_line_up_with_the_pedigree(self):
        # Two families of three, and the fam file in a different order from the pedigree, so that
        # positions have to come from the lookup rather than from the row number
        full_fam_ids = self.fam([("F2", "DAD2"), ("F1", "KID1"), ("F2", "KID2"),
                                 ("F1", "DAD1"), ("F2", "MUM2"), ("F1", "MUM1")])
        pedigree = self.pedigree([("F1", "KID1", "DAD1", "MUM1"),
                                  ("F2", "KID2", "DAD2", "MUM2")])

        child_pos, parent_pos, imputed_rows = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1", "F2"]), full_fam_ids=full_fam_ids)

        assert list(child_pos) == [1, 2]
        assert parent_pos.tolist() == [[3, 5], [0, 4]]
        assert list(imputed_rows) == [0, 1]


    def test__families_sorted_but_the_pedigree_is_not__matched_by_value(self):
        # snipar writes families lexicographically, and the pedigree in the order it built them.
        # Taking the imputed row from the pedigree's position rather than by looking the family up
        # would hand every individual another family's parental genotypes -- plausible numbers
        # attached to the wrong people.
        full_fam_ids = self.fam([("ZZZ", "KID_Z"), ("AAA", "KID_A")])
        pedigree = self.pedigree([("ZZZ", "KID_Z", "DAD_Z", "MUM_Z"),
                                  ("AAA", "KID_A", "DAD_A", "MUM_A")])

        _, _, imputed_rows = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["AAA", "ZZZ"]), full_fam_ids=full_fam_ids)

        assert list(imputed_rows) == [1, 0]


    def test__child_not_in_the_fam_file__no_position(self):
        # The pedigree covers everyone snipar saw, which can exceed the bfile being analysed now
        full_fam_ids = self.fam([("F1", "DAD1"), ("F1", "MUM1")])
        pedigree = self.pedigree([("F1", "ABSENT", "DAD1", "MUM1")])

        child_pos, parent_pos, _ = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1"]), full_fam_ids=full_fam_ids)

        assert list(child_pos) == [-1]
        assert parent_pos.tolist() == [[0, 1]]


    @pytest.mark.parametrize("father, mother, expected", [
        pytest.param("DAD1", "ABSENT", [0, -1], id="mother_not_genotyped"),
        pytest.param("ABSENT", "MUM1", [-1, 1], id="father_not_genotyped"),
        pytest.param("ABSENT", "ALSO_ABSENT", [-1, -1], id="neither_genotyped"),
    ])
    def test__a_parent_not_in_the_fam_file__no_position_for_that_one(self, father, mother,
                                                                     expected):
        # One parent genotyped and the other not is the commonest configuration in real data, and
        # the two columns have to be resolved independently for it to come out right
        full_fam_ids = self.fam([("F1", "DAD1"), ("F1", "MUM1"), ("F1", "KID1")])
        pedigree = self.pedigree([("F1", "KID1", father, mother)])

        child_pos, parent_pos, _ = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1"]), full_fam_ids=full_fam_ids)

        assert list(child_pos) == [2]
        assert parent_pos.tolist() == [expected]


    def test__family_absent_from_the_imputed_file__no_imputed_row(self):
        # A family whose parents were both genotyped had nothing to impute, so snipar wrote no row
        # for it.  The individuals still resolve: their parental genotypes come from the bed file.
        full_fam_ids = self.fam([("F1", "KID1"), ("F1", "DAD1"), ("F1", "MUM1")])
        pedigree = self.pedigree([("F1", "KID1", "DAD1", "MUM1")])

        child_pos, parent_pos, imputed_rows = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F2", "F3"]), full_fam_ids=full_fam_ids)

        assert list(imputed_rows) == [-1]
        assert list(child_pos) == [0]
        assert parent_pos.tolist() == [[1, 2]]


    def test__siblings__share_a_family_row_but_not_a_position(self):
        # One imputed row serves a whole sibship, since snipar imputes the parents FROM the
        # siblings.  Nothing deduplicates, and nothing should.
        full_fam_ids = self.fam([("F1", "KID1"), ("F1", "KID2"), ("F1", "KID3"),
                                 ("F1", "DAD1"), ("F1", "MUM1")])
        pedigree = self.pedigree([("F1", "KID1", "DAD1", "MUM1"),
                                  ("F1", "KID2", "DAD1", "MUM1"),
                                  ("F1", "KID3", "DAD1", "MUM1")])

        child_pos, parent_pos, imputed_rows = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1"]), full_fam_ids=full_fam_ids)

        assert list(child_pos) == [0, 1, 2]
        assert list(imputed_rows) == [0, 0, 0]
        assert parent_pos.tolist() == [[3, 4]] * 3


    def test__repeated_iid_the_pedigree_never_mentions__resolves_normally(self):
        # Matching is on IID alone because that is what snipar records, so a repeated IID is
        # ambiguous.  But two unrelated samples sharing one under different FIDs have nothing to
        # do with the families snipar imputed for, and stopping the run over them would reject
        # a file that is perfectly usable.
        full_fam_ids = self.fam([("F1", "KID1"), ("F1", "DAD1"), ("F1", "MUM1"),
                                 ("F8", "SHARED"), ("F9", "SHARED")])
        pedigree = self.pedigree([("F1", "KID1", "DAD1", "MUM1")])

        child_pos, parent_pos, _ = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1"]), full_fam_ids=full_fam_ids)

        assert list(child_pos) == [0]
        assert parent_pos.tolist() == [[1, 2]]


    @pytest.mark.parametrize("records", [
        pytest.param([("F1", "SHARED", "DAD1", "MUM1")], id="as_a_child"),
        pytest.param([("F1", "KID1", "SHARED", "MUM1")], id="as_a_father"),
        pytest.param([("F1", "KID1", "DAD1", "SHARED")], id="as_a_mother"),
    ])
    def test__repeated_iid_the_pedigree_refers_to__raises(self, records):
        # Here it genuinely cannot be resolved: two samples answer to the name, and nothing
        # distinguishes them.  Silently picking one would attach somebody else's genotypes.
        full_fam_ids = self.fam([("F1", "KID1"), ("F1", "DAD1"), ("F1", "MUM1"),
                                 ("F8", "SHARED"), ("F9", "SHARED")])

        with pytest.raises(ValueError) as error:
            sut.map_samples_to_imputed_families(
                pedigree=self.pedigree(records), families=np.array(["F1"]),
                full_fam_ids=full_fam_ids)

        assert "SHARED" in str(error.value)
        assert "IID alone" in str(error.value)


    def test__returned_shapes__one_entry_per_pedigree_record(self):
        # Sized by the PEDIGREE rather than by the fam file or the family list, all three of which
        # differ here so that a confusion between them cannot pass
        full_fam_ids = self.fam([("F1", "KID1"), ("F1", "DAD1"), ("F1", "MUM1"),
                                 ("F2", "KID2"), ("F2", "DAD2")])
        pedigree = self.pedigree([("F1", "KID1", "DAD1", "MUM1"),
                                  ("F2", "KID2", "DAD2", "MUM2")])

        child_pos, parent_pos, imputed_rows = sut.map_samples_to_imputed_families(
            pedigree=pedigree, families=np.array(["F1", "F2", "F3"]), full_fam_ids=full_fam_ids)

        assert child_pos.shape == (len(pedigree),)
        assert parent_pos.shape == (len(pedigree), 2)
        assert imputed_rows.shape == (len(pedigree),)



class TestDetermineFamFileTrios:

    # Fam file trio mode.  The ORIGINAL fam file is read rather than the filtered frame, because
    # the parental ID columns do not survive process_phenotypes() and because a parent dropped for
    # a missing phenotype still has usable genotypes.  An individual must be in the analysis
    # sample to be affected; their parents need not be.

    @staticmethod
    def fam(rows):
        """Builds an unfiltered fam frame from (fid, iid, father, mother) tuples"""

        return pd.DataFrame({bbf.FAM_FID_COL : [row[0] for row in rows],
                             bbf.FAM_IID_COL : [row[1] for row in rows],
                             bbf.FAM_IIDF_COL : [row[2] for row in rows],
                             bbf.FAM_IIDM_COL : [row[3] for row in rows],
                             bbf.FAM_SEX_COL : [sut.FAM_UNKNOWN_SEX] * len(rows),
                             bbf.FAM_PHENO_COL : [1.0] * len(rows)})

    @staticmethod
    def analysis_sample(full_fam, positions=None):
        """The filtered frame process_phenotypes() returns, holding the given rows"""

        positions = positions if positions is not None else range(len(full_fam))

        return full_fam.iloc[list(positions)][[bbf.FAM_FID_COL, bbf.FAM_IID_COL]]

    @staticmethod
    def king(pairs):
        """KING data confirming each (child position, parent position) pair as parent-offspring"""

        return pd.DataFrame({sut.FAM_INDEX_1 : [pair[0] for pair in pairs],
                             sut.FAM_INDEX_2 : [pair[1] for pair in pairs],
                             sut.KING_REL_COL : [sut.DEG_PARENT_OFFSPRING] * len(pairs)})


    def test__a_full_trio__affected_with_both_parent_positions(self, caplog):
        # Rows 0 and 1 are the parents, row 2 the child.  Nothing is imputed, so the three imputed
        # fields come back as None -- which is what tells everything downstream it is in fam file
        # trio mode rather than imputed parent mode.
        full_fam = self.fam([("F1", "DAD", "0", "0"),
                             ("F1", "MUM", "0", "0"),
                             ("F1", "KID", "DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]
        assert actual.imputed_rows is None
        assert actual.imputed_file is None
        assert actual.snipar_metadata is None

        logged = "\n".join(record.message for record in caplog.records)
        assert "1 individual(s) list both parents" in logged
        assert "0 not usable as trios" in logged
        assert "1 full trio(s) validated against KING" in logged


    def test__nobody_lists_both_parents__nothing_affected(self, caplog):
        # The ordinary case for a cohort with no pedigree in its fam file.  The early return fires
        # before any validation runs, so a file with no parental columns filled in costs nothing.
        full_fam = self.fam([("F1", "A", "0", "0"), ("F1", "B", "0", "0")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam), rel_df=self.king([]))

        assert len(actual.affected_samples) == 0
        logged = "\n".join(record.message for record in caplog.records)
        assert "0 individual(s) list both parents" in logged


    def test__only_one_parent_listed__never_a_candidate(self, caplog):
        # Half a trio is not a trio.  Residualizing against one parent needs the transmitted
        # allele of the other, which only an imputed parental genotype file supplies -- so in this
        # mode the individual simply falls back to ordinary within-group residualization.
        full_fam = self.fam([("F1", "DAD", "0", "0"),
                             ("F1", "KID", "DAD", "0")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(1, 0)]))

        assert len(actual.affected_samples) == 0
        logged = "\n".join(record.message for record in caplog.records)
        assert "0 individual(s) list both parents" in logged


    def test__a_listed_parent_is_not_in_the_fam_file__not_a_trio(self, caplog):
        # The father is named but was never genotyped, so there is nothing to difference against.
        # This is reported separately from the candidates count, since the two mean different
        # things to somebody wondering where their trios went.
        full_fam = self.fam([("F1", "MUM", "0", "0"),
                             ("F1", "KID", "ABSENT_DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(1, 0)]))

        assert len(actual.affected_samples) == 0

        logged = "\n".join(record.message for record in caplog.records)
        assert "1 individual(s) list both parents" in logged
        assert "1 not usable as trios" in logged
        assert "0 full trio(s) validated against KING" in logged


    def test__child_not_in_the_analysis_sample__not_affected(self, caplog):
        # The child was dropped for a missing phenotype, so they contribute to no result and
        # there is nothing to residualize.  Their parents being present changes nothing.
        full_fam = self.fam([("F1", "DAD", "0", "0"),
                             ("F1", "MUM", "0", "0"),
                             ("F1", "KID", "DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam, positions=[0, 1]),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert len(actual.affected_samples) == 0
        logged = "\n".join(record.message for record in caplog.records)
        assert "0 full trio(s) validated against KING" in logged


    def test__a_parent_not_in_the_analysis_sample__still_used(self, caplog):
        # The reason the unfiltered fam file is read at all.  The father was dropped for a missing
        # phenotype, but his genotypes are in the bed file and are exactly what the child needs.
        full_fam = self.fam([("F1", "DAD", "0", "0"),
                             ("F1", "MUM", "0", "0"),
                             ("F1", "KID", "DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam, positions=[1, 2]),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]
        logged = "\n".join(record.message for record in caplog.records)
        assert "1 full trio(s) validated against KING" in logged


    def test__parental_ids_are_within_family__resolved_against_the_row_s_own_fid(self, caplog):
        # PLINK's parental columns hold IIDs that are unique within a family rather than across
        # the file, so the key is (this row's FID, the parental ID).  Two families reusing the
        # same parental IDs is ordinary, and each child must resolve to its OWN parents.
        full_fam = self.fam([("F1", "DAD", "0", "0"), ("F1", "MUM", "0", "0"),
                             ("F2", "DAD", "0", "0"), ("F2", "MUM", "0", "0"),
                             ("F1", "KID", "DAD", "MUM"), ("F2", "KID", "DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(4, 0), (4, 1), (5, 2), (5, 3)]))

        assert list(actual.affected_samples) == [4, 5]
        assert actual.observed_parent_cols.tolist() == [[0, 1], [2, 3]]


    def test__trios_mixed_with_non_trios__only_the_usable_ones_returned(self, caplog):
        # Two candidates, of which one has an ungenotyped parent.  What comes back has to be the
        # usable one, in its own position -- not the first candidate, and not both.
        full_fam = self.fam([("F1", "DAD", "0", "0"), ("F1", "MUM", "0", "0"),
                             ("F1", "KID_BAD", "GHOST", "MUM"),
                             ("F1", "KID_OK", "DAD", "MUM")])

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(3, 0), (3, 1), (2, 1)]))

        assert list(actual.affected_samples) == [3]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]

        logged = "\n".join(record.message for record in caplog.records)
        assert "2 individual(s) list both parents" in logged
        assert "1 not usable as trios" in logged
        assert "1 full trio(s) validated against KING" in logged


    def test__king_contradicts_a_stated_parent__raises(self):
        # The validator is wired in here and given the usable mask, so a contradiction about
        # somebody who will actually be used stops the run
        full_fam = self.fam([("F1", "DAD", "0", "0"),
                             ("F1", "MUM", "0", "0"),
                             ("F1", "KID", "DAD", "MUM")])

        with pytest.raises(ValueError) as error:
            sut.determine_fam_file_trios(fam_file=full_fam,
                                         fam_df=self.analysis_sample(full_fam),
                                         rel_df=self.king([(2, 0)]))

        assert "KID (mother)" in str(error.value)


    def test__numeric_ids__not_mistaken_for_numbers(self, caplog):
        # A fam file whose IDs happen to be purely numeric is read as int64 unless something stops
        # it, and then 0 == "0" is False, so every row looks as though it lists both parents.
        # Passing a frame straight in skips the read entirely, which is why the conversion cannot
        # live only in the read parameters.
        full_fam = pd.DataFrame({bbf.FAM_FID_COL : [1, 1, 1],
                                 bbf.FAM_IID_COL : [10, 11, 12],
                                 bbf.FAM_IIDF_COL : [0, 0, 10],
                                 bbf.FAM_IIDM_COL : [0, 0, 11],
                                 bbf.FAM_SEX_COL : [0, 0, 0],
                                 bbf.FAM_PHENO_COL : [1.0, 1.0, 1.0]})

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=full_fam, fam_df=self.analysis_sample(full_fam),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        logged = "\n".join(record.message for record in caplog.records)
        assert "1 individual(s) list both parents" in logged


    def test__read_from_an_actual_fam_file__same_as_from_a_frame(self, tmp_path, caplog):
        # Everything else here hands the function a frame, which skips the read.  This is the only
        # case that exercises the file path, where the IDs arrive as text and the dtype given to
        # read_csv is what keeps them that way.
        path = str(tmp_path / "trio.fam")
        bbf.write_fam_file(fam_filename=path,
                           fid=np.array(["1", "1", "1"]),
                           iid=np.array(["10", "11", "12"]),
                           iidf=np.array(["0", "0", "10"]),
                           iidm=np.array(["0", "0", "11"]),
                           sex=np.array(["0", "0", "0"]),
                           pheno=np.array([1.0, 1.0, 1.0]))

        from_file, _ = sut.get_df(path, "fam_file", {"names" : bbf.FAM_COLS, "dtype" : str})

        with caplog.at_level(logging.INFO):
            actual = sut.determine_fam_file_trios(
                fam_file=path, fam_df=self.analysis_sample(from_file),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]


class TestProcessImputedParents:

    # Imputed parent mode.  The relationships come from snipar's own pedigree rather than from the
    # fam file's parental columns, which are not consulted at all here.
    #
    # Three populations appear in the file, and the third is easy to miss: the obvious reading of
    # "absent from families" is that both parents were genotyped and there was nothing to impute,
    # which holds for some families but not all -- snipar also leaves out the families it could
    # not impute for.

    NUM_SNPS = 2

    @classmethod
    def imputed_file(cls, tmp_path, pedigree_records, families):
        """Writes an imputed parental genotype file holding the given pedigree and families

        Only the pedigree and the family list matter here; the per-SNP arrays are filler, since
        nothing this function does reads them.
        """

        path = str(tmp_path / "chr_1.hdf5")
        num_families = len(families)

        snipar_hdf5.write_snipar_file(
            imputed_filename=path,
            imputed_gts=np.zeros((num_families, cls.NUM_SNPS)),
            num_obs_par_alleles=np.full((num_families, cls.NUM_SNPS), 4.0),
            standard_f=np.full(cls.NUM_SNPS, 0.25),
            parental_status=np.zeros((num_families, 3), dtype=bool),
            pedigree=np.array([[record[0], record[1], record[2], record[3], "False", "False"]
                               for record in pedigree_records]),
            families=np.array(families),
            bim_values=np.array([["1", f"rs{snp}", "0", str(snp * 100), "A", "G"]
                                 for snp in range(cls.NUM_SNPS)]))

        return path

    @staticmethod
    def fam(rows):
        """Builds an unfiltered fam frame from (fid, iid) pairs, with no parental IDs at all"""

        return pd.DataFrame({bbf.FAM_FID_COL : [row[0] for row in rows],
                             bbf.FAM_IID_COL : [row[1] for row in rows],
                             bbf.FAM_IIDF_COL : [sut.FAM_MISSING_PARENT] * len(rows),
                             bbf.FAM_IIDM_COL : [sut.FAM_MISSING_PARENT] * len(rows),
                             bbf.FAM_SEX_COL : [sut.FAM_UNKNOWN_SEX] * len(rows),
                             bbf.FAM_PHENO_COL : [1.0] * len(rows)})

    @staticmethod
    def analysis_sample(full_fam, positions=None):
        positions = positions if positions is not None else range(len(full_fam))

        return full_fam.iloc[list(positions)][[bbf.FAM_FID_COL, bbf.FAM_IID_COL]]

    @staticmethod
    def king(pairs):
        return pd.DataFrame({sut.FAM_INDEX_1 : [pair[0] for pair in pairs],
                             sut.FAM_INDEX_2 : [pair[1] for pair in pairs],
                             sut.KING_REL_COL : [sut.DEG_PARENT_OFFSPRING] * len(pairs)})


    def test__family_with_an_imputed_row__usable_with_any_genotyped_parent(self, tmp_path,
                                                                           caplog):
        # The first population.  The mother was genotyped and the father was not, so snipar
        # imputed the father and GRMA will use both: the mother from the bed file, the father
        # from the imputed row.
        full_fam = self.fam([("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "GHOST_DAD", "MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(1, 0)]))

        assert list(actual.affected_samples) == [1]
        assert actual.observed_parent_cols.tolist() == [[-1, 0]]
        assert list(actual.imputed_rows) == [0]
        assert actual.imputed_file == path


    def test__no_imputed_row_but_both_parents_genotyped__usable_from_the_bed_file(self, tmp_path,
                                                                                  caplog):
        # The second population.  Nothing needed imputing, so snipar wrote no row for this family
        # -- but both parents are in the bed file, which is all the differencing needs.
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "DAD", "MUM")], ["F_OTHER"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]
        assert list(actual.imputed_rows) == [-1]


    def test__no_imputed_row_and_no_genotyped_parent__not_usable(self, tmp_path, caplog):
        # The third population, and the one worth stating outright: a family can be missing from
        # the families list because snipar could not impute for it, not because it had no need.
        # There is then nothing to difference against, and ordinary within-group residualization
        # applies instead.
        full_fam = self.fam([("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "GHOST_DAD", "GHOST_MUM")],
                                 ["F_OTHER"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([]))

        assert len(actual.affected_samples) == 0

        logged = "\n".join(record.message for record in caplog.records)
        assert "1 pedigree record(s)" in logged
        assert "1 not usable (no imputed row and no genotyped parent)" in logged
        assert "0 individual(s) to residualize" in logged


    def test__fam_file_parental_columns__not_consulted(self, tmp_path, caplog):
        # The relationships come from snipar's pedigree.  Here the fam file names the WRONG
        # parents, and the result is the same as if it named none -- which is what the fixtures
        # elsewhere in this class rely on.
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM"), ("F1", "KID"), ("F1", "STRANGER")])
        full_fam.loc[2, bbf.FAM_IIDF_COL] = "STRANGER"
        full_fam.loc[2, bbf.FAM_IIDM_COL] = "STRANGER"

        path = self.imputed_file(tmp_path, [("F1", "KID", "DAD", "MUM")], ["F_OTHER"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(2, 0), (2, 1)]))

        assert actual.observed_parent_cols.tolist() == [[0, 1]]


    def test__child_not_in_the_fam_file__not_usable(self, tmp_path, caplog):
        # snipar's pedigree covers everyone it saw, which can exceed the bfile being analysed now
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM")])
        path = self.imputed_file(tmp_path, [("F1", "ABSENT_KID", "DAD", "MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([]))

        assert len(actual.affected_samples) == 0


    def test__child_not_in_the_analysis_sample__not_usable(self, tmp_path, caplog):
        # Dropped for a missing phenotype, so they contribute to no result
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "DAD", "MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam, positions=[0, 1]),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert len(actual.affected_samples) == 0


    def test__a_genotyped_parent_outside_the_analysis_sample__still_used(self, tmp_path, caplog):
        # The father was dropped for a missing phenotype, but his genotypes are in the bed file
        # and are what the child is differenced against
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "DAD", "MUM")], ["F_OTHER"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam, positions=[1, 2]),
                rel_df=self.king([(2, 0), (2, 1)]))

        assert list(actual.affected_samples) == [2]
        assert actual.observed_parent_cols.tolist() == [[0, 1]]


    def test__siblings__share_an_imputed_row(self, tmp_path, caplog):
        # snipar imputes the parents FROM the sibship, so one row serves all of them
        full_fam = self.fam([("F1", "KID_A"), ("F1", "KID_B")])
        path = self.imputed_file(tmp_path, [("F1", "KID_A", "GHOST_DAD", "GHOST_MUM"),
                                            ("F1", "KID_B", "GHOST_DAD", "GHOST_MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([]))

        assert list(actual.affected_samples) == [0, 1]
        assert list(actual.imputed_rows) == [0, 0]
        assert actual.observed_parent_cols.tolist() == [[-1, -1], [-1, -1]]


    def test__king_contradicts_the_pedigree__raises_naming_the_imputed_file(self, tmp_path):
        # The same validator the fam file branch uses, but told where the claim came from.  A
        # message blaming the fam file here would send the user to a file that says nothing about
        # these relationships at all.
        full_fam = self.fam([("F1", "DAD"), ("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "DAD", "MUM")], ["F_OTHER"])

        with pytest.raises(ValueError) as error:
            sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(2, 0)]))

        assert str(error.value).startswith("the imputed parental genotype file's pedigree")
        assert "KID (mother)" in str(error.value)


    def test__nobody_usable__empty_sources(self, tmp_path, caplog):
        # Every pedigree record describes somebody outside the analysis, so there is nothing to do
        full_fam = self.fam([("F1", "SOMEONE_ELSE")])
        path = self.imputed_file(tmp_path, [("F1", "ABSENT_KID", "DAD", "MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([]))

        assert len(actual.affected_samples) == 0
        assert actual.observed_parent_cols.shape == (0, 2)


    def test__metadata__returned_whole_for_the_steps_that_follow(self, tmp_path, caplog):
        # The SNP alignment reads bim_values and bim_columns from this, and the chunk cache reads
        # standard_f, so what comes back has to be the whole thing rather than the parts used here
        full_fam = self.fam([("F1", "MUM"), ("F1", "KID")])
        path = self.imputed_file(tmp_path, [("F1", "KID", "GHOST_DAD", "MUM")], ["F1"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(1, 0)]))

        for key in (snipar_hdf5.SNIPAR_PEDIGREE_KEY, snipar_hdf5.SNIPAR_FAMILIES_KEY,
                    snipar_hdf5.SNIPAR_BIM_VALUES_KEY, snipar_hdf5.SNIPAR_BIM_COLUMNS_KEY,
                    snipar_hdf5.SNIPAR_STANDARD_F_KEY, snipar_hdf5.SNIPAR_PARENTAL_STATUS_KEY):
            assert key in actual.snipar_metadata

        assert len(actual.snipar_metadata[snipar_hdf5.SNIPAR_STANDARD_F_KEY]) == self.NUM_SNPS

    def test__usable_mixed_with_unusable__only_the_usable_ones_returned(self, tmp_path, caplog):
        # Two pedigree records, of which only the second has anything to be residualized against.
        # What comes back has to be the second one, in its own position -- not the first, and not
        # both.  Without a case mixing them, a result that ignored the usable mask entirely would
        # look right: every other case here has everybody usable or nobody, and nobody returns
        # early before the mask is applied.
        full_fam = self.fam([("F1", "LONE_KID"), ("F2", "DAD"), ("F2", "MUM"), ("F2", "KID")])
        path = self.imputed_file(
            tmp_path,
            [("F1", "LONE_KID", "GHOST_DAD", "GHOST_MUM"), ("F2", "KID", "DAD", "MUM")],
            ["F_OTHER"])

        with caplog.at_level(logging.INFO):
            actual = sut.process_imputed_parents(
                imputed_parental_file=path, fam_file=full_fam,
                fam_df=self.analysis_sample(full_fam), rel_df=self.king([(3, 1), (3, 2)]))

        assert list(actual.affected_samples) == [3]
        assert actual.observed_parent_cols.tolist() == [[1, 2]]
        assert list(actual.imputed_rows) == [-1]

        logged = "\n".join(record.message for record in caplog.records)
        assert "2 pedigree record(s)" in logged
        assert "1 not usable (no imputed row and no genotyped parent)" in logged
        assert "1 individual(s) to residualize" in logged


class TestCalculateR:

    # Each sample's row centers them within their CLOSEST relatedness group: the minimum degree
    # they appear at, and every relative at that degree.  Multiplying by R subtracts that group's
    # mean, which is what removes the family-level confounding GRMA exists to remove.
    #
    # Samples residualized against their parents instead get an identity row, since centering them
    # a second time would subtract the family effect twice.

    @staticmethod
    def relatedness(pairs, degrees=None):
        """Builds a relatedness frame from (index1, index2) pairs, full siblings by default"""

        degrees = degrees if degrees is not None else [sut.DEG_FULLSIB] * len(pairs)

        return pd.DataFrame({sut.FAM_INDEX_1 : [pair[0] for pair in pairs],
                             sut.FAM_INDEX_2 : [pair[1] for pair in pairs],
                             sut.KING_REL_COL : degrees})


    def test__a_pair_of_siblings__each_centered_against_the_other(self):
        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=2).toarray()

        assert np.allclose(actual, [[0.5, -0.5],
                                    [-0.5, 0.5]])


    def test__three_siblings__each_centered_against_the_other_two(self):
        # The weight is 1/k for each of the k relatives at the minimum degree, so a larger sibship
        # spreads the subtraction further rather than subtracting more
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 2), (1, 2)]), N=3).toarray()

        third = 1.0 / 3.0
        assert np.allclose(actual, [[2 * third, -third, -third],
                                    [-third, 2 * third, -third],
                                    [-third, -third, 2 * third]])


    def test__sample_with_no_relatives__contributes_nothing(self):
        # A ZERO row, not an identity one.  Without a relative there is no within-family contrast
        # to take, so the sample carries no information and is weighted out entirely.
        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=3).toarray()

        assert np.allclose(actual[2], [0.0, 0.0, 0.0])


    def test__no_related_pairs_at_all__everything_zero(self):
        actual = sut.calculate_R(self.relatedness([]), N=3).toarray()

        assert actual.shape == (3, 3)
        assert np.allclose(actual, 0.0)


    def test__closest_relatives_only__more_distant_ones_ignored(self):
        # Sample 0 has a sibling and a third-degree relative.  Only the sibling counts: centering
        # against a distant relative as well would mix in exactly the population structure the
        # within-family contrast is meant to avoid.
        actual = sut.calculate_R(
            self.relatedness([(0, 1), (0, 2)], [sut.DEG_FULLSIB, sut.DEG_3RD]), N=3).toarray()

        assert np.allclose(actual[0], [0.5, -0.5, 0.0])


    def test__closest_relative_differs_between_two_samples__R_is_not_symmetric(self):
        # A and B are third-degree relatives; B and C are second-degree; A and C are unrelated.
        # So A's closest relative is B, while B's is C -- and B does not appear in its own
        # closest group from A's point of view or the reverse.
        #
        # R is therefore NOT symmetric in general, which matters wherever it is multiplied: the
        # covariance of the residualized phenotypes is R omega R', and R' omega R is a different
        # matrix.
        actual = sut.calculate_R(
            self.relatedness([(0, 1), (1, 2)], [sut.DEG_3RD, sut.DEG_2ND]), N=3).toarray()

        assert np.allclose(actual[0], [0.5, -0.5, 0.0])
        assert np.allclose(actual[1], [0.0, 0.5, -0.5])
        assert not np.allclose(actual, actual.T)


    def test__two_relatives_tied_at_the_closest_degree__both_counted(self):
        # Sample 0 has two second-degree relatives and one third-degree.  Both of the former
        # count; the latter does not.
        #
        # The group being centered against INCLUDES the sample itself, so two relatives give
        # thirds rather than halves -- the same arithmetic as a sibship of three.
        actual = sut.calculate_R(
            self.relatedness([(0, 1), (0, 2), (0, 3)],
                             [sut.DEG_2ND, sut.DEG_2ND, sut.DEG_3RD]), N=4).toarray()

        third = 1.0 / 3.0
        assert np.allclose(actual[0], [2 * third, -third, -third, 0.0])


    def test__rows_of_samples_with_relatives__sum_to_zero(self):
        # What centering means: the row takes a difference, so a constant added to everybody's
        # genotypes passes through it unchanged
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 2), (1, 2), (3, 4)]),
                                 N=5).toarray()

        assert np.allclose(actual[:4].sum(axis=1), 0.0)


    @pytest.mark.parametrize("pairs", [
        pytest.param([(0, 1), (1, 2)], id="as_listed"),
        pytest.param([(1, 0), (2, 1)], id="both_reversed"),
        pytest.param([(1, 0), (1, 2)], id="one_reversed"),
    ])
    def test__pairs_listed_in_either_direction__same_result(self, pairs):
        # KING writes each pair once, in whichever order it happened to consider them
        expected = sut.calculate_R(self.relatedness([(0, 1), (1, 2)]), N=3).toarray()

        assert np.allclose(sut.calculate_R(self.relatedness(pairs), N=3).toarray(), expected)


    @pytest.mark.xfail(strict=True,
                       reason="duplicate pairs are summed rather than deduplicated; the "
                              "relatedness file is not yet checked for them")
    def test__the_same_pair_listed_twice__counted_once(self):
        # A repeated pair should change nothing: it says the same thing twice.  What happens
        # instead is that the row weights are built with a sparse constructor that SUMS entries
        # landing in the same cell, so the row sum comes out at 2 rather than 1 and the pair is
        # centered at two thirds rather than a half.
        #
        # The same is true of calculate_omega(), where the two covariances are added together.
        # Both want the relatedness file checked for duplicates, which also has to normalise the
        # order of each pair rather than only dropping exact repeats.
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 1)]), N=2).toarray()

        assert np.allclose(actual, [[0.5, -0.5],
                                    [-0.5, 0.5]])


    # ---- Samples residualized against their parents ---------------------------------------

    def test__affected_sample__gets_an_identity_row(self):
        # Their genotypes have already had their parents subtracted, so R passes them through
        # unchanged rather than centering them a second time
        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=2,
                                 affected_samples=np.array([0])).toarray()

        assert np.allclose(actual[0], [1.0, 0.0])


    def test__affected_sample_with_relatives__still_an_identity_row(self):
        # Parental residualization REPLACES the within-group centering rather than supplementing
        # it: doing both would subtract the family effect twice.  So having siblings changes
        # nothing for the affected sample itself.
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 2), (1, 2)]), N=3,
                                 affected_samples=np.array([0])).toarray()

        assert np.allclose(actual[0], [1.0, 0.0, 0.0])


    def test__relatives_of_an_affected_sample__still_center_against_them(self):
        # Only the affected sample's own ROW changes.  They remain a COLUMN in everybody else's,
        # and still count towards those samples' group size -- sample 1 below centers against both
        # 0 and 2, not only 2.
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 2), (1, 2)]), N=3,
                                 affected_samples=np.array([0])).toarray()

        third = 1.0 / 3.0
        assert np.allclose(actual[1], [-third, 2 * third, -third])
        assert np.allclose(actual[2], [-third, -third, 2 * third])


    def test__affected_sample_with_no_relatives__identity_rather_than_zero(self):
        # The contrast that makes parental residualization worth having.  Unaffected, this sample
        # would get a zero row and contribute nothing at all; affected, it contributes its own
        # Mendelian residual.  That is how somebody whose parents are genotyped but who has no
        # siblings in the sample becomes usable.
        unaffected = sut.calculate_R(self.relatedness([]), N=2).toarray()
        affected = sut.calculate_R(self.relatedness([]), N=2,
                                   affected_samples=np.array([0])).toarray()

        assert np.allclose(unaffected[0], [0.0, 0.0])
        assert np.allclose(affected[0], [1.0, 0.0])


    @pytest.mark.parametrize("affected_samples", [
        pytest.param(None, id="none_given"),
        pytest.param(np.array([], dtype=np.int64), id="empty_array"),
    ])
    def test__nobody_affected__ordinary_centering_throughout(self, affected_samples):
        expected = sut.calculate_R(self.relatedness([(0, 1)]), N=2).toarray()

        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=2,
                                 affected_samples=affected_samples).toarray()

        assert np.allclose(actual, expected)
        assert np.allclose(actual, [[0.5, -0.5], [-0.5, 0.5]])


    def test__several_affected_samples__each_gets_its_own_identity_row(self):
        actual = sut.calculate_R(self.relatedness([(0, 1), (0, 2), (1, 2)]), N=3,
                                 affected_samples=np.array([0, 2])).toarray()

        assert np.allclose(actual[0], [1.0, 0.0, 0.0])
        assert np.allclose(actual[2], [0.0, 0.0, 1.0])

        # Sample 1 is unaffected and still centers against both of them
        third = 1.0 / 3.0
        assert np.allclose(actual[1], [-third, 2 * third, -third])


    def test__every_sample_affected__the_identity_matrix(self):
        # Nothing is centered at all, since every genotype has already been differenced against
        # its own parents
        actual = sut.calculate_R(self.relatedness([(0, 1), (1, 2)]), N=3,
                                 affected_samples=np.array([0, 1, 2])).toarray()

        assert np.allclose(actual, np.identity(3))


    def test__affected_rows__no_division_by_zero(self):
        # An affected sample contributes no entries to its own row, so that row is empty when the
        # normalisation runs and its row sum is zero.  Dividing by it unguarded would put
        # infinities or NaNs into the matrix rather than the intended identity.
        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=3,
                                 affected_samples=np.array([0, 2])).toarray()

        assert np.isfinite(actual).all()


    def test__returned_matrix__sparse_and_square_over_every_sample(self):
        # N decides the size, not the relatedness data: samples appearing in no pair at all still
        # need rows, so that everything indexed by sample lines up
        actual = sut.calculate_R(self.relatedness([(0, 1)]), N=6)

        assert sp.issparse(actual)
        assert actual.shape == (6, 6)


class TestProcessGenotypes:

    def test__simple_inputs__expected_results(self):
        pass

    def test__sample_filter__expected_results(self):
        pass

    def test__snp_filter__expected_results(self):
        pass

    def test__process_multiple_blocks__expected_results(self):
        pass

    def test__filter_out_a_whole_block__expected_logging(self):
        pass

    def test__all_inputs__expected_results(self):
        pass


class TestCalculatePVals:

    @pytest.mark.parametrize("betas, ses, expected_p", [
        (np.array([1.96, 2.576]), np.array([1.0, 1.0]), np.array([0.05, 0.01])),
        (np.array([1.0, 1.0]), np.array([1.0/1.96, 1.0/2.576]), np.array([0.05, 0.01])),
        (np.array([1.96, 2.576]), np.array([-1.0, -1.0]), np.array([0.05, 0.01]))
    ])
    def test__precanned_tcs__expected_results(self, betas, ses, expected_p):
        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.allclose(actual_p.astype(float), expected_p, atol=0.00001)


    @pytest.mark.parametrize("rng_seed", [54, 87])
    def test__simple_inputs__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100

        betas = 2.0 * rng.random(N) - 1.0
        ses = 2.0 * rng.random(N) + 10.0

        expected_p = 2.0 * (1.0 - stats.norm.cdf(np.abs(betas / ses)))

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.allclose(actual_p.astype(float), expected_p)


    @pytest.mark.parametrize("rng_seed", [316, 9999])
    def test__negative_z__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100

        betas = 2.0 * rng.random(N) - 1.0
        ses = 2.0 * rng.random(N) + 10.0
        mask = rng.choice([True, False], size=N)

        first_p = sut.calculate_pvals(betas=betas, ses=ses)

        betas[mask] = -betas[mask]
        second_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.all(first_p == second_p)


    def test__include_very_small_p__expected_results(self):
        betas = np.array([39.0, -39.0])
        ses = np.array([1.0, 1.0])

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.all(np.char.find(actual_p, "-332") >= 0)


    def test__include_p_equals_1__expected_results(self):
        betas = np.array([0.0, 0.0, 0.0])
        ses = np.array([1.0, -1.0, 93478.0])

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.all(actual_p == "1.0e0")


class TestCreateOutput:

    def test__simple_inputs__expected_results(self):
        pass



# -------------------------
# Running a case with modified inputs, for the tests below that compare a case's results with those
# of a modified copy.  The modifications themselves live in the tests.

def _case_inputs(key):
    """Copies of a case's inputs that a test can modify freely: grma()'s data plus the snipar data"""

    case = tcs.REGISTRY[key]
    return dict(fam=case["FAM_DF"].copy(), rel=case["REL_DF"].copy(),
                bim=case["BIM_DF"].copy(), G=case["G"].copy(),
                pheno=copy.deepcopy(case["PHENO_DF"]), covar=copy.deepcopy(case["COVAR_DF"]),
                id_list=copy.deepcopy(case["ID_LIST"]), snp_list=copy.deepcopy(case["SNP_LIST"]),
                snps_per_block=case["SNPS_PER_BLOCK"], imputed=copy.deepcopy(case["IMPUTED_FILE"]))


def _run_inputs(inputs, degree, directory, name):
    """Runs grma() on inputs from _case_inputs(), writing the snipar data to a file if there is any"""

    imputed_file = None
    if inputs["imputed"] is not None:
        imputed_file = os.path.join(directory, name + snipar_hdf5.SNIPAR_FILE_SUFFIX)
        snipar_hdf5.write_snipar_file(imputed_file, **inputs["imputed"])

    kwargs = dict(rel_file=inputs["rel"], bed_file=inputs["G"], bim_file=inputs["bim"],
                  fam_file=inputs["fam"], rel_degree=degree, pheno_file=inputs["pheno"],
                  covar_file=inputs["covar"], id_list=inputs["id_list"],
                  snp_list=inputs["snp_list"], imputed_parental_file=imputed_file)
    if inputs["snps_per_block"] is not None:
        kwargs["snps_per_block"] = inputs["snps_per_block"]
    return sut.grma(**kwargs)


def _assert_same_results(original, modified, rsid_map=None, negated_rsids=()):
    """The two runs agree SNP for SNP, to rounding

    Args:
        rsid_map: original rsid -> the name it has in the modified run, if SNPs were renamed
        negated_rsids: SNPs whose beta should have changed sign (reported for the other allele)
    """

    original_rsids = original[bbf.BIM_RSID_COL].astype(str).tolist()
    expected_rsids = [rsid_map[r] for r in original_rsids] if rsid_map else original_rsids
    assert modified[bbf.BIM_RSID_COL].astype(str).tolist() == expected_rsids

    sign = np.array([-1.0 if r in negated_rsids else 1.0 for r in original_rsids])
    assert np.allclose(modified[sut.OUTPUT_BETA_COL].to_numpy(),
                       sign * original[sut.OUTPUT_BETA_COL].to_numpy(), atol=1e-10, rtol=0.0)
    assert np.allclose(modified[sut.OUTPUT_SE_COL].to_numpy(),
                       original[sut.OUTPUT_SE_COL].to_numpy(), atol=1e-10, rtol=0.0)


def _permuted_names(values, prefix, rng):
    """New names for values, assigned by a random permutation of their sorted order

    The permutation is redrawn until it is not the identity, so the new names never sort in the same
    order as the old -- anything that depends on IDs being sorted then shows up.
    """

    values = sorted(set(values))
    perm = rng.permutation(len(values))
    while len(values) > 1 and np.array_equal(perm, np.arange(len(values))):
        perm = rng.permutation(len(values))
    width = len(str(len(values)))
    return {v: f"{prefix}{p:0{width}d}" for v, p in zip(values, perm)}


def _new_ids(existing, count, prefix):
    """IDs for added samples, of the same kind as the existing ones so the column keeps its type"""

    if pd.api.types.is_integer_dtype(existing):
        start = int(existing.max()) + 1
        return list(range(start, start + count))
    return [f"{prefix}{n}" for n in range(count)]


class TestGRMA:

    @pytest.mark.parametrize("key, degree", tcs.params("add_singletons", default_xfail=tcs.XFAIL_13_6))
    def test__add_singletons__same_results(self, key, degree, tmp_path):
        """Unrelated samples are not in the regression, so adding them changes nothing (notes
        §13.6).  xfailed until §13.6 is fixed: today omega's variance includes them, and with
        anyone affected so does the phenotype centering mean."""

        original = _run_inputs(_case_inputs(key), degree, tmp_path, "original")

        inputs = _case_inputs(key)
        fam, num_new = inputs["fam"], 3
        new_fids = _new_ids(fam[bbf.FAM_FID_COL], num_new, "singleton_fam")
        new_iids = _new_ids(fam[bbf.FAM_IID_COL], num_new, "singleton")
        no_parent = 0 if pd.api.types.is_integer_dtype(fam[bbf.FAM_IIDF_COL]) else "0"
        new_phenos = [1.7, -2.3, 0.4]
        added = pd.DataFrame({bbf.FAM_FID_COL : new_fids, bbf.FAM_IID_COL : new_iids,
                              bbf.FAM_IIDF_COL : no_parent, bbf.FAM_IIDM_COL : no_parent,
                              bbf.FAM_SEX_COL : fam[bbf.FAM_SEX_COL].iloc[0],
                              bbf.FAM_PHENO_COL : new_phenos}, columns=bbf.FAM_COLS)
        inputs["fam"] = pd.concat([fam, added], ignore_index=True)

        num_snps = inputs["G"].shape[0]
        new_genotypes = np.array([[(snp + n) % 3 for n in range(num_new)]
                                  for snp in range(num_snps)], dtype=float)
        inputs["G"] = np.hstack([inputs["G"], new_genotypes])

        # Wherever the case filters samples, the singletons must pass the filter to reach the
        # analysis at all
        ids = added[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]]
        if inputs["id_list"] is not None:
            inputs["id_list"] = pd.concat([inputs["id_list"], ids.set_axis(
                inputs["id_list"].columns, axis=1)], ignore_index=True)
        if inputs["pheno"] is not None:
            rows = ids.copy(); rows[2] = new_phenos
            inputs["pheno"] = pd.concat([inputs["pheno"], rows.set_axis(
                inputs["pheno"].columns, axis=1)], ignore_index=True)
        if inputs["covar"] is not None:
            rows = ids.copy()
            for n, col in enumerate(inputs["covar"].columns[2:]):
                rows[col] = inputs["covar"][col].mean() + np.arange(num_new) - 1.0 + n
            inputs["covar"] = pd.concat([inputs["covar"], rows.set_axis(
                inputs["covar"].columns, axis=1)], ignore_index=True)

        _assert_same_results(original, _run_inputs(inputs, degree, tmp_path, "modified"))


    @pytest.mark.parametrize("key, degree", tcs.params("rename_samples"))
    def test__rename_samples__same_results(self, key, degree, tmp_path):
        """Sample IDs are labels: renaming every one consistently, in an order unrelated to the
        original, changes nothing"""

        original = _run_inputs(_case_inputs(key), degree, tmp_path, "original")
        rng = np.random.default_rng(seed=1206)
        inputs = _case_inputs(key)
        fam, rel, imputed = inputs["fam"], inputs["rel"], inputs["imputed"]
        id_files = [df for df in (inputs["id_list"], inputs["pheno"], inputs["covar"]) if df is not None]

        # Every IID, including parents named only in parent columns or in snipar's pedigree, but not
        # the missing-parent code
        iids = [*fam[bbf.FAM_IID_COL], *fam[bbf.FAM_IIDF_COL], *fam[bbf.FAM_IIDM_COL],
                *rel[sut.KING_IID1_COL], *rel[sut.KING_IID2_COL], *[v for df in id_files for v in df.iloc[:, 1]]]
        if imputed is not None:
            iids += list(imputed["pedigree"][:, 1:4].ravel())
        iid_map = _permuted_names([str(i) for i in iids if str(i) != "0"], "iid", rng)
        iid_map["0"] = "0"

        fids = [*fam[bbf.FAM_FID_COL], *rel[sut.KING_FID1_COL], *rel[sut.KING_FID2_COL],
                *[v for df in id_files for v in df.iloc[:, 0]]]
        fid_map = _permuted_names([str(f) for f in fids], "fid", rng)

        for col in (bbf.FAM_IID_COL, bbf.FAM_IIDF_COL, bbf.FAM_IIDM_COL):
            fam[col] = [iid_map[str(v)] for v in fam[col]]
        fam[bbf.FAM_FID_COL] = [fid_map[str(v)] for v in fam[bbf.FAM_FID_COL]]
        for col in (sut.KING_IID1_COL, sut.KING_IID2_COL):
            rel[col] = [iid_map[str(v)] for v in rel[col]]
        for col in (sut.KING_FID1_COL, sut.KING_FID2_COL):
            rel[col] = [fid_map[str(v)] for v in rel[col]]
        for df in id_files:
            df[df.columns[0]] = [fid_map[str(v)] for v in df.iloc[:, 0]]
            df[df.columns[1]] = [iid_map[str(v)] for v in df.iloc[:, 1]]

        # snipar's own family IDs link its pedigree to its imputed rows, so they are renamed too
        if imputed is not None:
            family_map = _permuted_names(list(imputed["families"]), "family", rng)
            pedigree = imputed["pedigree"].astype(object)
            pedigree[:, 0] = [family_map.get(str(f), str(f)) for f in pedigree[:, 0]]
            pedigree[:, 1:4] = np.vectorize(lambda v: iid_map[str(v)])(pedigree[:, 1:4])
            imputed["pedigree"] = pedigree.astype(str)
            imputed["families"] = np.array([family_map[str(f)] for f in imputed["families"]])

        _assert_same_results(original, _run_inputs(inputs, degree, tmp_path, "modified"))


    @pytest.mark.parametrize("key, degree", tcs.params("rename_snps"))
    def test__rename_snps__same_results(self, key, degree, tmp_path):
        """rsids are labels: renaming every one consistently changes nothing but the rsid column,
        and the output keeps the bim file's order whatever the names"""

        original = _run_inputs(_case_inputs(key), degree, tmp_path, "original")
        rng = np.random.default_rng(seed=1207)
        inputs = _case_inputs(key)
        bim, imputed = inputs["bim"], inputs["imputed"]

        rsids = [*bim[bbf.BIM_RSID_COL]]
        if imputed is not None:
            rsids += list(imputed["bim_values"][:, 1])
        rsid_map = _permuted_names([str(r) for r in rsids], "snp", rng)

        bim[bbf.BIM_RSID_COL] = [rsid_map[str(r)] for r in bim[bbf.BIM_RSID_COL]]
        if inputs["snp_list"] is not None:
            snp_list = inputs["snp_list"]
            snp_list[snp_list.columns[0]] = [rsid_map[str(r)] for r in snp_list.iloc[:, 0]]
        if imputed is not None:
            bim_values = imputed["bim_values"].astype(object)
            bim_values[:, 1] = [rsid_map[str(r)] for r in bim_values[:, 1]]
            imputed["bim_values"] = bim_values.astype(str)

        _assert_same_results(original, _run_inputs(inputs, degree, tmp_path, "modified"),
                             rsid_map=rsid_map)


    # TODO(jonbjala) Definitely need error cases



# Case harness ====================================================================================
#
# Generic tests over every case in tcs.REGISTRY.  A case states what it is built to have and,
# optionally, any intermediate values; anything it does not state is derived by the mocks in
# helper.py.  See tcs.py for the format.

def _assert_close(key, name, actual, expected):
    """Compares a value against a case's expectation, naming the case and quantity on failure"""

    actual = actual.toarray() if sp.issparse(actual) else np.asarray(actual, dtype=float)
    expected = expected.toarray() if sp.issparse(expected) else np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, \
        f"{key}: {name} has shape {actual.shape}, expected {expected.shape}"
    worst = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=tcs.tolerance(key), rtol=0.0), \
        f"{key}: {name} differs by up to {worst:.3g} ({tcs.provenance(key, name)} expectation)"


class TestCaseFormat:

    @pytest.mark.parametrize("key", tcs.KEYS)
    def test__case__follows_the_format(self, key):
        assert tcs.format_problems(key) == []


class TestCaseSelfConsistency:
    """Every value a case states that the mocks can also derive, checked against the mocks

    No production code runs here.  A failure means the case and the mocks disagree, so one of them
    is wrong -- a hand-derived matrix with a slip in it, a reference value computed under an older
    estimator (notes §15.1), or a mock that has drifted.  Either way it must be resolved before the
    production tests that use the case mean anything.
    """

    @pytest.mark.parametrize("key, name, degree", tcs.stated_derivable_params())
    def test__stated_value__agrees_with_mocks(self, key, name, degree):
        stated, derived = tcs.stated(key, name, degree), tcs.derived(key, name, degree)

        if name in ("AFFECTED", "ANALYSIS_IIDS", "RETAINED_RSIDS"):
            assert sorted(stated) == sorted(derived) if name == "AFFECTED" else stated == derived
        elif name == "PARENTAL_MEANS":
            assert sorted(stated) == sorted(derived)
            for iid in stated:
                assert np.allclose(stated[iid], derived[iid], atol=tcs.tolerance(key),
                                   equal_nan=True), f"{key}: PARENTAL_MEANS[{iid}]"
        else:
            _assert_close(key, name, stated, derived)


class TestCaseEndToEnd:

    @pytest.mark.parametrize("key, degree", tcs.params("end_to_end"))
    def test__grma__matches_expected(self, key, degree, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG):
            actual = sut.grma(**tcs.grma_kwargs(key, degree, tmp_path))

        assert actual[bbf.BIM_RSID_COL].astype(str).tolist() == tcs.get(key, "RETAINED_RSIDS")
        _assert_close(key, "BETAS", actual[sut.OUTPUT_BETA_COL], tcs.get(key, "BETAS", degree))
        _assert_close(key, "SES", actual[sut.OUTPUT_SE_COL], tcs.get(key, "SES", degree))

        for message, count in (tcs.REGISTRY[key]["EXPECTED_LOG_COUNTS"] or {}).items():
            assert caplog.text.count(message) == count, \
                f"{key}: expected {count} occurrence(s) of {message!r}"


class TestCaseInvariance:
    """Properties every case must have, checked against production alone

    These need no reimplementation of the method, so they can catch a gap the mocks share with
    production -- as notes §13.5 was: the mocks did not center phenotypes either, so they agreed.
    """

    @pytest.mark.parametrize("key, degree", tcs.params("phenotype_shift"))
    def test__phenotypes_shifted_by_a_constant__results_unchanged(self, key, degree, tmp_path):
        kwargs = tcs.grma_kwargs(key, degree, tmp_path)
        baseline = sut.grma(**kwargs)

        shifted_fam = kwargs["fam_file"].copy()
        shifted_fam[bbf.FAM_PHENO_COL] = shifted_fam[bbf.FAM_PHENO_COL].astype(float) + 10.0
        shifted_kwargs = dict(kwargs, fam_file=shifted_fam)
        if kwargs["pheno_file"] is not None:
            shifted_pheno = kwargs["pheno_file"].copy()
            shifted_pheno.iloc[:, 2] = shifted_pheno.iloc[:, 2].astype(float) + 10.0
            shifted_kwargs["pheno_file"] = shifted_pheno
        shifted = sut.grma(**shifted_kwargs)

        for col in (sut.OUTPUT_BETA_COL, sut.OUTPUT_SE_COL):
            assert np.allclose(shifted[col], baseline[col], atol=1e-9, rtol=0.0), \
                f"{key}: {col} depends on the phenotype's origin"


    @pytest.mark.parametrize("key, degree", tcs.params("allele_swap",
                                                       default_xfail=tcs.coding_dependence_xfail))
    def test__alleles_swapped__betas_negated(self, key, degree, tmp_path):
        """Swapping A1 and A2 in the bim, and recoding the genotypes to match, describes the same
        data with the other allele counted.  Betas are reported for A2, so a swapped SNP's beta
        changes sign and nothing else changes.

        For an imputed case the snipar file is left alone, so on the swapped SNPs it now disagrees
        with the bim and must be flipped -- the flip path, checked against exact truth.  Only
        non-palindromic SNPs are swapped: for A/T or C/G a swap looks like a strand difference,
        which is the unimplemented palindromic rescue (notes §12.1), not something to test here.
        """

        original = _run_inputs(_case_inputs(key), degree, tmp_path, "original")
        inputs = _case_inputs(key)
        bim = inputs["bim"]

        retained = set(tcs.get(key, "RETAINED_RSIDS"))
        palindromic = [{a1, a2} in sut.PALINDROMIC_ALLELE_PAIRS
                       for a1, a2 in zip(bim[bbf.BIM_A1_COL], bim[bbf.BIM_A2_COL])]
        candidates = [n for n, r in enumerate(bim[bbf.BIM_RSID_COL].astype(str))
                      if r in retained and not palindromic[n]]
        swap = candidates[::2]
        assert swap, f"{key} has no retained non-palindromic SNP to swap"

        bim.loc[swap, [bbf.BIM_A1_COL, bbf.BIM_A2_COL]] = \
            bim.loc[swap, [bbf.BIM_A2_COL, bbf.BIM_A1_COL]].to_numpy()
        inputs["G"][swap] = 2.0 - inputs["G"][swap]

        _assert_same_results(original, _run_inputs(inputs, degree, tmp_path, "modified"),
                             negated_rsids=set(bim[bbf.BIM_RSID_COL].astype(str).iloc[swap]))


class TestCaseEquivalence:
    """Cases that must give exactly the results of another case (EQUIVALENT_TO): the same data run
    a different way that should make no difference -- imputed rather than observed parents
    (notes §15.3), a different block size, parents named but absent"""

    @pytest.mark.parametrize("key, degree", tcs.params("equivalence", keys=tcs.equivalent_keys()))
    def test__results__identical_to_the_equivalent_case(self, key, degree, tmp_path):
        other = tcs.REGISTRY[key]["EQUIVALENT_TO"]
        actual = sut.grma(**tcs.grma_kwargs(key, degree, tmp_path))
        expected = sut.grma(**tcs.grma_kwargs(other, degree, tmp_path))
        _assert_same_results(expected, actual)
