"""
Testing of grma_lib.py
"""

import itertools as it
import logging
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

import bedbimfam as bbf
import grma
import grma_lib as sut

import helper
import tcs
import tc_allopt  # Test case that is really only used once


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

    def test__simple_fam_file__return_unchanged(self):
        N = 100

        fam_df = make_fam_df(N)

        actual_N_orig, actual_N, actual_indices, actual_id_df, actual_pheno = \
            sut.process_phenotypes(fam_file=fam_df)

        assert actual_N == actual_N_orig
        assert actual_N == N
        assert np.array_equal(actual_indices, np.arange(N))
        assert fam_df[sut.FAM_KEY].equals(actual_id_df)
        assert np.allclose(actual_pheno, fam_df[bbf.FAM_PHENO_COL].to_numpy())


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
        assert np.allclose(actual_pheno, phenotypes)


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
        assert np.allclose(actual_pheno, pheno_df[bbf.FAM_PHENO_COL].sort_index().to_numpy())
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

        actual_omega = sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno)

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

        actual_omega = sut.calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=pheno)

        # Verify result is PSD
        actual_eigenvalues = np.linalg.eigvalsh(actual_omega.toarray())
        assert np.all(actual_eigenvalues >= -1e-8)

        # Cross-validate: closed-form formula (production) vs. binary search (mock).
        # Both methods should find the same alpha; atol accounts for binary search precision.
        expected_omega = helper.mock_calc_omega(
            fam_df=fam_df, N=N, rel_df=king_rel_df, unresidualized_phenotypes=pheno
        )
        assert np.allclose(actual_omega.toarray(), expected_omega.toarray(), atol=1e-3)

    def test__per_block_alpha__single_component__matches_global(self):
        # With only one multi-person component there is nothing to separate, so
        # per-block shrinkage must reproduce the global result exactly.
        N = 5
        pheno = np.array([1.0, -1.0, 0.1, 0.1, 0.1])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0],
            sut.FAM_INDEX_2: [1],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING],
        })

        global_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                           unresidualized_phenotypes=pheno)
        block_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                          unresidualized_phenotypes=pheno,
                                          per_block_alpha=True)

        assert np.allclose(global_omega.toarray(), block_omega.toarray())

    def test__per_block_alpha__default_is_unchanged(self):
        # The flag must be opt-in: omitting it has to give the same output as
        # the released behaviour.
        #
        # allclose rather than array_equal: get_lambda_min calls ARPACK, which
        # seeds itself randomly, so repeated calls on the SAME matrix differ in
        # the last ULP or two (-1.0 vs -0.99999999999999989 observed).  That
        # non-determinism predates this change and affects the global path too.
        N = 6
        pheno = np.array([1.0, -1.0, 0.9, -0.8, 0.1, 0.1])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0, 2],
            sut.FAM_INDEX_2: [1, 3],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING, sut.DEG_3RD],
        })

        default_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                            unresidualized_phenotypes=pheno)
        explicit_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                             unresidualized_phenotypes=pheno,
                                             per_block_alpha=False)

        assert np.allclose(default_omega.toarray(), explicit_omega.toarray(),
                           rtol=0, atol=1e-12)

    def test__per_block_alpha__innocent_block_not_shrunk(self):
        # Two disjoint components carrying the SAME relatedness degree, so both
        # get the same covariance rho, and any difference in their shrinkage is
        # due to STRUCTURE alone:
        #
        #   block A, a star 0-1 / 0-2 / 0-3   lambda_min = -rho*sqrt(3)
        #   block B, a single pair 4-5        lambda_min = -rho
        #
        # The phenotypes below put var(y) between the two, so the star breaks
        # positive-definiteness and the pair does not.  Under a global alpha the
        # innocent pair is shrunk anyway; under per-block alpha it is untouched.
        N = 10
        pheno = np.array([0.680193, -0.845163, -0.010330, 0.699210, -1.368729,
                          -1.756903, 0.215909, 0.036958, -0.204671, 0.032086])
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [0, 0, 0, 4],
            sut.FAM_INDEX_2: [1, 2, 3, 5],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * 4,
        })

        rel_to_cov = sut.get_rel_covariances(rel_df=rel_df,
                                             unresidualized_phenotypes=pheno)
        rho = rel_to_cov[sut.DEG_PARENT_OFFSPRING]
        pheno_var = float(np.var(pheno))
        # the premise of the test
        assert rho < pheno_var < rho * np.sqrt(3.0)

        global_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                           unresidualized_phenotypes=pheno).toarray()
        block_omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                          unresidualized_phenotypes=pheno,
                                          per_block_alpha=True).toarray()

        # Global: the star's shrinkage is imposed on the innocent pair too.
        assert np.isclose(global_omega[0, 1], global_omega[4, 5])
        assert abs(global_omega[4, 5]) < abs(rho)

        # Per block: the pair keeps its unshrunk covariance, the star does not.
        assert np.isclose(block_omega[4, 5], rho)
        assert abs(block_omega[0, 1]) < abs(rho)
        assert not np.isclose(block_omega[0, 1], block_omega[4, 5])

        # And the whole matrix is still positive semi-definite.
        assert np.all(np.linalg.eigvalsh(block_omega) >= -1e-8)

    def test__per_block_alpha__result_is_psd(self):
        # The guarantee that matters: shrinking per block must still leave the
        # whole matrix positive semi-definite.
        rng = np.random.default_rng(20260831)
        N = 40
        pheno = rng.normal(size=N)
        pairs = [(i, i + 1) for i in range(0, N - 1, 2)] + [(0, 2), (4, 6), (8, 10)]
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [p[0] for p in pairs],
            sut.FAM_INDEX_2: [p[1] for p in pairs],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * len(pairs),
        })

        omega = sut.calculate_omega(rel_df=rel_df, N=N,
                                    unresidualized_phenotypes=pheno,
                                    per_block_alpha=True)

        assert np.all(np.linalg.eigvalsh(omega.toarray()) >= -1e-8)

    def test__get_block_alphas__gershgorin_screen_is_safe(self):
        # The Gershgorin screen skips blocks it can prove are definite.  Verify
        # it never skips one that needed shrinking, by checking every returned
        # alpha against a direct per-block eigendecomposition.
        rng = np.random.default_rng(1234)
        N = 60
        pheno = rng.normal(size=N)
        pairs = [(i, i + 1) for i in range(0, N - 1, 3)] + [(0, 3), (6, 9)]
        rel_df = pd.DataFrame({
            sut.FAM_INDEX_1: [p[0] for p in pairs],
            sut.FAM_INDEX_2: [p[1] for p in pairs],
            sut.KING_REL_COL: [sut.DEG_PARENT_OFFSPRING] * len(pairs),
        })
        rel_to_cov = sut.get_rel_covariances(rel_df=rel_df,
                                             unresidualized_phenotypes=pheno)
        i1 = rel_df[sut.FAM_INDEX_1].to_numpy(np.int64)
        i2 = rel_df[sut.FAM_INDEX_2].to_numpy(np.int64)
        data = np.tile(rel_df[sut.KING_REL_COL].map(rel_to_cov).to_numpy(), 2)
        off_diag = sp.coo_array((data, (np.concatenate([i1, i2]),
                                        np.concatenate([i2, i1]))),
                                shape=(N, N)).tocsr()
        pheno_var = float(np.var(pheno))

        alphas, labels = sut.get_block_alphas(off_diag, pheno_var)

        n_comp = labels.max() + 1
        for c in range(n_comp):
            idx = np.flatnonzero(labels == c)
            if len(idx) < 2:
                continue
            block = off_diag[idx][:, idx].toarray()
            lmin = float(np.linalg.eigvalsh(block).min())
            expected = (min((sut.DEFAULT_OMEGA_EPSILON - pheno_var) / lmin, 1.0)
                        if lmin < 0.0 else 1.0)
            assert np.isclose(alphas[c], expected, atol=1e-6)




class TestProcessRelatedness:

    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED, tcs.KEY_PT])
    def test__precanned_tcs__expected_results(self, rel_degree, tc_key):
        rel_df, fam_df = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]
        N = len(fam_df)

        expected_R = tcs.R_MATRICES[tc_key][rel_degree]

        unres_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy()


        expected_omega = helper.mock_calc_omega(fam_df=fam_df, N=N, rel_df=rel_df,
                                                unresidualized_phenotypes=unres_pheno)
        expected_se_matrix = expected_R @ expected_omega @ expected_R.T

        expected_res_pheno = tcs.DEMEANED_P[tc_key][rel_degree]


        actual_R, actual_se_matrix, actual_res_pheno = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)


        assert np.allclose(expected_R.toarray(), actual_R.toarray(), atol=0.001)
        assert np.allclose(expected_res_pheno, actual_res_pheno, atol=0.001)
        assert np.allclose(expected_se_matrix.toarray(), actual_se_matrix.toarray(), atol=0.001)


    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("rng_seed", [1, 34])
    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED])
    def test__randomized_phenos__expected_results(self, rel_degree, rng_seed, tc_key):
        rng = np.random.default_rng(seed=rng_seed)

        rel_df, fam_df = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]
        N = len(fam_df)

        expected_R = tcs.R_MATRICES[tc_key][rel_degree]

        unres_pheno = 10.0 * rng.random(N)
        expected_omega = helper.mock_calc_omega(fam_df=fam_df, N=N, rel_df=rel_df,
                                                unresidualized_phenotypes=unres_pheno)
        expected_se_matrix = expected_R @ expected_omega @ expected_R.T
        
        expected_res_pheno = expected_R @ unres_pheno


        actual_R, actual_se_matrix, actual_res_pheno = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        assert np.allclose(expected_R.toarray(), actual_R.toarray(), atol=0.001)
        assert np.allclose(expected_res_pheno, actual_res_pheno, atol=0.001)
        assert np.allclose(expected_se_matrix.toarray(), actual_se_matrix.toarray(), atol=0.001)


    @pytest.mark.parametrize("rng_seed", [1, 34])
    def test__permute_rel_file__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]
        rel_df_2 = rel_df_1.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
        N = len(fam_df)


        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_se_matrix_1, actual_res_pheno_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        actual_R_2, actual_se_matrix_2, actual_res_pheno_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_se_matrix_1.toarray(), actual_se_matrix_2.toarray(), atol=0.001)



    @pytest.mark.parametrize("rng_seed", [5, 74])
    def test__swap_id_cols_in_rel_file__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]
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

        actual_R_1, actual_se_matrix_1, actual_res_pheno_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        actual_R_2, actual_se_matrix_2, actual_res_pheno_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_se_matrix_1.toarray(), actual_se_matrix_2.toarray(), atol=0.001)


    @pytest.mark.parametrize("rng_seed", [44, 90233])
    def test__rearrange_fam_file__rearranged_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_2FAM

        rel_df, fam_df_1 = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]

        N = len(fam_df_1)
        permutation = rng.permutation(N)
        fam_df_2 = fam_df_1.iloc[permutation].reset_index(drop=True)

        
        unres_pheno = 10.0 * rng.random(N)

        actual_R_1, actual_se_matrix_1, actual_res_pheno_1 = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df_1, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        actual_R_2, actual_se_matrix_2, actual_res_pheno_2 = \
            sut.process_relatedness(rel_file=rel_df, fam_df=fam_df_2, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno[permutation])


        assert np.allclose(actual_R_1.toarray()[permutation][:, permutation], actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1[permutation], actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_se_matrix_1.toarray()[permutation][:, permutation], actual_se_matrix_2.toarray(), atol=0.001)


    @pytest.mark.parametrize("rng_seed", [64, 738])
    def test__permute_ids__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        rel_degree = sut.MAX_GRMA_RELATEDNESS
        tc_key = tcs.KEY_VARIED

        rel_df_1, fam_df_1 = tcs.REL_DFS[tc_key], tcs.FAM_DFS[tc_key]
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

        actual_R_1, actual_se_matrix_1, actual_res_pheno_1 = \
            sut.process_relatedness(rel_file=rel_df_1, fam_df=fam_df_1, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)

        actual_R_2, actual_se_matrix_2, actual_res_pheno_2 = \
            sut.process_relatedness(rel_file=rel_df_2, fam_df=fam_df_2, rel_degree=rel_degree,
                                    unresidualized_phenotypes=unres_pheno)


        assert np.allclose(actual_R_1.toarray(), actual_R_2.toarray(), atol=0.001)
        assert np.allclose(actual_res_pheno_1, actual_res_pheno_2, atol=0.001)
        assert np.allclose(actual_se_matrix_1.toarray(), actual_se_matrix_2.toarray(), atol=0.001)


    def test__fully_randomized__expected_results(self):
        pass


class TestProcessBimFile:

    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED])
    def test__precanned_tcs__expected_results(self, tc_key):
        bim_df = tcs.BIM_DFS[tc_key]

        expected_M_orig = len(bim_df)
        expected_snp_filter = tcs.SNP_FILTERS[tc_key]
        expected_M = len(expected_snp_filter) if expected_snp_filter is not None else expected_M_orig

        actual_M_orig, actual_M, actual_snp_filter = sut.process_bim_file(
            bim_file=bim_df, snp_list=tcs.SNP_LISTS[tc_key])

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

        actual_M_orig, actual_M, actual_snp_filter = sut.process_bim_file(
            bim_file=bim_df, snp_list=snp_list)

        assert actual_M_orig == M
        assert actual_M == expected_M
        assert np.allclose(actual_snp_filter, expected_snp_filter)


    @pytest.mark.parametrize("rng_seed", [1, 10])
    def test__shuffle_snp_list__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        tc_key = tcs.KEY_VARIED
        bim_df = tcs.BIM_DFS[tc_key]

        expected_M_orig = len(bim_df)

        expected_snp_filter = tcs.SNP_FILTERS[tc_key]
        expected_M = len(expected_snp_filter) if expected_snp_filter is not None else expected_M_orig

        actual_M_orig, actual_M, actual_snp_filter = sut.process_bim_file(
            bim_file=bim_df, snp_list=tcs.SNP_LISTS[tc_key].sample(frac=1, random_state=rng))

        assert expected_M_orig == actual_M_orig
        assert expected_M == actual_M
        assert (expected_snp_filter is None and actual_snp_filter is None) or \
               np.allclose(expected_snp_filter, actual_snp_filter)


class TestGetResidualizedGenotypeData:

    # TODO(jonbjala)  Test some with an actual bed file?
    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("tc_key", [tcs.KEY_SIMPLE, tcs.KEY_2FAM, tcs.KEY_VARIED])
    def test__precanned_tcs__expected_results(self, tc_key, rel_degree):
        
        snp_filter = tcs.SNP_FILTERS[tc_key]
        sample_filter = tcs.SAMPLE_FILTERS[tc_key]
        expected_resid_G = tcs.DEMEANED_G[tc_key][rel_degree][snp_filter] if snp_filter is not None\
            else tcs.DEMEANED_G[tc_key][rel_degree]
        expected_resid_G = expected_resid_G[:, sample_filter] if sample_filter is not None \
            else expected_resid_G

        orig_G = tcs.G[tc_key]

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=orig_G,
            M_orig=orig_G.shape[0], N_orig=orig_G.shape[1], snp_filter=snp_filter,
            sample_filter=sample_filter, R=tcs.R_MATRICES[tc_key][rel_degree],
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


    @pytest.mark.parametrize("rng_seed", [33, 87, 904, 8756])
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


    @pytest.mark.parametrize("rng_seed", [68, 4444])
    def test__snp_filter_all_snps__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        num_snps_selected = rng.integers(int(M / 3), int(2 * M/3))
        snp_filter = np.sort(rng.choice(M, size=num_snps_selected, replace=False))

        G = rng.choice(M, size=(M, N), replace=True)
        R = sp.csr_array(np.identity(N))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=snp_filter, sample_filter=None, R=R, M_start=0, num_snps=M)

        assert actual_M == num_snps_selected
        assert actual_resid_G.shape == (num_snps_selected, N)
        assert np.allclose(actual_resid_G, G[snp_filter])


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


    @pytest.mark.parametrize("rng_seed", [5, 25])
    def test__sample_filter__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        num_samples_selected = rng.integers(int(N / 3), int(2 * N/3))
        sample_filter = np.sort(rng.choice(N, size=num_samples_selected, replace=False))

        G = rng.choice(M, size=(M, N), replace=True)
        R = sp.csr_array(np.identity(num_samples_selected))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=None, sample_filter=sample_filter, R=R, M_start=0, num_snps=M)

        assert actual_resid_G.shape == (M, num_samples_selected)
        assert np.allclose(actual_resid_G, G[:, sample_filter])


    @pytest.mark.parametrize("rng_seed", [7, 59, 96])
    def test__both_filters__expected_filtering(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        num_snps_selected = rng.integers(int(M / 3), int(2 * M/3))
        num_samples_selected = rng.integers(int(N / 3), int(2 * N/3))
        snp_filter = np.sort(rng.choice(M, size=num_snps_selected, replace=False))
        sample_filter = np.sort(rng.choice(N, size=num_samples_selected, replace=False))

        G = rng.choice(M, size=(M, N), replace=True)
        R = sp.csr_array(np.identity(num_samples_selected))

        actual_resid_G, actual_M = sut.get_residualized_genotype_data(bed_file=G,
            M_orig=M, N_orig=N, snp_filter=snp_filter, sample_filter=sample_filter, R=R, M_start=0, num_snps=M)

        assert actual_M == num_snps_selected
        assert actual_resid_G.shape == (num_snps_selected, num_samples_selected)
        assert np.allclose(actual_resid_G, G[snp_filter][:, sample_filter])


    @pytest.mark.parametrize("rng_seed", [8, 67])
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


    @pytest.mark.parametrize("rng_seed", [102, 345])
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


class TestCalculateBetas:

    @pytest.mark.parametrize("rng_seed", [467, 4892, 94720])
    def test__simple_inputs__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = 2.0 * rng.random(size=(M, N)) - 1.0
        pheno = 4.0 * rng.random(size=N) - 2.0

        expected_betas = helper.mock_run_regressions(demeaned_G=G, demeaned_P=pheno)
        expected_xtx = np.sum(G * G, axis=1)

        actual_betas, actual_xtx = sut.calculate_betas(genotypes=G, phenotypes=pheno)

        assert np.allclose(actual_betas, expected_betas)
        assert np.allclose(actual_xtx, expected_xtx)


    @pytest.mark.parametrize("rng_seed", [4, 67])
    def test__include_0_xtx__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = 2.0 * rng.random(size=(M, N)) - 1.0
        row = rng.integers(0, M)
        G[row] = 0.0
        pheno = 4.0 * rng.random(size=N) - 2.0

        expected_betas = helper.mock_run_regressions(demeaned_G=G, demeaned_P=pheno)
        expected_xtx = helper.mock_calc_xtx(G)

        actual_betas, actual_xtx = sut.calculate_betas(genotypes=G, phenotypes=pheno)

        assert np.allclose(actual_betas, expected_betas)
        assert np.allclose(actual_xtx, expected_xtx)
        assert actual_xtx[row] == np.finfo(actual_xtx.dtype).eps


class TestCalculateSEs:

    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    @pytest.mark.parametrize("tc_key", tcs.KEYS)
    def test__precanned_tcs__expected_results(self, tc_key, rel_degree):

        demeaned_G, fam_df, R, rel_df = tcs.DEMEANED_G[tc_key][rel_degree], \
                                        tcs.FAM_DFS[tc_key], \
                                        tcs.R_MATRICES[tc_key][rel_degree], \
                                        tcs.REL_DFS[tc_key]

        unres_pheno = fam_df[bbf.FAM_PHENO_COL].to_numpy()

        expected_omega = helper.mock_calc_omega(fam_df=fam_df, N=len(fam_df), rel_df=rel_df,
                                                unresidualized_phenotypes=unres_pheno)

        expected_se_matrix = R @ expected_omega @ R.T
                                                           

        expected_ses = helper.mock_calc_ses(demeaned_G=demeaned_G,
                                            se_matrix=expected_se_matrix)

        actual_ses = sut.calculate_ses(genotypes=demeaned_G, se_matrix=expected_se_matrix,
                                       XtX=helper.mock_calc_xtx(demeaned_G))

        assert np.allclose(expected_ses, actual_ses, equal_nan=True, atol=0.001)



    @pytest.mark.parametrize("rng_seed", [57, 88, 902])
    def test__randomized_entries__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = 2.0 * rng.random(size=(M, N)) - 1.0

        se_matrix = rng.random(size=(N, N))

        expected_ses = helper.mock_calc_ses(demeaned_G=G, se_matrix=se_matrix)

        actual_ses = sut.calculate_ses(genotypes=G, se_matrix=se_matrix,
                                       XtX=helper.mock_calc_xtx(G))

        assert np.allclose(expected_ses, actual_ses, equal_nan=True, atol=0.0001)


    @pytest.mark.parametrize("rng_seed", [89, 376])
    def test__include_0_genotype_entries__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)
        M = 20
        N = 100

        G = 2.0 * rng.random(size=(M, N)) - 1.0
        row = rng.integers(0, M)
        G[row] = 0.0

        se_matrix = rng.random(size=(N, N))

        expected_ses = helper.mock_calc_ses(demeaned_G=G, se_matrix=se_matrix)

        actual_ses = sut.calculate_ses(genotypes=G, se_matrix=se_matrix,
                                       XtX=helper.mock_calc_xtx(G))

        assert np.allclose(expected_ses, actual_ses, equal_nan=True, atol=0.0001)

    # TODO(jonbjala) Add more tests?

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



class TestGRMA:

    @pytest.mark.parametrize("tc_key", tcs.KEYS)
    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    def test__precanned_tcs__expected_results(self, tc_key, rel_degree):

        filtered_G = tcs.G[tc_key]
        filtered_pheno = tcs.FAM_DFS[tc_key][bbf.FAM_PHENO_COL].to_numpy()
        if tcs.SNP_FILTERS[tc_key] is not None:
            filtered_G = filtered_G[tcs.SNP_FILTERS[tc_key]]
        if tcs.SAMPLE_FILTERS[tc_key] is not None:
            filtered_G = filtered_G[:, tcs.SAMPLE_FILTERS[tc_key]]
            filtered_pheno = filtered_pheno[tcs.SAMPLE_FILTERS[tc_key]]

        actual_results = sut.grma(
            rel_file=tcs.REL_DFS[tc_key],
            bed_file=tcs.G[tc_key],
            bim_file=tcs.BIM_DFS[tc_key],
            fam_file=tcs.FAM_DFS[tc_key],
            rel_degree=rel_degree,
            id_list=tcs.SAMPLE_LISTS[tc_key],
            snp_list=tcs.SNP_LISTS[tc_key]
        )


        test_betas, test_ses = helper.mock_grma(
            rel_df=tcs.REL_DFS[tc_key],
            fam_df=tcs.FAM_DFS[tc_key],
            G=filtered_G,
            pheno=filtered_pheno,
            rel_deg=rel_degree
        )
        expected_betas, expected_ses = test_betas, test_ses
        assert np.allclose(actual_results[sut.OUTPUT_BETA_COL].to_numpy(), expected_betas, atol=0.001, equal_nan=True)
        assert np.allclose(actual_results[sut.OUTPUT_SE_COL].to_numpy(), expected_ses, atol=0.001, equal_nan=True)
        # TODO(jonbjala)  Should test P and Sum_Sq_X at some point


    @pytest.mark.parametrize("rel_degree", range(sut.MAX_GRMA_RELATEDNESS + 1))
    def test__all_opts__expected_results(self, rel_degree, caplog):

        with caplog.at_level(logging.DEBUG):
            actual_results = sut.grma(
                rel_file=tc_allopt.REL_DF,
                bed_file=tc_allopt.G,
                bim_file=tc_allopt.BIM_DF,
                fam_file=tc_allopt.FAM_DF,
                pheno_file=tc_allopt.PHENO_DF,
                covar_file=tc_allopt.COVAR_DF,
                rel_degree=rel_degree,
                id_list=tc_allopt.SAMPLE_DF,
                snp_list=tc_allopt.SNP_LIST,
                snps_per_block=tc_allopt.SNPS_PER_BLOCK
            )
        
        captured = caplog.text


        assert "All SNPs filtered out in this block" in captured
        assert len(actual_results) == len(tc_allopt.SNP_LIST)
        assert actual_results[bbf.BIM_RSID_COL].equals(tc_allopt.SNP_LIST[bbf.BIM_RSID_COL])
        # TODO(jonbjala) Should probably compare actual results (betas, ses, etc)

    def test__add_singletons__same_results(self):
        pass


    def test__rename_samples__same_results(self):
        pass


    def test__rename_snps__same_results(self):
        pass


    # TODO(jonbjala) Definitely need error cases
