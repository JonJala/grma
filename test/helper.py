import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import scipy.sparse.linalg as spla

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
import grma
import grma_lib
import bedbimfam as bbf

'''
This contains a straightforward, unscalable version of GRMA to use to test against in small enough
examples.

'''

# Assumes IDs are integers in the range of 0 to N (and that FID is unused)
def mock_create_rel_info(king_df: pd.DataFrame, fam_df: pd.DataFrame, rel_deg: int, N: int):
    ids = sorted(set(king_df[grma_lib.KING_IID1_COL].tolist()).union(
                 set(king_df[grma_lib.KING_IID2_COL].tolist())))

    rel_thresh_val = rel_deg

    fam_df_copy = fam_df.copy()
    fam_df_copy["INDEX"] = list(range(N))

    rel_df = king_df.copy()
    rel_df = pd.merge(left=rel_df, right=fam_df_copy, left_on=[grma_lib.KING_FID1_COL, grma_lib.KING_IID1_COL],
                      right_on=[bbf.FAM_FID_COL, bbf.FAM_IID_COL])
    rel_df.rename(columns={"INDEX": "INDEX1"}, inplace=True)
    rel_df = pd.merge(left=rel_df, right=fam_df_copy, left_on=[grma_lib.KING_FID2_COL, grma_lib.KING_IID2_COL],
                      right_on=[bbf.FAM_FID_COL, bbf.FAM_IID_COL])
    rel_df.rename(columns={"INDEX": "INDEX2"}, inplace=True)

    # Find the closest relationship levels for everyone
    lowest_rel = [5] * N

    for index, row in rel_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]

        if cur_rel < lowest_rel[row["INDEX1"]]:
            lowest_rel[row["INDEX1"]] = cur_rel
        if cur_rel < lowest_rel[row["INDEX2"]]:
            lowest_rel[row["INDEX2"]] = cur_rel


    # Construct the rel_info object and se_info objects
    rel_info = [[i] for i in range(N)]
    for index, row in rel_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]

        if cur_rel == lowest_rel[row["INDEX1"]] and cur_rel <= rel_thresh_val:
            rel_info[row["INDEX1"]].append(row["INDEX2"])

        if cur_rel == lowest_rel[row["INDEX2"]] and cur_rel <= rel_thresh_val:
            rel_info[row["INDEX2"]].append(row["INDEX1"])

    # Sort the rel_info entries
    for i, rel in enumerate(rel_info):
        rel_info[i] = sorted(rel)

    return rel_info



def mock_shrink_offdiag(matrix: sp.csr_array, weight: float = 0.75):
    """Independent per-block reimplementation of the shrinkage applied in calculate_omega().

    Deliberately does NOT use the closed form (epsilon - variance) / lambda_min that production
    uses.  Instead it binary-searches, within each connected component, for the largest alpha
    that keeps that component positive definite.  Agreement between the two is therefore a real
    cross-check rather than a restatement of the same formula.

    Note that this makes no attempt to replicate the Gershgorin screen in get_block_alphas().
    That screen only decides which blocks can skip an eigendecomposition -- it never changes an
    alpha -- so examining every block here is equivalent by construction, and means a faulty
    screen shows up as a disagreement rather than being replicated into the expected value.
    The same reasoning applies to the dense/ARPACK routing in get_lambda_min(): this uses a dense
    eigendecomposition throughout, so a mistake in that routing is also visible here.
    """

    diag = matrix.diagonal()
    off_diag = matrix.copy()
    off_diag.setdiag(0)
    off_diag.eliminate_zeros()

    n_comp, labels = connected_components(off_diag, directed=False)
    dense_off = off_diag.toarray()
    alphas = np.ones(n_comp)

    for c in range(n_comp):
        idx = np.nonzero(labels == c)[0]
        if len(idx) < 2:
            continue

        block_off = dense_off[np.ix_(idx, idx)]
        block_diag = np.diag(diag[idx])

        if np.linalg.eigvalsh(block_diag + block_off).min() > 0.0:
            continue

        alpha_min = 0.0
        alpha_max = 1.0
        for _ in range(35):
            alpha = weight * alpha_max + (1.0 - weight) * alpha_min
            if np.linalg.eigvalsh(block_diag + alpha * block_off).min() > 0.0:
                alpha_min = alpha
            else:
                alpha_max = alpha

        alphas[c] = alpha_min

    # Both members of a pair are in the same component, so scaling row i by its own component's
    # alpha scales every entry correctly.
    return sp.csr_array(np.diag(diag) + dense_off * alphas[labels][:, None])


def mock_calc_omega(fam_df: pd.DataFrame, N: int, rel_df: pd.DataFrame,
                    unresidualized_phenotypes: np.ndarray):

    FAM_INDEX1_COL = "Fam Index 1"
    FAM_INDEX2_COL = "Fam Index 2"

    master_df = rel_df.copy()


    fam_idx = pd.MultiIndex.from_frame(fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]])

    master_df[FAM_INDEX1_COL] = fam_idx.get_indexer(
        pd.MultiIndex.from_arrays([rel_df[grma_lib.KING_FID1_COL], rel_df[grma_lib.KING_IID1_COL]])
    )
    master_df[FAM_INDEX2_COL] = fam_idx.get_indexer(
        pd.MultiIndex.from_arrays([rel_df[grma_lib.KING_FID2_COL], rel_df[grma_lib.KING_IID2_COL]])
    )

    master_df[grma_lib.KING_REL_COL] = master_df[grma_lib.KING_REL_COL].map(grma_lib.INFTYPE_TO_DEG_MAP)

    master_df = master_df[[FAM_INDEX1_COL, FAM_INDEX2_COL, grma_lib.KING_REL_COL]]
    master_df = master_df[master_df[grma_lib.KING_REL_COL] <= grma_lib.SE_RELATEDNESS]

    rel_to_cov_dict = dict()
    for rel_deg in range(grma_lib.SE_RELATEDNESS + 1):
        temp_df = master_df[master_df[grma_lib.KING_REL_COL] == rel_deg]

        ind1 = np.concatenate([temp_df[FAM_INDEX1_COL].to_numpy(int), temp_df[FAM_INDEX2_COL].to_numpy(int)])
        ind2 = np.concatenate([temp_df[FAM_INDEX2_COL].to_numpy(int), temp_df[FAM_INDEX1_COL].to_numpy(int)])

        covar = np.cov(unresidualized_phenotypes[ind1], unresidualized_phenotypes[ind2], ddof=0)[0, 1] \
            if len(temp_df) > 0 else np.nan

        rel_to_cov_dict[rel_deg] = covar


    master_df["COV"] = master_df[grma_lib.KING_REL_COL].map(rel_to_cov_dict)

    variance = np.var(unresidualized_phenotypes)

    rows = np.concatenate([master_df[FAM_INDEX1_COL].to_numpy(int), master_df[FAM_INDEX2_COL].to_numpy(int), np.arange(N)])
    cols = np.concatenate([master_df[FAM_INDEX2_COL].to_numpy(int), master_df[FAM_INDEX1_COL].to_numpy(int), np.arange(N)])
    data = np.concatenate([master_df["COV"].to_numpy(), master_df["COV"].to_numpy(), variance * np.ones(N)])

    omega = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    shrunk_omega = mock_shrink_offdiag(omega)

    return shrunk_omega


def mock_construct_R(rel_input: list):
    N = len(rel_input)

    result = np.identity(N, dtype=float)
    for row_num, row in enumerate(rel_input):
        result[row_num, row] -= np.reciprocal(float(len(row)))

    return sp.csr_array(result)


# Assume K x N (where K is likely M or 1 depending on whether this is G or the phenotypes)
def mock_demean(rel_input: list, arr: np.ndarray):
    result = arr.copy()
    
    for person_num, row in enumerate(rel_input):
        result[:, person_num] -= np.mean(arr[:, row], axis=1)

    result[np.isnan(result)] = 0.0

    return result


#################################
def mock_gls_estimates(genotypes, gls_weights, weighted_phenotypes):
    """Works the GLS estimates out one SNP at a time

    The library does this with two einsums over a whole block at once, so the obvious scalar form
    is a genuinely separate route rather than a restatement of it.

    Args:
        genotypes: (snps, samples) residualized genotypes
        gls_weights: (samples, samples) weight matrix, sparse or dense
        weighted_phenotypes: (samples,) phenotypes already multiplied by the weights

    Returns:
        A tuple (betas, ses, XtWX) of (snps,) arrays, matching the library's return order
    """

    weights = gls_weights.toarray() if sp.issparse(gls_weights) else gls_weights

    XtWX, ses, betas = [], [], []
    for snp in genotypes:
        denominator = snp @ weights @ snp
        XtWX.append(denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            ses.append(1.0 / np.sqrt(denominator))
            betas.append((snp @ weighted_phenotypes) / denominator)

    return np.array(betas), np.array(ses), np.array(XtWX)


def mock_grma(rel_df: pd.DataFrame, fam_df: pd.DataFrame, G: np.ndarray, pheno: np.ndarray, rel_deg: int):
    M, N = G.shape

    rel_info = mock_create_rel_info(rel_df, fam_df, rel_deg, N)

    R = mock_construct_R(rel_info)

    demeaned_geno = mock_demean(rel_info, G)
    demeaned_pheno = mock_demean(rel_info, pheno.reshape(1,-1))

    omega = mock_calc_omega(fam_df=fam_df, N=N, rel_df=rel_df, unresidualized_phenotypes=pheno)

    # The covariance of the residualized phenotypes, whose pseudo-inverse weights the regression.
    # Production inverts this block by block over the relatedness components, which is exact
    # rather than approximate; taking one pinv over the whole matrix is a separate route to the
    # same answer.
    resid_pheno_cov = (R @ omega @ R.T).toarray() if sp.issparse(R) else R @ omega @ R.T
    gls_weights = np.linalg.pinv(resid_pheno_cov)

    betas, ses, _ = mock_gls_estimates(
        genotypes=demeaned_geno, gls_weights=gls_weights,
        weighted_phenotypes=gls_weights @ np.ravel(demeaned_pheno))

    return -betas, ses

#################################
def mock_snipar_imputation(rng: np.random.Generator, num_snps: int, neither_observed: np.ndarray,
                           frequency_range: tuple = (0.05, 0.45)):
    """Builds imputed parental values the way snipar builds them

    This runs the construction FORWARDS -- start from the parental alleles snipar managed to
    observe, fill the rest in with the population frequency -- so that GRMA's correction, which
    runs it backwards, can be checked against the alleles it ought to recover.  Writing it out
    independently rather than reusing the library's arithmetic is the point: agreement then means
    the inversion is right, where a restatement of the same expression would prove nothing.

    What snipar stores depends on how many parents it had, and the two forms count different
    alleles.  Both were confirmed against a file generated from snipar's own example data.

        neither parent genotyped: imputed_par_gts holds the AVERAGE of the two parents, so all
            four parental alleles are in scope and the frequency contribution is halved along
            with everything else

        one parent genotyped: imputed_par_gts holds only the OTHER parent's genotype.  Two of the
            k observed alleles belong to the genotyped parent and are not in it at all, so only
            the remaining k - 2 are, and nothing is halved

    Either way the result is a genotype on a 0 to 2 scale, which is worth checking: a model that
    produced something outside that range would not be describing snipar.

    Args:
        rng: Seeded generator, so that a failure can be reproduced
        num_snps: Number of SNPs to generate
        neither_observed: Per family, whether snipar had no genotyped parent to work from
        frequency_range: Bounds to draw allele frequencies from.  The default stays clear of 0
            and of 0.5 so that a correction which was dropped, halved or doubled shows up as a
            clear difference rather than a marginal one.  Anything testing behaviour that depends
            on the frequency itself -- resolving a palindromic SNP, say, where values near 0.5
            are the hard case -- will want to say so explicitly.

    Returns:
        A tuple (imputed_gts, num_obs_par_alleles, standard_f, expected_contribution) where the
        first three are what the file would hold and the last is what
        calculate_imputed_contribution() should return for them
    """

    num_families = len(neither_observed)

    standard_f = rng.uniform(frequency_range[0], frequency_range[1], num_snps)

    # Where neither parent was genotyped, snipar can see as few as two of the four parental
    # alleles.  Where one was, its two alleles are always seen, so the count starts at three.
    num_obs_par_alleles = np.where(neither_observed[:, None],
                                   rng.choice([2.0, 3.0, 4.0], size=(num_families, num_snps)),
                                   rng.choice([3.0, 4.0], size=(num_families, num_snps)))

    # How many of the alleles that went INTO the imputed value carried the counted allele: all
    # four for an averaged pair, but only the ones belonging to the ungenotyped parent otherwise
    alleles_in_scope = np.where(neither_observed[:, None],
                                num_obs_par_alleles, num_obs_par_alleles - 2.0)
    observed_alleles = rng.integers(0, alleles_in_scope.astype(int) + 1).astype(float)

    # The alleles snipar could not see contribute the population frequency instead
    divisor = np.where(neither_observed, 2.0, 1.0)[:, None]
    imputed_gts = (observed_alleles + standard_f[None, :] * (4.0 - num_obs_par_alleles)) / divisor

    # GRMA removes the frequency and, for an averaged pair, doubles what is left so that it counts
    # for both parents.  What should come back is the allele count it started from.
    expected_contribution = observed_alleles.T

    return imputed_gts, num_obs_par_alleles, standard_f, expected_contribution


##################################

