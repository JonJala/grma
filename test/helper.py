import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
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
def mock_create_rel_info(king_df: pd.DataFrame, rel_thresh: str):
    ids = sorted(set(king_df[grma_lib.KING_IID1_COL].tolist()).union(
                 set(king_df[grma_lib.KING_IID2_COL].tolist())))
    N = len(ids)

    rel_thresh_val = grma.REL_TO_DEG_MAP[rel_thresh]

    # Find the closest relationship levels for everyone
    lowest_rel = [5] * N
    for index, row in king_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]
        id1 = row[grma_lib.KING_IID1_COL]
        id2 = row[grma_lib.KING_IID2_COL]

        if cur_rel < lowest_rel[id1]:
            lowest_rel[id1] = cur_rel
        if cur_rel < lowest_rel[id2]:
            lowest_rel[id2] = cur_rel


    # Construct the rel_info object and se_info objects
    rel_info = [[i] for i in range(N)]
    se_rel = np.identity(N, dtype=np.int8)
    for index, row in king_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]
        id1 = row[grma_lib.KING_IID1_COL]
        id2 = row[grma_lib.KING_IID2_COL]


        if cur_rel == lowest_rel[id1] and cur_rel <= rel_thresh_val:
            rel_info[id1].append(id2)

        if cur_rel == lowest_rel[id2] and cur_rel <= rel_thresh_val:
            rel_info[id2].append(id1)

        if cur_rel <= grma_lib.SE_RELATEDNESS:
            se_rel[id1, id2] = 1
            se_rel[id2, id1] = 1


    # Sort the rel_info entries
    for i, rel in enumerate(rel_info):
        rel_info[i] = sorted(rel)


    return rel_info, se_rel



def mock_shrink_offdiag(matrix: sp.csr_array, weight: float = 0.75):

    lam_min = spla.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
    if lam_min > 0.0:
        return matrix


    D = sp.diags(matrix.diagonal())
    off_diag = matrix.copy()
    off_diag.setdiag(0)
    off_diag.eliminate_zeros()

    alpha_min = 0.0
    alpha_max = 1.0

    num_iterations = 35
    for iter_num in range(num_iterations):
        alpha = weight * alpha_max + (1.0 - weight) * alpha_min

        result = D + alpha * off_diag
        lam_min = spla.eigsh(result, k=1, which="SA", return_eigenvectors=False)[0]
        if lam_min > 0:
            alpha_min = alpha
        else:
            alpha_max = alpha

    print(f"TEST: {alpha=}")

    lam_min = spla.eigsh(result, k=1, which="SA", return_eigenvectors=False)[0]
    if lam_min < 0:
        result = D + alpha_min * off_diag


    return result


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

    print(f"TEST: unshrunk omega=\n{omega.toarray()}")
    print(f"TEST: shrunk omega=\n{shrunk_omega.toarray()}")

    return shrunk_omega




def mock_construct_R(rel_input: list):
    N = len(rel_input)

    result = np.identity(N, dtype=float)
    for row_num, row in enumerate(rel_input):
        result[row_num, row] -= np.reciprocal(float(len(row)))

    return result


# Assume K x N (where K is likely M or 1 depending on whether this is G or the phenotypes)
def mock_demean(rel_input: list, arr: np.ndarray):
    result = arr.copy()
    
    for person_num, row in enumerate(rel_input):
        result[:, person_num] -= np.mean(arr[:, row], axis=1)

    result[np.isnan(result)] = 0.0

    return result


def mock_calc_xtx(G: np.ndarray):
    XtX = np.sum(np.square(G), axis=1)
    XtX[XtX == 0.0] = np.finfo(XtX.dtype).eps
    return XtX


def mock_run_regressions(demeaned_G: np.ndarray, demeaned_P: np.ndarray):
    M, N = demeaned_G.shape

    XtX = mock_calc_xtx(demeaned_G)

    betas = np.zeros(M)
    for snp in range(M):
        betas[snp] = np.linalg.lstsq(demeaned_G[snp].reshape((1,-1)).T, demeaned_P.T)[0].item()

    return betas


def mock_calc_ses(demeaned_G: np.ndarray, se_matrix: np.ndarray):
    M, N = demeaned_G.shape

    ses = np.zeros(M, dtype=float)


    for snp in range(M):
        X = demeaned_G[snp].T
        XtX = np.dot(X, X)
        if XtX == 0.0:
            XtX = np.finfo(XtX.dtype).eps
        XXinv = np.reciprocal(XtX)

        central_value = X.T @ se_matrix @ X

        ses[snp] = XXinv * np.sqrt(central_value)

    ses[ses == 0.0] = np.finfo(ses.dtype).eps

    return ses


def mock_calc_residuals(betas: np.ndarray, G: np.ndarray, pheno: np.ndarray) -> np.ndarray:
    M, N = G.shape
    residuals = np.zeros(N)

    residuals = -((G * betas.reshape((M, 1))) - pheno)

    return residuals


def mock_grma(rel_input: list, se_rel: np.ndarray, G: np.ndarray, pheno: np.ndarray):

    R = mock_construct_R(rel_input)

    demeaned_geno = mock_demean(rel_input, G)
    demeaned_pheno = mock_demean(rel_input, pheno)

    betas = mock_run_regressions(demeaned_G=demeaned_geno, demeaned_P=demeaned_pheno)
    residuals = mock_calc_residuals(betas=betas, G=G, pheno=pheno)
    ses = mock_calc_ses(demeaned_G=demeaned_geno, P=pheno, R=R,
                        se_rel=se_rel)

    return -betas, ses




##################################

