import os
import sys

import numpy as np
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
import grma_lib

'''
This contains a straightforward, unscalable version of GRMA to use to test against in small enough
examples.

'''


# KING_REL_COL = "InfType"
# KING_FID1_COL = "FID1"
# KING_IID1_COL = "ID1"
# KING_FID2_COL = "FID2"
# KING_IID2_COL = "ID2"
# KING_KINSHIP_COL = "Kinship"
# REL_TO_DEG_MAP = {"FS": 0, "1": 1, "2": 2, "3": 3, "4": 4, "Pop": 5}
# rel_thresh = ["FS, "1", "2", "3", "4"] or TODO - a number > 0.0 and < 0.0625
# Assumes IDs are integers in the range of 0 to N (and that FID is unused)
def mock_create_rel_info(king_df: pd.DataFrame, rel_thresh: str):
    ids = sorted(set(king_df[grma_lib.KING_IID1_COL].tolist()).union(
                 set(king_df[grma_lib.KING_IID2_COL].tolist())))
    N = len(ids)
    rel_thresh_val = grma_lib.REL_TO_DEG_MAP[rel_thresh]

    # Find the closest relationship levels for everyone
    lowest_rel = [5] * N
    for index, row in king_df.iterrows():
        cur_rel = grma_lib.INF_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]
        id1 = row[grma_lib.KING_IID1_COL]
        id2 = row[grma_lib.KING_IID2_COL]

        if cur_rel < lowest_rel[id1]:
            lowest_rel[id1] = cur_rel
        if cur_rel < lowest_rel[id2]:
            lowest_rel[id2] = cur_rel


    # Construct the rel_info object
    rel_info = [[i] for i in range(N)]
    for index, row in king_df.iterrows():
        cur_rel = grma_lib.INF_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]
        id1 = row[grma_lib.KING_IID1_COL]
        id2 = row[grma_lib.KING_IID2_COL]


        if cur_rel == lowest_rel[id1] and cur_rel <= rel_thresh_val:
            rel_info[id1].append(id2)

        if cur_rel == lowest_rel[id2] and cur_rel <= rel_thresh_val:
            rel_info[id2].append(id1)


    # Sort the rel_info entries
    for i, rel in enumerate(rel_info):
        rel_info[i] = sorted(rel)


    return rel_info



def mock_calc_duplicates(rel_input: list):
    N = len(rel_input)
    dups = [False] * N

    for cur_rownum, cur_row in enumerate(rel_input):
        maybe_dup = all(rel_input[i] == cur_row for i in cur_row)
        if not maybe_dup:
            continue

        false_alarm = any(i in row for i in cur_row
                                   for row_num, row in enumerate(rel_input)
                                       if row_num not in cur_row)
        if false_alarm:
            continue

        dups[cur_rownum] = True

    return np.array(dups)


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

    return result


def mock_run_regressions(demeaned_G: np.ndarray, demeaned_P: np.ndarray):
    M, N = demeaned_G.shape

    betas = np.zeros(M)
    for snp in range(M):
        betas[snp] = np.linalg.lstsq(demeaned_G[snp].reshape((1,-1)).T, demeaned_P.T)[0].item()

    return betas


def mock_calc_se(demeaned_G: np.ndarray, demeaned_P: np.ndarray, R: np.ndarray):
    M, N = demeaned_G.shape

    RR = R @ R.T
    tRR = np.trace(RR)
    ESSR = np.sum(np.square(demeaned_P))

    ses = np.zeros(M, dtype=float)

    for snp in range(M):
        X = demeaned_G[snp].T
        XXinv = np.reciprocal(np.dot(X, X))

        XRRX = X.T @ RR @ X

        ses[snp] = np.sqrt(ESSR * XXinv * XRRX * XXinv / (tRR - XRRX * XXinv))

    return ses



def mock_grma(rel_input: list, G: np.ndarray, pheno: np.ndarray):

    R = mock_construct_R(rel_input)

    demeaned_geno = mock_demean(rel_input, G)
    demeaned_pheno = mock_demean(rel_input, pheno)

    betas = mock_run_regressions(demeaned_G=demeaned_geno, demeaned_P=demeaned_pheno)
    ses = mock_calc_se(demeaned_G=demeaned_geno, demeaned_P=demeaned_pheno, R=R)

    return betas, ses




##################################

