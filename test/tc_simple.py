"""
SIMPLE: sample 0 is related to each of six others at a different degree -- full sib, parent-offspring,
2nd, 3rd, 4th, and one listed as unrelated -- so each degree adds exactly one row to R.  A SNP list
keeps three of the five SNPs.

No parent columns, so parental residualization is a no-op (AFFECTED is empty).  R_MATRICES and the
residualized values below are hand-derived; TestCaseSelfConsistency checks them against the mocks.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib as lib


BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
        bbf.BIM_CM_COL : [1.0, 2.0, 3.0, 4.0, 5.0],
        bbf.BIM_BP_COL : [10, 20, 30, 40, 50],
        bbf.BIM_A1_COL : 'A',
        bbf.BIM_A2_COL : 'T'
    },
    columns=bbf.BIM_COLS
)

SNP_LIST = pd.DataFrame({bbf.BIM_RSID_COL : ['rs1', 'rs3', 'rs4']})
_SNP_FILTER = np.array([0, 2, 3])

FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : list(range(7)),
        bbf.FAM_IID_COL : list(range(7)),
        bbf.FAM_IIDF_COL : 0,
        bbf.FAM_IIDM_COL : 0,
        bbf.FAM_SEX_COL : 0,
        bbf.FAM_PHENO_COL : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    },
    columns=bbf.FAM_COLS
)

_M = len(BIM_DF)
_N = len(FAM_DF)

G = np.array([[(i + j * j + 1) % 3  for j in range(_N)] for i in range(_M)], dtype=float)

REL_DF = pd.DataFrame(
    {
        lib.KING_FID1_COL : [0, 0, 0, 0, 0, 0],
        lib.KING_IID1_COL : [0, 0, 0, 0, 0, 0],
        lib.KING_FID2_COL : [1, 2, 3, 4, 5, 6],
        lib.KING_IID2_COL : [1, 2, 3, 4, 5, 6],
        lib.KING_KINSHIP_COL : 0.0, # Don't need right now
        lib.KING_REL_COL : [
                        lib.INF_FULLSIB,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_2ND,
                        lib.INF_3RD,
                        lib.INF_4TH,
                        lib.INF_UNRELATED  # Not in an actual KING file, but useful to include here
                        ]
    },
    columns=lib.NEEDED_KING_COLS
)

    

# R Matrices
_DEG_FS_R = np.array(
    [
        [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
)

_DEG_PO_R = _DEG_FS_R.copy(); _DEG_PO_R[2] = np.array([-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
_DEG_2ND_R = _DEG_PO_R.copy(); _DEG_2ND_R[3] = np.array([-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
_DEG_3RD_R = _DEG_2ND_R.copy(); _DEG_3RD_R[4] = np.array([-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])


R_MATRICES = {
    lib.DEG_FULLSIB : sp.csr_array(_DEG_FS_R),
    lib.DEG_PARENT_OFFSPRING : sp.csr_array(_DEG_PO_R),
    lib.DEG_2ND : sp.csr_array(_DEG_2ND_R),
    lib.DEG_3RD : sp.csr_array(_DEG_3RD_R)
}


# Stated facts and intermediates =================================================================

AFFECTED = []

RETAINED_RSIDS = BIM_DF[bbf.BIM_RSID_COL].iloc[_SNP_FILTER].tolist()

# Phenotypes are centered before residualization (notes §13.5).  With nobody affected every row of R
# sums to zero or is empty, so this equals R applied to the raw phenotypes, as it was first written.
_PHENOS = FAM_DF[bbf.FAM_PHENO_COL].to_numpy(dtype=float)
RESID_PHENOS = {deg : r_matrix @ (_PHENOS - _PHENOS.mean()) for deg, r_matrix in R_MATRICES.items()}

RESID_GENOS = {deg : np.nan_to_num(G @ r_matrix.T)[_SNP_FILTER] for deg, r_matrix in R_MATRICES.items()}
