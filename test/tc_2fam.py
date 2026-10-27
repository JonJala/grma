import numpy as np
import pandas as pd
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib as lib



BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : ['rs1', 'rs2', 'rs3'],
        bbf.BIM_CM_COL : [2.0, 4.0, 6.0],
        bbf.BIM_BP_COL : [50, 100, 150],
        bbf.BIM_A1_COL : 'A',
        bbf.BIM_A2_COL : 'T'
    },
    columns=bbf.BIM_COLS
)

SNP_LIST = None

SNP_FILTER = None

SAMPLE_LIST = None

SAMPLE_FILTER = None

FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        bbf.FAM_IID_COL : list(range(10)),
        bbf.FAM_IIDF_COL : 0,
        bbf.FAM_IIDM_COL : 0,
        bbf.FAM_SEX_COL : 0,
        # Within-family correlation is positive and moderate (FS and PO both r = 0.44 by the
        # omega estimator), so omega is positive definite without any shrinkage.  The previous
        # values put every child of family 0 far above the mean and every child of family 1 far
        # below it, which made the full-sib covariance EXCEED the variance (r = 1.30).  Shrinkage
        # then left the sib blocks nearly rank-1 along exactly the direction R removes, so
        # R omega R' held nothing but floor-level noise.  See notes §7.4j.
        bbf.FAM_PHENO_COL : [3.5, 0.5, -5.0, 2.0, 0.0, 1.0, 3.0, -2.5, -1.5, -3.5]
    },
    columns=bbf.FAM_COLS
)

M = len(BIM_DF)
N = len(FAM_DF)

G = np.array([[(i + j) % 3  for j in range(N)] for i in range(M)], dtype=float)




REL_DF = pd.DataFrame(
    {
        lib.KING_FID1_COL : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        lib.KING_IID1_COL : [0, 0, 0, 1, 1, 1, 4, 4, 5, 2, 2, 2, 3, 3, 3, 7, 7, 8],
        lib.KING_FID2_COL : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        lib.KING_IID2_COL : [4, 5, 6, 4, 5, 6, 5, 6, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9],
        lib.KING_KINSHIP_COL : 0.0, # Don't need right now
        lib.KING_REL_COL : [
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB
                        ]
    },
    columns=lib.NEEDED_KING_COLS
)
    


# R Matrices
DEG_FS_R = np.zeros((10, 10))
DEG_FS_R[4:10] = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 2.0/3.0, -1.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0/3.0, 2.0/3.0, -1.0/3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0/3.0, -1.0/3.0, 2.0/3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0/3.0, -1.0/3.0, -1.0/3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/3.0, 2.0/3.0, -1.0/3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0/3.0, -1.0/3.0, 2.0/3.0]
    ]
)

DEG_PO_R = DEG_FS_R.copy()
DEG_PO_R[0:4] = np.array(
    [
        [0.75, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0],
        [0.0, 0.75, 0.0, 0.0, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25],
        [0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25]
    ]
)

DEG_2ND_R = DEG_PO_R.copy()  # No change after PO
DEG_3RD_R = DEG_PO_R.copy()  # No change after PO
DEG_4TH_R = DEG_PO_R.copy()  # No change after PO



R_MATRICES = {
    lib.DEG_FULLSIB : sp.csr_array(DEG_FS_R),
    lib.DEG_PARENT_OFFSPRING : sp.csr_array(DEG_PO_R),
    lib.DEG_2ND : sp.csr_array(DEG_2ND_R),
    lib.DEG_3RD : sp.csr_array(DEG_3RD_R),
    lib.DEG_4TH : sp.csr_array(DEG_4TH_R)
}



DEMEANED_P = {deg : r_matrix @ FAM_DF[bbf.FAM_PHENO_COL].to_numpy() for deg, r_matrix in R_MATRICES.items()}
DEMEANED_G = {deg : np.nan_to_num(G @ r_matrix.T) for deg, r_matrix in R_MATRICES.items()}
