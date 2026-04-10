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
SNP_FILTER = np.array([0, 2, 3])

SAMPLE_LIST = None

SAMPLE_FILTER = None

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

M = len(BIM_DF)
N = len(FAM_DF)

G = np.array([[(i + j * j + 1) % 3  for j in range(N)] for i in range(M)], dtype=float)

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
DEG_FS_R = np.array(
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

DEG_PO_R = DEG_FS_R.copy(); DEG_PO_R[2] = np.array([-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
DEG_2ND_R = DEG_PO_R.copy(); DEG_2ND_R[3] = np.array([-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
DEG_3RD_R = DEG_2ND_R.copy(); DEG_3RD_R[4] = np.array([-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
DEG_4TH_R = DEG_3RD_R.copy(); DEG_4TH_R[5] = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])


R_MATRICES = {
    lib.DEG_FULLSIB : sp.csr_array(DEG_FS_R),
    lib.DEG_PARENT_OFFSPRING : sp.csr_array(DEG_PO_R),
    lib.DEG_2ND : sp.csr_array(DEG_2ND_R),
    lib.DEG_3RD : sp.csr_array(DEG_3RD_R),
    lib.DEG_4TH : sp.csr_array(DEG_4TH_R)
}


DEMEANED_P = {deg : r_matrix @ FAM_DF[bbf.FAM_PHENO_COL].to_numpy() for deg, r_matrix in R_MATRICES.items()}
DEMEANED_G = {deg : np.nan_to_num(G @ r_matrix.T) for deg, r_matrix in R_MATRICES.items()}
