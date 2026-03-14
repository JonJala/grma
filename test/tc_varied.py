import numpy as np
import pandas as pd
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib_new as lib


NUM_SNPS = 10
BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : [f"rs{num+1:02d}" for num in range(NUM_SNPS)],
        bbf.BIM_CM_COL : [float(num) for num in range(NUM_SNPS)],
        bbf.BIM_BP_COL : [100*num for num in range(NUM_SNPS)],
        bbf.BIM_A1_COL : 'A',
        bbf.BIM_A2_COL : 'T'
    },
    columns=bbf.BIM_COLS
)


SNP_LIST = pd.DataFrame({bbf.BIM_RSID_COL : [f"rs{2 * num + 1:02d}" for num in range(NUM_SNPS >> 1)]})

SNP_FILTER = np.array([0, 2, 4, 6, 8])


FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        bbf.FAM_IID_COL : list(range(18)),
        bbf.FAM_IIDF_COL : 0,
        bbf.FAM_IIDM_COL : 0,
        bbf.FAM_SEX_COL : 0,
        bbf.FAM_PHENO_COL : np.arange(18, dtype=float) + 1.0
    },
    columns=bbf.FAM_COLS
)


M = len(BIM_DF)
N = len(FAM_DF)

G = np.array([[(i + j + 1) % 3  for j in range(N)] for i in range(M)], dtype=float)

REL_DF = pd.DataFrame(
    {
        lib.KING_FID1_COL : [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 5, 7, 7, 9],
        lib.KING_IID1_COL : [0, 0, 1, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 8, 8, 11, 12, 14, 14, 16],
        lib.KING_FID2_COL : [0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 3, 5, 6, 8, 9, 10],
        lib.KING_IID2_COL : [1, 2, 2, 4, 5, 6, 7, 5, 6, 7, 7, 9, 9, 9, 10, 12, 13, 15, 16, 17],
        lib.KING_KINSHIP_COL : 0.0, # Don't need right now
        lib.KING_REL_COL : [
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_FULLSIB,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_3RD,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_3RD,
                        lib.INF_2ND,
                        lib.INF_3RD,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_PARENT_OFFSPRING,
                        lib.INF_3RD,
                        lib.INF_3RD,
                        lib.INF_3RD,
                        lib.INF_2ND,
                        lib.INF_4TH,
                        lib.INF_4TH
                        ]
    },
    columns=lib.NEEDED_KING_COLS
)
    


# R Matrices
DEG_FS_R = np.zeros((18, 18))
DEG_FS_R[0:3, 0:3] = np.array(
    [
        [2.0/3.0, -1.0/3.0, -1.0/3.0],
        [-1.0/3.0, 2.0/3.0, -1.0/3.0],
        [-1.0/3.0, -1.0/3.0, 2.0/3.0]
    ]
)
DEG_FS_R[3:5, 3:5] = np.array(
    [
        [0.5, -0.5],
        [-0.5, 0.5]
    ]
)


DEG_PO_R = DEG_FS_R.copy()
DEG_PO_R[5:7, 3:7] = np.array(
    [
        [-1.0/3.0, -1.0/3.0, 2.0/3.0, 0.0],
        [-1.0/3.0, -1.0/3.0, 0.0, 2.0/3.0]
    ]
)
DEG_PO_R[7:10, 7:10] = np.array(
    [
        [0.5, 0.0, -0.5],
        [0.0, 0.5, -0.5],
        [-1.0/3.0, -1.0/3.0, 2.0/3.0]
    ]
)


DEG_2ND_R = DEG_PO_R.copy()
DEG_2ND_R[14:16, 14:16] = np.array(
    [
        [0.5, -0.5],
        [-0.5, 0.5]
    ]
)


DEG_3RD_R = DEG_2ND_R.copy()
DEG_3RD_R[10, 8:11] = np.array([-0.5, 0.0, 0.5])
DEG_3RD_R[11:14, 11:14] = np.array(
    [
        [0.5, -0.5, 0.0],
        [-1.0/3.0, 2.0/3.0, -1.0/3.0],
        [0.0, -0.5, 0.5]
    ]
)

DEG_4TH_R = DEG_3RD_R.copy()
DEG_4TH_R[16:18, 16:18] = np.array(
    [
        [0.5, -0.5],
        [-0.5, 0.5]
    ]
)



R_MATRICES = {
    lib.DEG_FULLSIB : DEG_FS_R,
    lib.DEG_PARENT_OFFSPRING : DEG_PO_R,
    lib.DEG_2ND : DEG_2ND_R,
    lib.DEG_3RD : DEG_3RD_R,
    lib.DEG_4TH : DEG_4TH_R
}



BASE_SE_MATRIX = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ],
    dtype=float
)

OMEGA = BASE_SE_MATRIX * np.outer(FAM_DF[bbf.FAM_PHENO_COL].to_numpy(),
                                  FAM_DF[bbf.FAM_PHENO_COL].to_numpy())


SE_MATRICES = {deg : sp.csr_array(R_MATRICES[deg] @ OMEGA @ R_MATRICES[deg].T)
               for deg in R_MATRICES.keys()
}

DEMEANED_P = {deg : r_matrix @ FAM_DF[bbf.FAM_PHENO_COL].to_numpy()
    for deg, r_matrix in R_MATRICES.items()}

DEMEANED_G = {deg : np.nan_to_num(G @ r_matrix.T) for deg, r_matrix in R_MATRICES.items()}
