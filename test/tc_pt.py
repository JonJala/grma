"""
PT: the reference case from Patrick, who leads GRMA and devised the method.  16 samples in four
groups linked by full-sib, 2nd- and 3rd-degree pairs, with one missing genotype.

Its expected betas and SEs were computed under the PRE-GLS estimator (OLS with a sandwich SE; notes
§15.1), so they cannot match the current method and are kept under underscore names, where the
harness never asserts them.  The expectations used are therefore derived by the mocks.  Patrick will
supply a replacement case that includes parental residualization.

R_MATRICES were supplied printed to 8 decimal places, so they differ from the exact values by up to
about 2e-8 -- well inside the default tolerance.

No parent columns, so parental residualization is a no-op (AFFECTED is empty).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib as lib


_NUM_SNPS = 5
BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : [f"rs{num+1:02d}" for num in range(_NUM_SNPS)],
        bbf.BIM_CM_COL : [float(num) for num in range(_NUM_SNPS)],
        bbf.BIM_BP_COL : [100*num for num in range(_NUM_SNPS)],
        bbf.BIM_A1_COL : 'A',
        bbf.BIM_A2_COL : 'T'
    },
    columns=bbf.BIM_COLS
)

FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : list(range(16)),
        bbf.FAM_IID_COL : list(range(16)),
        bbf.FAM_IIDF_COL : 0,
        bbf.FAM_IIDM_COL : 0,
        bbf.FAM_SEX_COL : 0,
        bbf.FAM_PHENO_COL : [-0.341905000073181, -0.521146710987144, -0.457787817929638,
                       -0.649079974170937, 1.31901689906512, 0.0469918382036728, 0.683195789569862,
                       0.854897375543712, -0.251033113382805, -0.213594117514055, -1.37435039334038,
                       1.01505338064033, -0.741416243216773, -0.13940270416759, -0.251812536912985,
                       1.01079848109058]
    },
    columns=bbf.FAM_COLS
)


_M = len(BIM_DF)
_N = len(FAM_DF)

G = np.array([[np.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0],
                      [2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                      [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0]])


REL_DF = pd.DataFrame(
    {
        lib.KING_FID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
        lib.KING_IID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
        lib.KING_FID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
        lib.KING_IID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
        lib.KING_KINSHIP_COL : 0.0, # Don't need right now
        lib.KING_REL_COL : [
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_2ND,
            lib.INF_FULLSIB,
            lib.INF_3RD,
            lib.INF_FULLSIB,
            lib.INF_3RD,
            lib.INF_FULLSIB,
            lib.INF_3RD,
            lib.INF_FULLSIB,
            lib.INF_3RD,
            lib.INF_3RD,
            lib.INF_3RD,
            lib.INF_3RD,
            lib.INF_3RD
        ]
    },
    columns=lib.NEEDED_KING_COLS
)

_DEG_FS_R = np.array(
    [
        [ 0.,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. , -0.5,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. , -0.5,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0.5,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0.5,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ],
 [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
   0. ,  0. ,  0. ,  0. ]]
)

_DEG_PO_R = _DEG_FS_R.copy()

_DEG_2ND_R = np.array(
    [[ 0.66666667,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        , -0.33333333,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.66666667,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.66666667,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.66666667,  0.        ,
   0.        ,  0.        , -0.33333333,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.5       ,
   0.        ,  0.        ,  0.        , -0.5       ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.5       ,  0.        ,  0.        ,  0.        , -0.5       ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.5       ,  0.        ,  0.        ,  0.        ,
  -0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.5       ,  0.        ,  0.        ,
   0.        , -0.5       ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        , -0.5       ,
   0.        ,  0.        ,  0.        ,  0.5       ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
  -0.5       ,  0.        ,  0.        ,  0.        ,  0.5       ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        , -0.5       ,  0.        ,  0.        ,  0.        ,
   0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        , -0.5       ,  0.        ,  0.        ,
   0.        ,  0.5       ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ]]
)


_DEG_3RD_R = np.array(
    [[ 0.66666667,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        , -0.33333333,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.66666667,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.66666667,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.66666667,  0.        ,
   0.        ,  0.        , -0.33333333,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.5       ,
   0.        ,  0.        ,  0.        , -0.5       ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.5       ,  0.        ,  0.        ,  0.        , -0.5       ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.5       ,  0.        ,  0.        ,  0.        ,
  -0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        ,  0.5       ,  0.        ,  0.        ,
   0.        , -0.5       ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        , -0.5       ,
   0.        ,  0.        ,  0.        ,  0.5       ,  0.        ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
  -0.5       ,  0.        ,  0.        ,  0.        ,  0.5       ,
   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        , -0.5       ,  0.        ,  0.        ,  0.        ,
   0.5       ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        , -0.5       ,  0.        ,  0.        ,
   0.        ,  0.5       ,  0.        ,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        , -0.33333333,  0.        ,
   0.        ,  0.        ,  0.66666667,  0.        ,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        , -0.33333333,
   0.        ,  0.        ,  0.        ,  0.66666667,  0.        ,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
  -0.33333333,  0.        ,  0.        ,  0.        ,  0.66666667,
   0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   0.        ,  0.        , -0.33333333,  0.        ,  0.        ,
   0.        , -0.33333333,  0.        ,  0.        ,  0.        ,
   0.66666667]]

)


R_MATRICES = {
    lib.DEG_FULLSIB : sp.csr_array(_DEG_FS_R),
    lib.DEG_PARENT_OFFSPRING : sp.csr_array(_DEG_PO_R),
    lib.DEG_2ND : sp.csr_array(_DEG_2ND_R),
    lib.DEG_3RD : sp.csr_array(_DEG_3RD_R)
}


_PRE_GLS_BETAS = {
    lib.DEG_FULLSIB : -np.array([-0.2606, -0.1602, -0.2606, 0.4899, -1.1223]),
    lib.DEG_PARENT_OFFSPRING : -np.array([-0.2606, -0.1602, -0.2606, 0.4899, -1.1223]),
    lib.DEG_2ND : -np.array([-0.3724, 0.0515, -0.1684, 0.7521, -0.4618]),
    lib.DEG_3RD : -np.array([-0.4644, -0.1047, 0.1630, 0.4977, -0.4589]),
    lib.DEG_4TH : -np.array([-0.4644, -0.1047, 0.1630, 0.4977, -0.4589])
}

_PRE_GLS_SES = {
    lib.DEG_FULLSIB : np.array([0.2606, 0.1602, 0.2606, 0.5332, 0.8671]),
    lib.DEG_PARENT_OFFSPRING : np.array([0.2606, 0.1602, 0.2606, 0.5332, 0.8671]),
    lib.DEG_2ND : np.array([0.3724, 0.3421, 0.2207, 0.5843, 0.6962]),
    lib.DEG_3RD : np.array([0.3806, 0.2347, 0.2779, 0.3636, 0.6009]),
    lib.DEG_4TH : np.array([0.3806, 0.2347, 0.2779, 0.3636, 0.6009])
}


# Stated facts and intermediates =================================================================

AFFECTED = []

# Phenotypes are centered before residualization (notes §13.5).  With nobody affected every row of R
# sums to zero or is empty, so this equals R applied to the raw phenotypes, as it was first written.
_PHENOS = FAM_DF[bbf.FAM_PHENO_COL].to_numpy(dtype=float)
RESID_PHENOS = {deg : r_matrix @ (_PHENOS - _PHENOS.mean()) for deg, r_matrix in R_MATRICES.items()}

RESID_GENOS = {deg : np.nan_to_num(G @ r_matrix.T) for deg, r_matrix in R_MATRICES.items()}
