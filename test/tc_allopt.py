import numpy as np
import pandas as pd
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib as lib


NUM_SNPS = 8
BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : [f"rs{num+1}" for num in range(NUM_SNPS)],
        bbf.BIM_CM_COL : [float(num) for num in range(NUM_SNPS)],
        bbf.BIM_BP_COL : [100*num for num in range(NUM_SNPS)],
        bbf.BIM_A1_COL : 'A',
        bbf.BIM_A2_COL : 'T'
    },
    columns=bbf.BIM_COLS
)


SNP_LIST = pd.DataFrame({bbf.BIM_RSID_COL : ["rs2", "rs8"]})

SNP_FILTER = np.array([1, 7])


FAM_OFFSET = 1
PHENO_OFFSET = 2
COVAR_OFFSET = 4
SAMPLE_OFFSET = 8
NUM_BASE_SAMPLES = 16

LOWER_FIDS = [i + 2**5 for i in range(NUM_BASE_SAMPLES)]
LOWER_IIDS = [i for i in range(NUM_BASE_SAMPLES)]

FIDS = LOWER_FIDS + [FAM_OFFSET + PHENO_OFFSET + COVAR_OFFSET + SAMPLE_OFFSET + 2**5 + 2**6]
IIDS = LOWER_IIDS + [FAM_OFFSET + PHENO_OFFSET + COVAR_OFFSET + SAMPLE_OFFSET + 2**6]

NUM_SAMPLES = len(FIDS)


FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [i for i in FIDS if i & FAM_OFFSET],
        bbf.FAM_IID_COL : [i for i in IIDS if i & FAM_OFFSET],
        bbf.FAM_IIDF_COL : 0,
        bbf.FAM_IIDM_COL : 0,
        bbf.FAM_SEX_COL : 0,
        bbf.FAM_PHENO_COL : 0.0
    },
    columns=bbf.FAM_COLS
)

PHENO_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [i for i in FIDS if i & PHENO_OFFSET],
        bbf.FAM_IID_COL : [i for i in IIDS if i & PHENO_OFFSET],
        bbf.FAM_PHENO_COL : [1.0 * i for i in IIDS if i & PHENO_OFFSET]
    },
    columns=[bbf.FAM_FID_COL, bbf.FAM_IID_COL, bbf.FAM_PHENO_COL]
)

COVAR_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [i for i in FIDS if i & COVAR_OFFSET],
        bbf.FAM_IID_COL : [i for i in IIDS if i & COVAR_OFFSET],
        "COVAR_1" : [2.0 * i for i in IIDS if i & COVAR_OFFSET],
        "COVAR_2" : [1.0 * i for i in IIDS if i & COVAR_OFFSET]
    },
    columns=[bbf.FAM_FID_COL, bbf.FAM_IID_COL, "COVAR_1", "COVAR_2"]
)

SAMPLE_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : [i for i in FIDS if i & SAMPLE_OFFSET],
        bbf.FAM_IID_COL : [i for i in IIDS if i & SAMPLE_OFFSET],
    },
    columns=[bbf.FAM_FID_COL, bbf.FAM_IID_COL]
)

SAMPLE_FILTER = [15, 16]



M = len(BIM_DF)
N = len(FAM_DF)

G = np.ones((NUM_SNPS, NUM_SAMPLES), dtype=float)
G[:, -1] = 2.0

REL_DF = pd.DataFrame(
    {
        lib.KING_FID1_COL : [x for i, x in enumerate(FIDS) for _ in range(NUM_SAMPLES - 1 - i)],
        lib.KING_IID1_COL : [x for i, x in enumerate(IIDS) for _ in range(NUM_SAMPLES - 1 - i)],
        lib.KING_FID2_COL : [x for i in range(NUM_SAMPLES) for x in FIDS[i+1:]],
        lib.KING_IID2_COL : [x for i in range(NUM_SAMPLES) for x in IIDS[i+1:]],
        lib.KING_KINSHIP_COL : 0.0, # Don't need right now
        lib.KING_REL_COL : lib.INF_FULLSIB
    },
    columns=lib.NEEDED_KING_COLS
)
    


# R Matrices
# DEG_FS_R = np.identity(NUM_SAMPLES) - np.full((NUM_SAMPLES, NUM_SAMPLES), 1.0 / NUM_SAMPLES)
# DEG_PO_R = DEG_FS_R.copy()
# DEG_2ND_R = DEG_PO_R.copy()
# DEG_3RD_R = DEG_2ND_R.copy()
# DEG_4TH_R = DEG_3RD_R.copy()

# R_MATRICES = {
#     lib.DEG_FULLSIB : sp.csr_array(DEG_FS_R),
#     lib.DEG_PARENT_OFFSPRING : sp.csr_array(DEG_PO_R),
#     lib.DEG_2ND : sp.csr_array(DEG_2ND_R),
#     lib.DEG_3RD : sp.csr_array(DEG_3RD_R),
#     lib.DEG_4TH : sp.csr_array(DEG_4TH_R)
# }



# DEMEANED_P = {deg : r_matrix @ PHENO_DF[bbf.FAM_PHENO_COL].to_numpy()
#     for deg, r_matrix in R_MATRICES.items()}

# DEMEANED_G = {deg : np.nan_to_num(G @ r_matrix.T) for deg, r_matrix in R_MATRICES.items()}

SNPS_PER_BLOCK = 3
