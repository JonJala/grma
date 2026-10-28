"""
VARIED_OPTIONS: VARIED's families and genotypes, run with every option that filters or adjusts the
samples -- the one case whose expected values pass through a phenotype file, covariate regression
and sample filtering end to end.  Replaces tc_allopt, which exercised the same options but could only
check which rsids came out.

Each dropped sample is removed by a different mechanism:

    IIDs 5, 13    absent from the phenotype file
    IID 9         missing a covariate value
    IIDs 2, 16    left out of the ID list

leaving 13 of the 18.  The phenotype file's values differ from the fam file's, so the fam file's must
be overridden; they are residualized on two non-collinear covariates plus an intercept.  VARIED's SNP
list, with blocks of 3, puts the retained SNPs in different blocks.

No parent columns, so parental residualization is a no-op (AFFECTED is empty).  Expected values are
derived by the mocks, except ANALYSIS_IIDS, stated from the list above and checked against them.
"""

import numpy as np
import pandas as pd

import bedbimfam as bbf

from tc_varied import FAM_DF, REL_DF, BIM_DF, G, SNP_LIST, RETAINED_RSIDS


_IDS = FAM_DF[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]]

# Chosen by a seeded search (seed 389 of a family-effect-plus-noise model, rounded to one decimal) so
# that omega is positive definite WITHOUT shrinkage (smallest eigenvalue 0.80 x variance) and
# R omega R' is well-conditioned (worst condition number 4.1).  See notes §7.4j for why that matters.
_PHENOS = [-2.7, -1.2, 1.7, -1.4, -0.1, 1.4, -1.7, -1.1, 2.3, 2.8, -2.4, 1.4, 1.1, 2.0, 0.8, 0.2,
           -1.5, -0.2]

PHENO_DF = _IDS.copy()
PHENO_DF[bbf.FAM_PHENO_COL] = _PHENOS
PHENO_DF = PHENO_DF[~FAM_DF[bbf.FAM_IID_COL].isin([5, 13])].reset_index(drop=True)

# From a seeded generator (seed 7); PC1 is missing for IID 9
COVAR_DF = _IDS.copy()
COVAR_DF["AGE"] = [51.0, 65.0, 59.0, 31.0, 35.0, 64.0, 20.0, 61.0, 60.0, 43.0, 35.0, 34.0, 33.0, 42.0,
                   45.0, 48.0, 70.0, 60.0]
COVAR_DF["PC1"] = [-1.9, -1.29, -1.84, -0.24, -1.27, 0.27, 0.16, -0.19, -2.52, np.nan, -0.05, 0.11,
                   -1.53, -0.48, -0.98, -0.81, 1.06, -0.81]

ID_LIST = _IDS[~FAM_DF[bbf.FAM_IID_COL].isin([2, 16])].reset_index(drop=True)

SNPS_PER_BLOCK = 3


# Stated facts and intermediates =================================================================

AFFECTED = []

ANALYSIS_IIDS = [str(iid) for iid in FAM_DF[bbf.FAM_IID_COL] if iid not in (2, 5, 9, 13, 16)]
