"""
TWO_FAMILY_TRIOS: TWO_FAMILY's data with its parent columns filled in, so that all six children are
residualized against their parents in fam file trio mode.  Same KING data, genotypes and phenotypes
(imported from tc_2fam.py); only the fam file's parent and sex columns differ.

Side by side with TWO_FAMILY, this separates what parental residualization changes from everything
else.  The R matrices below are hand-derived and checked against the mocks by
TestCaseSelfConsistency; everything else is derived by the mocks.
"""

import numpy as np
import scipy.sparse as sp

import bedbimfam as bbf
import grma_lib as lib

import tc_2fam as _base
from tc_2fam import REL_DF, BIM_DF, G


# Samples, in fam order: parents 1, 2 (family 0) and 3, 4 (family 1), then children 5, 6, 7 of 1 and 2,
# and 8, 9, 10 of 3 and 4
_PARENTS = {5 : (1, 2), 6 : (1, 2), 7 : (1, 2), 8 : (3, 4), 9 : (3, 4), 10 : (3, 4)}

FAM_DF = _base.FAM_DF.copy()
FAM_DF[bbf.FAM_IIDF_COL] = [_PARENTS.get(iid, (0, 0))[0] for iid in FAM_DF[bbf.FAM_IID_COL]]
FAM_DF[bbf.FAM_IIDM_COL] = [_PARENTS.get(iid, (0, 0))[1] for iid in FAM_DF[bbf.FAM_IID_COL]]
FAM_DF[bbf.FAM_SEX_COL] = [{1 : 1, 3 : 1, 2 : 2, 4 : 2}.get(iid, 0) for iid in FAM_DF[bbf.FAM_IID_COL]]


# Stated facts and intermediates =================================================================

AFFECTED = [str(iid) for iid in _PARENTS]

# Degree 0: every child is affected, so passes through unchanged (an identity row), and the parents
# have no full-sib relatives, so they are not in the regression (a zero row).
_DEG_FS_R = np.diag([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Degree 1 and up: each parent is centered on themselves and their three children -- at the
# children's DIFFERENCED values, since the children's genotypes have had their parental means
# subtracted first.  The children still pass through unchanged.
_DEG_PO_R = _DEG_FS_R.copy()
_DEG_PO_R[0, [0, 4, 5, 6]] = [0.75, -0.25, -0.25, -0.25]
_DEG_PO_R[1, [1, 4, 5, 6]] = [0.75, -0.25, -0.25, -0.25]
_DEG_PO_R[2, [2, 7, 8, 9]] = [0.75, -0.25, -0.25, -0.25]
_DEG_PO_R[3, [3, 7, 8, 9]] = [0.75, -0.25, -0.25, -0.25]

# Nobody's closest relative is 2nd or 3rd degree, so degrees 2 and 3 are the same as degree 1
R_MATRICES = {
    lib.DEG_FULLSIB : sp.csr_array(_DEG_FS_R),
    lib.DEG_PARENT_OFFSPRING : sp.csr_array(_DEG_PO_R),
    lib.DEG_2ND : sp.csr_array(_DEG_PO_R),
    lib.DEG_3RD : sp.csr_array(_DEG_PO_R),
}

# Phenotypes are centered before residualization (notes §13.5).  An affected child's row is an
# identity row, so its centered phenotype passes straight through.
_PHENOS = FAM_DF[bbf.FAM_PHENO_COL].to_numpy(dtype=float)
RESID_PHENOS = {deg : r_matrix @ (_PHENOS - _PHENOS.mean()) for deg, r_matrix in R_MATRICES.items()}
