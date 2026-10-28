"""
VARIED_ABSENT_PARENTS: VARIED with parent columns that name people who are not in the fam file, so
no one has two genotyped parents and parental residualization is a no-op.  Results must be exactly
VARIED's (EQUIVALENT_TO).

Three patterns, each of which must leave a sample unaffected in fam file trio mode:

    IIDs 0, 1, 2    both parents named, neither in the fam file
    IID 7           only a father named, and absent
    IID 11          only a mother named, and absent
"""

import bedbimfam as bbf

import tc_varied as _base
from tc_varied import (REL_DF, BIM_DF, G, SNP_LIST, RETAINED_RSIDS, R_MATRICES, RESID_PHENOS,
                       RESID_GENOS)


_NAMED = {0 : ("absent_dad_0", "absent_mum_0"), 1 : ("absent_dad_0", "absent_mum_0"),
          2 : ("absent_dad_0", "absent_mum_0"), 7 : ("absent_dad_2", "0"),
          11 : ("0", "absent_mum_4")}

FAM_DF = _base.FAM_DF.copy()
FAM_DF[bbf.FAM_IIDF_COL] = [_NAMED.get(iid, ("0", "0"))[0] for iid in FAM_DF[bbf.FAM_IID_COL]]
FAM_DF[bbf.FAM_IIDM_COL] = [_NAMED.get(iid, ("0", "0"))[1] for iid in FAM_DF[bbf.FAM_IID_COL]]

AFFECTED = []

EQUIVALENT_TO = "VARIED"
