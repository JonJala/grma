"""
VARIED_UNUSABLE_IMPUTED: VARIED run in imputed mode with a snipar file that nobody in the analysis
can use, so parental residualization is a no-op.  Results must be exactly VARIED's (EQUIVALENT_TO).

The file has one family with an imputed row, but its only member is not in the fam file.  It also
lists IIDs 0, 1 and 2 under a family of their own that has no imputed row, with parents who are not
genotyped -- snipar's third population (notes §6.2), unusable without an observed parent.  The file
shares VARIED's SNPs, order and alleles, so every SNP aligns and VARIED's SNP list decides which are
kept; its one palindromic SNP (rs09) is kept because the non-palindromic ones show the files share a
strand.
"""

import numpy as np

import bedbimfam as bbf

from tc_varied import (FAM_DF, REL_DF, BIM_DF, G, SNP_LIST, RETAINED_RSIDS, R_MATRICES,
                       RESID_PHENOS, RESID_GENOS)


_NUM_SNPS = len(BIM_DF)

IMPUTED_FILE = dict(
    imputed_gts=np.ones((1, _NUM_SNPS)),
    num_obs_par_alleles=np.full((1, _NUM_SNPS), 4.0),
    standard_f=np.full(_NUM_SNPS, 0.3),
    parental_status=np.array([[False, False, False]]),
    pedigree=np.array(
        [["ghost_family", "ghost_child", "ghost_dad", "ghost_mum", "False", "False"]] +
        [["varied_sibs", str(iid), "absent_dad_0", "absent_mum_0", "False", "False"]
         for iid in (0, 1, 2)]),
    families=np.array(["ghost_family"]),
    bim_values=np.array([["1", r, "0", str(bp), a1, a2] for r, bp, a1, a2 in zip(
        BIM_DF[bbf.BIM_RSID_COL], BIM_DF[bbf.BIM_BP_COL],
        BIM_DF[bbf.BIM_A1_COL], BIM_DF[bbf.BIM_A2_COL])]),
    chunk_snps=2)

AFFECTED = []

PARENTAL_MEANS = {}

EQUIVALENT_TO = "VARIED"
