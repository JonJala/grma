"""
Parental residualization, imputed mode: snipar's imputed parental genotypes, over the same
individuals, genotypes and phenotypes as tc_parental.py.  See that module for the families and
which of them each mode can use.

    k2a, k2b    one parent genotyped (fa2) plus imputed row "B", in the one-observed form
    k3a, k3b    neither genotyped, imputed row "C", in the averaged form
    k4a, k4b    neither genotyped and no imputed row: snipar's third population, NOT usable

The imputed file has a swapped-allele SNP (rs4, flipped), lacks rs7 (dropped), keeps the palindromic
rs6 (the files share a strand), and has an extra SNP rs99 mid-file.  Its chunk width of 2 against
blocks of 3 makes the reader cross chunk boundaries, including mid-block.
"""

import tc_parental as _base
from tc_parental import FAM_DF, REL_DF, BIM_DF, G, SNPS_PER_BLOCK


IMPUTED_FILE = _base.imputed_file_arrays()

AFFECTED = ["fa1", "k1a", "k1b", "k1c", "k2a", "k2b", "k3a", "k3b"]

# From the allele counts the imputed values were built from, not from inverting the stored values
PARENTAL_MEANS = _base.expected_parental_means("imputed")

RETAINED_RSIDS = _base.imputed_retained_rsids()

# The retained SNPs sit at file positions 0, 1, 2, 4, 5, 6, which at a chunk width of 2 span four
# chunks.  Reading each exactly once confirms the cache crossed chunk boundaries, including
# mid-block, rather than the case being too small to need to.
EXPECTED_LOG_COUNTS = {"Read imputed parental SNPs": 4}
