"""
Parental residualization, fam file trio mode.  tc_parental_imputed.py is the imputed-mode case over
the SAME individuals, genotypes and phenotypes, and imports them from here.  See tcs.py for the
format and for the variants registered from both.

One set of individuals serves both modes.  Who is affected differs between them, and that
difference is itself part of what is tested:

    sample           fam file trio mode            imputed mode
    ------           ------------------            ------------
    X                unrelated, listed FIRST so that filtering it out shifts every position
    gf, gm           grandparents (founders)
    fa1              affected (gf, gm)             affected (gf, gm, both genotyped)
    mo1              founder
    k1a, k1b, k1c    affected (fa1, mo1)           affected (fa1, mo1, both genotyped)
    fa2              founder
    k2a, k2b         NOT a trio (mo2 not geno'd)   affected: fa2 genotyped + imputed row "B"
    k3a, k3b         NOT a trio                    affected: imputed row "C", neither genotyped
    k4a, k4b         NOT a trio                    NOT usable: no imputed row, no genotyped parent
    u1               singleton

fa1 is both affected and a parent, so his children must be differenced against his RAW genotypes,
not against his own Mendelian residual.

IIDs are strings throughout.  An integer IID 0 would collide with the fam file's missing-parent
code "0", which silently turns a trio into a non-trio.
"""

import numpy as np
import pandas as pd

import bedbimfam as bbf
import grma_lib as lib


# Samples ===========================================================================================

_IIDS = ["X", "gf", "gm", "fa1", "mo1", "k1a", "k1b", "k1c", "fa2", "k2a", "k2b",
        "k3a", "k3b", "k4a", "k4b", "u1"]
_FIDS = ["FX", "A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "U"]
_FATHERS = ["0", "0", "0", "gf", "0", "fa1", "fa1", "fa1", "0", "fa2", "fa2",
           "pa3", "pa3", "pa4", "pa4", "0"]
_MOTHERS = ["0", "0", "0", "gm", "0", "mo1", "mo1", "mo1", "0", "mo2", "mo2",
           "ma3", "ma3", "ma4", "ma4", "0"]
_SEXES = ["0", "1", "2", "1", "2", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0"]

# Chosen by a seeded search (seed 123 of a parents-plus-noise model, rounded to one decimal) so
# that omega is positive definite WITHOUT shrinkage (smallest eigenvalue 0.57 x variance) and
# R omega R' is well-conditioned (worst condition number 11.9) at every degree, in both modes,
# with and without each filter below.  See notes §7.4j for why that matters.
#
# The grandparent-grandchild correlation comes out near zero (-0.02) for every seed tried, not by
# chance: omega pairs each relationship both ways round and centers on the POOLED mean, so a
# difference between the two generations' means contributes -(a - b)^2 / 4.  See notes §13.3.
_PHENOS = [-1.5, -0.6, 1.9, -1.1, 0.3, 1.0, -1.2, 0.8, 1.4, 1.3, 3.0, -0.9, -0.4, -0.1, -3.1, 0.1]

_N_ORIG = len(_IIDS)
_POS = {iid: i for i, iid in enumerate(_IIDS)}


# Relatedness =======================================================================================

def _pairs(inf_type, pairs):
    fid = dict(zip(_IIDS, _FIDS))
    return [(fid[a], a, fid[b], b, inf_type) for a, b in pairs]

_K1 = ["k1a", "k1b", "k1c"]
_KING_ROWS = (
    _pairs(lib.INF_PARENT_OFFSPRING,
           [("gf", "fa1"), ("gm", "fa1")] + [(p, k) for p in ("fa1", "mo1") for k in _K1] +
           [("fa2", "k2a"), ("fa2", "k2b")]) +
    _pairs(lib.INF_FULLSIB,
           [("k1a", "k1b"), ("k1a", "k1c"), ("k1b", "k1c"),
            ("k2a", "k2b"), ("k3a", "k3b"), ("k4a", "k4b")]) +
    _pairs(lib.INF_2ND, [(g, k) for g in ("gf", "gm") for k in _K1]) +
    _pairs(lib.INF_3RD, [("k3a", "k4a")])  # Cousins, linking families C and D into one component
)

REL_DF = pd.DataFrame(_KING_ROWS, columns=[lib.KING_FID1_COL, lib.KING_IID1_COL,
                                           lib.KING_FID2_COL, lib.KING_IID2_COL, lib.KING_REL_COL])
REL_DF[lib.KING_KINSHIP_COL] = 0.0  # Not used
REL_DF = REL_DF[lib.NEEDED_KING_COLS]


# SNPs =============================================================================================

# rs4 has its alleles swapped in the imputed file, so must be flipped.  rs6 is palindromic, and is
# kept because every non-palindromic SNP shows the two files share a strand.  rs7 is absent from
# the imputed file, so is dropped in imputed mode only.
BIM_DF = pd.DataFrame(
    {
        bbf.BIM_CHR_COL : 1,
        bbf.BIM_RSID_COL : [f"rs{i}" for i in range(1, 8)],
        bbf.BIM_CM_COL : 0.0,
        bbf.BIM_BP_COL : [100 * i for i in range(1, 8)],
        bbf.BIM_A1_COL : ["A", "C", "G", "A", "C", "A", "G"],
        bbf.BIM_A2_COL : ["C", "G", "T", "G", "T", "T", "C"],
    },
    columns=bbf.BIM_COLS
)
_M_ORIG = len(BIM_DF)

_FLIPPED_RSIDS = ("rs4",)
_NOT_IN_IMPUTED_FILE_RSIDS = ("rs7",)

# The imputed file's own SNP order.  rs99 exists only there and must be ignored; sitting mid-file
# it also means a retained SNP's position in the file differs from its position in the bim.
_HDF5_RSIDS = ["rs1", "rs2", "rs3", "rs99", "rs4", "rs5", "rs6"]

# A cross-chunk swap (rs2/rs3 straddle the first chunk boundary at width 2).  See notes §13.2.
_HDF5_RSIDS_OUT_OF_ORDER = ["rs1", "rs3", "rs2", "rs99", "rs4", "rs5", "rs6"]

_CHUNK_SNPS = 2
SNPS_PER_BLOCK = 3


# Genotypes ========================================================================================

_rng = np.random.default_rng(seed=20261026)
G = _rng.integers(0, 3, size=(_M_ORIG, _N_ORIG)).astype(float)

# A missing grandparental genotype makes fa1's parental mean missing at that SNP, which must then
# propagate to fa1 and to everyone centered against him
G[4, _POS["gm"]] = np.nan


# Imputed parental genotypes =======================================================================

# Families in the imputed file, in snipar's lexicographic order.  "A" is absent because both of
# its parents were genotyped; "D" is absent because snipar could not impute for it.
_IMPUTED_FAMILIES = ["B", "C"]
_NEITHER_OBSERVED = np.array([False, True])

# Allele frequencies on a 1/64 grid, so that every imputed value is exactly representable in
# float16 and the stored values carry no rounding error into the comparison
_STANDARD_F = np.array([13, 22, 9, 30, 17, 26, 11]) / 64.0

# Built FORWARDS from known observed allele counts, as in helper.mock_snipar_imputation(), so that
# the expected parental means below come from the truth rather than from inverting the stored
# values.  These are in the IMPUTED FILE's allele coding and SNP order.
_NUM_OBS_PAR_ALLELES = np.where(_NEITHER_OBSERVED[:, None],
                               _rng.choice([2.0, 3.0, 4.0], size=(2, len(_HDF5_RSIDS))),
                               _rng.choice([3.0, 4.0], size=(2, len(_HDF5_RSIDS))))
_ALLELES_IN_SCOPE = np.where(_NEITHER_OBSERVED[:, None], _NUM_OBS_PAR_ALLELES,
                             _NUM_OBS_PAR_ALLELES - 2.0)
_OBSERVED_ALLELES = _rng.integers(0, _ALLELES_IN_SCOPE.astype(int) + 1).astype(float)
_DIVISOR = np.where(_NEITHER_OBSERVED, 2.0, 1.0)[:, None]
_IMPUTED_GTS = (_OBSERVED_ALLELES + _STANDARD_F[None, :] * (4.0 - _NUM_OBS_PAR_ALLELES)) / _DIVISOR


def imputed_file_arrays(hdf5_rsids=_HDF5_RSIDS):
    """The arguments to snipar_hdf5.write_snipar_file(), with the SNPs in the order given

    The per-SNP arrays above are defined in _HDF5_RSIDS order; this permutes them for any other.
    """

    order = [_HDF5_RSIDS.index(r) for r in hdf5_rsids]
    bim_by_rsid = {r: (str(bp), a1, a2) for r, bp, a1, a2 in zip(
        BIM_DF[bbf.BIM_RSID_COL], BIM_DF[bbf.BIM_BP_COL], BIM_DF[bbf.BIM_A1_COL],
        BIM_DF[bbf.BIM_A2_COL])}
    bim_by_rsid["rs99"] = ("950", "C", "T")

    bim_values = []
    for r in hdf5_rsids:
        bp, a1, a2 = bim_by_rsid[r]
        if r in _FLIPPED_RSIDS:
            a1, a2 = a2, a1
        # Columns as a real file gives them: Chr, id, morgans, coordinate, allele1, allele2
        bim_values.append(["1", r, "0", bp, a1, a2])

    pedigree = [
        # FID   IID    FATHER  MOTHER  has_father has_mother
        ["A0", "fa1", "gf", "gm", "True", "True"],
        *[["A", k, "fa1", "mo1", "True", "True"] for k in _K1],
        *[["B", k, "fa2", "mo2", "True", "False"] for k in ("k2a", "k2b")],
        *[["C", k, "pa3", "ma3", "False", "False"] for k in ("k3a", "k3b")],
        *[["D", k, "pa4", "ma4", "False", "False"] for k in ("k4a", "k4b")],
        # Founders, which no branch can use
        *[[f"_{i}", i, "0", "0", "False", "False"] for i in ("gf", "gm", "mo1", "fa2", "u1")],
    ]

    return dict(imputed_gts=_IMPUTED_GTS[:, order],
                num_obs_par_alleles=_NUM_OBS_PAR_ALLELES[:, order],
                standard_f=_STANDARD_F[order],
                parental_status=np.array([[True, False, True], [False, False, False]]),
                pedigree=np.array(pedigree),
                families=np.array(_IMPUTED_FAMILIES),
                bim_values=np.array(bim_values),
                chunk_snps=_CHUNK_SNPS)


# Shared with tc_parental_imputed =================================================================

def hdf5_rsids_out_of_order():
    """The imputed file's SNPs with a cross-chunk swap: rs2 and rs3 straddle the first chunk
    boundary at width 2.  See notes §13.2."""
    return list(_HDF5_RSIDS_OUT_OF_ORDER)


def imputed_retained_rsids():
    """The bim SNPs that align to the imputed file"""
    return [r for r in BIM_DF[bbf.BIM_RSID_COL] if r not in _NOT_IN_IMPUTED_FILE_RSIDS]


def _imputed_truth(family):
    """Observed parental allele count for a family, per BIM SNP, in GRMA's allele coding

    For a flipped SNP the imputed file counts the other allele, so of the alleles in scope the
    complement is what GRMA counts.  NaN for SNPs absent from the imputed file.
    """

    row = _IMPUTED_FAMILIES.index(family)
    truth = np.full(_M_ORIG, np.nan)
    for snp, rsid in enumerate(BIM_DF[bbf.BIM_RSID_COL]):
        if rsid not in _HDF5_RSIDS:
            continue
        col = _HDF5_RSIDS.index(rsid)
        observed = _OBSERVED_ALLELES[row, col]
        truth[snp] = (_ALLELES_IN_SCOPE[row, col] - observed if rsid in _FLIPPED_RSIDS
                      else observed)
    return truth


def expected_parental_means(mode):
    """What should be subtracted from each affected individual, per BIM SNP, in mode "fam" or
    "imputed"

    Computed from the RAW genotypes for everyone, so a multi-generation pedigree is differenced
    against unadjusted parental values.

    Returns:
        dict of IID -> (_M_ORIG,) array
    """

    both = lambda f, m: (G[:, _POS[f]] + G[:, _POS[m]]) / 2.0
    means = {"fa1": both("gf", "gm"), **{k: both("fa1", "mo1") for k in _K1}}

    if mode == "imputed":
        # One parent genotyped: half the observed parent, plus the missing parent's observed
        # alleles in full -- the transmitted allele is subtracted exactly (calculate_parental_means)
        for k in ("k2a", "k2b"):
            means[k] = (G[:, _POS["fa2"]] + _imputed_truth("B")) / 2.0
        # Neither genotyped: the observed alleles of both, halved
        for k in ("k3a", "k3b"):
            means[k] = _imputed_truth("C") / 2.0

    return means



# The case ========================================================================================

FAM_DF = pd.DataFrame(
    {
        bbf.FAM_FID_COL : _FIDS,
        bbf.FAM_IID_COL : _IIDS,
        bbf.FAM_IIDF_COL : _FATHERS,
        bbf.FAM_IIDM_COL : _MOTHERS,
        bbf.FAM_SEX_COL : _SEXES,
        bbf.FAM_PHENO_COL : _PHENOS,
    },
    columns=bbf.FAM_COLS
)

# Full trios only: k2a/k2b's mother, and both parents of families C and D, are not genotyped
AFFECTED = ["fa1", "k1a", "k1b", "k1c"]

# Stated even though the mocks can derive them in this mode, so that the fixture self-check
# compares the two
PARENTAL_MEANS = expected_parental_means("fam")
