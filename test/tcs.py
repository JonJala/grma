"""
The case registry: one format for every end-to-end test case, and one harness to run it.

A case is a module describing one complete GRMA run.  Its PUBLIC upper-case names must all come from
the format below -- anything else is rejected by TestCaseFormat, so that a misspelt name (BETA for
BETAS, say) is caught rather than silently ignored.  Implementation details belong in names with a
leading underscore, and helper functions in lower-case names; neither is examined.

Format
======

Inputs, required
    FAM_DF          the fam file, as a DataFrame with bedbimfam.FAM_COLS
    REL_DF          KING relatedness pairs, with grma_lib.NEEDED_KING_COLS
    BIM_DF          the bim file, with bedbimfam.BIM_COLS
    G               (snps, samples) genotypes for every SNP of BIM_DF and every sample of FAM_DF

Inputs, optional (absent means unused)
    PHENO_DF, COVAR_DF, ID_LIST, SNP_LIST, SNPS_PER_BLOCK
    IMPUTED_FILE    keyword arguments for snipar_hdf5.write_snipar_file().  Present: the harness
                    writes the file to a temporary directory and the run is in imputed mode.
                    Absent: no file, and the run is in fam file trio mode.

Stated facts -- what the case is built to have
    AFFECTED        IIDs residualized against their parents.  [] for a case where parental
                    residualization is a no-op.  Required.
    PARENTAL_MEANS  IID -> (snps,) value to subtract, over every SNP of BIM_DF, from RAW genotypes.
                    Required in imputed mode, where it comes from the truth the imputed values
                    were built from.  Derived from the fam file's parents otherwise.
    RETAINED_RSIDS  The SNPs that survive filtering and alignment, in bim order.  Required in
                    imputed mode; derived from SNP_LIST otherwise.

Intermediates, optional -- stated where known (hand-derived, or supplied with a reference case),
otherwise derived by the mocks in helper.py.  Matrices and vectors are over the ANALYSIS sample, in
fam file order, after filtering.
    ANALYSIS_IIDS   IIDs kept for analysis
    PHENOTYPES      their final phenotypes
    OMEGA           the phenotypic covariance, after shrinkage, (N, N)
    R_MATRICES      degree -> R, (N, N)
    RESID_PHENOS    degree -> R @ phenotypes, (N,)
    GLS_WEIGHTS     degree -> pinv(R omega R'), (N, N)
    RESID_GENOS     degree -> residualized genotypes, (retained snps, N), NaN set to zero
    BETAS           degree -> betas AS REPORTED, i.e. for A2, (retained snps,)
    SES             degree -> standard errors, (retained snps,)
    A per-degree name may state only some degrees; the rest are derived.

Meta, optional
    DEGREES               the relatedness degrees to run, default all
    TOLERANCE             absolute tolerance for comparisons, default 1e-6.  A reference case
                          rounded to 4 decimals wants about 1e-4.
    XFAIL                 dict(reason=..., raises=...) for a case that hits a known bug, applied
                          strictly.  Optionally applies_to=(kinds...) to limit it to some of the
                          tests that run production on cases -- see TEST_KINDS -- since a bug need
                          not break every property.  Default: all of them.  xfails are for known
                          bugs only, each tied to an item in notes §16 (notes §15.6).
    EQUIVALENT_TO         the key of another case whose rsids, betas and SEs this one must match
                          exactly (to about 1e-12), e.g. a case run a different way that should
                          make no difference
    EXPECTED_LOG_COUNTS   log message substring -> exact number of occurrences in a DEBUG-level
                          run, for confirming the case exercises a code path at all

Variants
========

A variant that differs from a case in a name or two is registered with overrides rather than given a
module of its own.  Registry entries are Case(module, NAME=value, ...).
"""

import functools
import os
import sys

import numpy as np
import pandas as pd
import pytest

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma_lib as lib
import snipar_hdf5

import helper
import tc_2fam
import tc_2fam_trios
import tc_parental
import tc_parental_imputed
import tc_pt
import tc_simple
import tc_varied
import tc_varied_absent_parents
import tc_varied_options
import tc_varied_unusable_imputed


# Format ===========================================================================================

INPUTS_REQUIRED = ("FAM_DF", "REL_DF", "BIM_DF", "G")
INPUTS_OPTIONAL = ("PHENO_DF", "COVAR_DF", "ID_LIST", "SNP_LIST", "SNPS_PER_BLOCK",
                   "IMPUTED_FILE")
FACTS = ("AFFECTED", "PARENTAL_MEANS", "RETAINED_RSIDS")
INTERMEDIATES = ("ANALYSIS_IIDS", "PHENOTYPES", "OMEGA")
PER_DEGREE = ("R_MATRICES", "RESID_PHENOS", "GLS_WEIGHTS", "RESID_GENOS", "BETAS", "SES")
META = ("DEGREES", "TOLERANCE", "XFAIL", "EXPECTED_LOG_COUNTS", "EQUIVALENT_TO")

FORMAT_NAMES = INPUTS_REQUIRED + INPUTS_OPTIONAL + FACTS + INTERMEDIATES + PER_DEGREE + META

ALL_DEGREES = tuple(range(lib.MAX_GRMA_RELATEDNESS + 1))
DEFAULT_TOLERANCE = 1e-6


class Case:
    """A case module, optionally with some of its names overridden"""

    def __init__(self, module, **overrides):
        unknown = set(overrides) - set(FORMAT_NAMES)
        if unknown:
            raise ValueError(f"Overrides {sorted(unknown)} are not format names")
        self.module = module
        self.overrides = overrides

    def __getitem__(self, name):
        """The value a case states for a name, or None"""
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.module, name, None)


# Registry =========================================================================================

def _drop_sample(case_module, iid):
    fam = case_module.FAM_DF
    return fam[fam[bbf.FAM_IID_COL] != iid][[bbf.FAM_FID_COL, bbf.FAM_IID_COL]]


# The kinds of test that run production on cases, which an XFAIL may be limited to
TEST_KINDS = ("end_to_end", "equivalence", "phenotype_shift", "allele_swap", "rename_samples",
              "rename_snps", "add_singletons", "process_relatedness", "process_bim_file")


def _kinds_except(*excluded):
    return tuple(k for k in TEST_KINDS if k not in excluded)


# Only the end-to-end comparison is affected.  The tests that compare two runs of this case see the
# same wrong SNP's data in both, so they pass; marking them would turn those passes into strict
# XPASS failures.
_XFAIL_13_2 = dict(raises=AssertionError, applies_to=("end_to_end",),
                   reason="notes §13.2: the chunk cache assumes the imputed file lists SNPs in the "
                          "bim's order.  Once a guard lands this case should expect its error "
                          "instead; once out-of-order files are supported it should simply pass.")

# Adding singletons changes omega's variance, and with anyone affected the centering mean too, so
# results change for every case until §13.6 is fixed.  Applied by the singleton test to any case
# without an XFAIL of its own covering it.
XFAIL_13_6 = dict(raises=AssertionError,
                  reason="notes §13.6: omega's variance, and the phenotype centering mean, are taken "
                         "over every analysis sample rather than only those in the regression")

KEY_SIMPLE = "SIMPLE"
KEY_2FAM = "TWO_FAMILY"
KEY_2FAM_TRIOS = "TWO_FAMILY_TRIOS"
KEY_VARIED = "VARIED"
KEY_PT = "PT"
KEY_VARIED_OPTIONS = "VARIED_OPTIONS"

REGISTRY = {
    KEY_SIMPLE : Case(tc_simple),
    KEY_2FAM : Case(tc_2fam),
    KEY_2FAM_TRIOS : Case(tc_2fam_trios),
    KEY_VARIED : Case(tc_varied),
    KEY_PT : Case(tc_pt),
    KEY_VARIED_OPTIONS : Case(tc_varied_options),
    "VARIED_ABSENT_PARENTS" : Case(tc_varied_absent_parents),
    "VARIED_UNUSABLE_IMPUTED" : Case(tc_varied_unusable_imputed),

    # One SNP per block, so that the five SNPs VARIED's SNP list drops each leave a block with
    # nothing in it.  Covers process_genotypes() skipping such a block, which only the deleted
    # tc_allopt test exercised before (notes §15.5).
    "VARIED_ONE_SNP_PER_BLOCK" : Case(tc_varied, SNPS_PER_BLOCK=1, EQUIVALENT_TO=KEY_VARIED,
                                      EXPECTED_LOG_COUNTS={"All SNPs filtered out in this block": 5}),

    "PARENTAL_FAM" : Case(tc_parental),
    "PARENTAL_IMPUTED" : Case(tc_parental_imputed),

    # Dropping a sample must not change who is affected: X is unrelated and sits first in the fam
    # file, so every position shifts; mo1 is a parent, and a parent need not be in the analysis.
    "PARENTAL_FAM_DROP_X" : Case(tc_parental, ID_LIST=_drop_sample(tc_parental, "X")),
    "PARENTAL_FAM_DROP_PARENT" : Case(tc_parental, ID_LIST=_drop_sample(tc_parental, "mo1")),
    "PARENTAL_IMPUTED_DROP_X" : Case(tc_parental_imputed,
                                     ID_LIST=_drop_sample(tc_parental, "X")),
    "PARENTAL_IMPUTED_DROP_PARENT" : Case(tc_parental_imputed,
                                          ID_LIST=_drop_sample(tc_parental, "mo1")),

    # A cross-chunk swap in the imputed file's SNP order
    "PARENTAL_IMPUTED_OUT_OF_ORDER" : Case(
        tc_parental_imputed,
        IMPUTED_FILE=tc_parental.imputed_file_arrays(tc_parental.hdf5_rsids_out_of_order()),
        EXPECTED_LOG_COUNTS=None, XFAIL=_XFAIL_13_2),
}

KEYS = list(REGISTRY)


# Accessors ========================================================================================

def stated(key, name, degree=None):
    """What a case states for a name, or None.  For a per-degree name, the value at that degree."""

    value = REGISTRY[key][name]
    if name in PER_DEGREE and value is not None:
        return value.get(degree)
    return value


def get(key, name, degree=None):
    """The value of a name for a case.  An input or meta name is returned as the case states it, None
    if absent.  A fact or intermediate is returned as stated if the case states it, otherwise as
    derived by the mocks."""

    value = stated(key, name, degree)
    if value is not None or name in INPUTS_REQUIRED + INPUTS_OPTIONAL + META:
        return value
    return derived(key, name, degree)


def provenance(key, name, degree=None):
    return "stated" if stated(key, name, degree) is not None else "derived"


def is_imputed(key):
    return REGISTRY[key]["IMPUTED_FILE"] is not None


def degrees(key):
    return REGISTRY[key]["DEGREES"] or ALL_DEGREES


def tolerance(key):
    return REGISTRY[key]["TOLERANCE"] or DEFAULT_TOLERANCE


def derived(key, name, degree=None):
    """A name's value as the mock chain derives it, ignoring anything the case states for it"""

    if name in ("AFFECTED", "PARENTAL_MEANS") and is_imputed(key):
        raise ValueError(f"{name} cannot be derived in imputed mode; {key} must state it")
    if name == "RETAINED_RSIDS" and is_imputed(key):
        raise ValueError(f"RETAINED_RSIDS cannot be derived in imputed mode; {key} must state it")

    return _mock_chain(key, degree if name in PER_DEGREE else None)[name]


def derivable(key, name):
    """Whether the mocks can derive a name for this case, independently of what it states"""

    if name in ("AFFECTED", "PARENTAL_MEANS", "RETAINED_RSIDS"):
        return not is_imputed(key)
    return name in INTERMEDIATES + PER_DEGREE


@functools.lru_cache(maxsize=None)
def _mock_chain(key, degree):
    """Every derivable name for a case, at one degree (None for the degree-independent ones)

    Facts are taken as STATED where the case states them, since they are what the case is built to
    have.  Intermediates are never taken from the case: each is computed from the stage before.
    """

    case = REGISTRY[key]
    fam_df, rel_df, bim_df, G = case["FAM_DF"], case["REL_DF"], case["BIM_DF"], case["G"]

    analysis_df = helper.mock_analysis_sample(fam_df, id_list=case["ID_LIST"],
                                              pheno_df=case["PHENO_DF"], covar_df=case["COVAR_DF"])
    analysis_iids = analysis_df[bbf.FAM_IID_COL].astype(str).tolist()
    phenos = analysis_df[bbf.FAM_PHENO_COL].to_numpy(dtype=float)
    rel_df = helper.mock_restrict_rel_df(rel_df, analysis_df)

    out = {"ANALYSIS_IIDS": analysis_iids, "PHENOTYPES": phenos}

    if not is_imputed(key):
        out["AFFECTED"] = helper.mock_fam_trios(fam_df, analysis_df)
        rsids = bim_df[bbf.BIM_RSID_COL].astype(str).tolist()
        snp_list = case["SNP_LIST"]
        wanted = (set(rsids) if snp_list is None
                  else set(snp_list.iloc[:, 0].astype(str)))
        out["RETAINED_RSIDS"] = [r for r in rsids if r in wanted]

    affected = [a for a in (case["AFFECTED"] if case["AFFECTED"] is not None
                            else out["AFFECTED"]) if a in analysis_iids]
    if not is_imputed(key):
        out["PARENTAL_MEANS"] = helper.mock_fam_parental_means(fam_df, G, affected)
    parental_means = (case["PARENTAL_MEANS"] if case["PARENTAL_MEANS"] is not None
                      else out["PARENTAL_MEANS"])
    retained_rsids = (case["RETAINED_RSIDS"] if case["RETAINED_RSIDS"] is not None
                      else out["RETAINED_RSIDS"])
    retained = [bim_df[bbf.BIM_RSID_COL].astype(str).tolist().index(r) for r in retained_rsids]

    out["OMEGA"] = helper.mock_calc_omega(fam_df=analysis_df, N=len(analysis_df), rel_df=rel_df,
                                          unresidualized_phenotypes=phenos).toarray()
    if degree is None:
        return out

    rel_info = helper.mock_create_rel_info(rel_df, analysis_df, degree, len(analysis_df))
    R = helper.mock_R(rel_info, analysis_df, affected)
    weights = np.linalg.pinv(R @ out["OMEGA"] @ R.T)
    resid_phenos = R @ phenos
    resid_genos = helper.mock_resid_genos(G, fam_df, analysis_df, rel_info, affected,
                                          parental_means, retained)
    betas, ses, _ = helper.mock_gls_estimates(genotypes=resid_genos, gls_weights=weights,
                                              weighted_phenotypes=weights @ resid_phenos)

    out.update({"R_MATRICES": R, "RESID_PHENOS": resid_phenos, "GLS_WEIGHTS": weights,
                "RESID_GENOS": resid_genos, "BETAS": -betas, "SES": ses})
    return out


# Running a case ===================================================================================

def snp_filter(key):
    """The retained SNPs as positions in the bim file, or None where no SNP list filters them --
    the form process_bim_file() returns and get_residualized_genotype_data() takes"""

    if REGISTRY[key]["SNP_LIST"] is None and not is_imputed(key):
        return None
    rsids = REGISTRY[key]["BIM_DF"][bbf.BIM_RSID_COL].astype(str).tolist()
    return np.array([rsids.index(r) for r in get(key, "RETAINED_RSIDS")])


def sample_filter(key):
    """The analysis samples as positions in the fam file, or None where every sample is kept"""

    iids = REGISTRY[key]["FAM_DF"][bbf.FAM_IID_COL].astype(str).tolist()
    kept = get(key, "ANALYSIS_IIDS")
    return None if len(kept) == len(iids) else np.array([iids.index(i) for i in kept])


def write_imputed_file(key, directory):
    """Writes a case's imputed parental genotype file, returning its path, or None if it has none"""

    if not is_imputed(key):
        return None
    path = os.path.join(directory, key + snipar_hdf5.SNIPAR_FILE_SUFFIX)
    snipar_hdf5.write_snipar_file(path, **REGISTRY[key]["IMPUTED_FILE"])
    return path


def grma_kwargs(key, degree, directory):
    """Keyword arguments for grma_lib.grma() that run this case at this degree"""

    case = REGISTRY[key]
    kwargs = dict(rel_file=case["REL_DF"], bed_file=case["G"], bim_file=case["BIM_DF"],
                  fam_file=case["FAM_DF"], rel_degree=degree, pheno_file=case["PHENO_DF"],
                  covar_file=case["COVAR_DF"], id_list=case["ID_LIST"],
                  snp_list=case["SNP_LIST"],
                  imputed_parental_file=write_imputed_file(key, directory))
    if case["SNPS_PER_BLOCK"] is not None:
        kwargs["snps_per_block"] = case["SNPS_PER_BLOCK"]
    return kwargs


def params(kind, keys=None, default_xfail=None):
    """pytest parameters (key, degree) over every case and degree, for one kind of production test,
    with each case's XFAIL applied where it covers that kind

    Args:
        kind: One of TEST_KINDS
        keys: The cases to run, default all
        default_xfail: An xfail for a known bug that affects this kind of test, applied where the
            case's own XFAIL does not cover it -- so that each failure is pinned to the bug that
            actually causes it.  Either a dict, for every case and degree, or a function of
            (key, degree) returning a dict or None, for a bug that only some combinations hit.
    """

    if kind not in TEST_KINDS:
        raise ValueError(f"Unknown test kind {kind!r}; expected one of {TEST_KINDS}")

    out = []
    for key in (KEYS if keys is None else keys):
        xfail = dict(REGISTRY[key]["XFAIL"] or {})
        applies = xfail.pop("applies_to", TEST_KINDS)
        for d in degrees(key):
            mark = xfail if (xfail and kind in applies) else (
                default_xfail(key, d) if callable(default_xfail) else default_xfail)
            marks = [pytest.mark.xfail(strict=True, **mark)] if mark else []
            out.append(pytest.param(key, d, id=f"{key}-{d}", marks=marks))
    return out


def key_params(kind, keys=None):
    """pytest parameters (key,) over every case, for a test that does not depend on the degree,
    with each case's XFAIL applied where it covers that kind"""

    if kind not in TEST_KINDS:
        raise ValueError(f"Unknown test kind {kind!r}; expected one of {TEST_KINDS}")

    out = []
    for key in (KEYS if keys is None else keys):
        xfail = dict(REGISTRY[key]["XFAIL"] or {})
        applies = xfail.pop("applies_to", TEST_KINDS)
        marks = [pytest.mark.xfail(strict=True, **xfail)] if xfail and kind in applies else []
        out.append(pytest.param(key, id=key, marks=marks))
    return out


def stated_derivable_params():
    """pytest parameters (key, name, degree) for every stated value the mocks can also derive"""

    out = []
    for key in KEYS:
        for name in FACTS + INTERMEDIATES:
            if stated(key, name) is not None and derivable(key, name):
                out.append(pytest.param(key, name, None, id=f"{key}-{name}"))
        for name in PER_DEGREE:
            for d in degrees(key):
                if stated(key, name, d) is not None:
                    out.append(pytest.param(key, name, d, id=f"{key}-{name}-{d}"))
    return out


def format_problems(key):
    """Every way a case departs from the format, as readable strings; empty if none"""

    case = REGISTRY[key]
    problems = []

    public = [n for n in dir(case.module) if n.isupper() and not n.startswith("_")]
    problems += [f"{n} is not a format name (prefix it with _ if it is an implementation detail)"
                 for n in public if n not in FORMAT_NAMES]
    problems += [f"required input {n} is missing" for n in INPUTS_REQUIRED if case[n] is None]

    if case["AFFECTED"] is None:
        problems.append("AFFECTED is required (use [] where nobody is affected)")
    if is_imputed(key):
        problems += [f"{n} is required in imputed mode" for n in ("PARENTAL_MEANS", "RETAINED_RSIDS")
                     if case[n] is None]

    if case["G"] is not None and case["FAM_DF"] is not None and case["BIM_DF"] is not None:
        if case["G"].shape != (len(case["BIM_DF"]), len(case["FAM_DF"])):
            problems.append(f"G is {case['G'].shape}, expected (bim SNPs, fam samples)")

    iids = set(case["FAM_DF"][bbf.FAM_IID_COL].astype(str)) if case["FAM_DF"] is not None else set()
    for iid in (case["AFFECTED"] or []):
        if iid not in iids:
            problems.append(f"AFFECTED lists {iid}, who is not in the fam file")
    for iid in (case["PARENTAL_MEANS"] or {}):
        if iid not in (case["AFFECTED"] or []):
            problems.append(f"PARENTAL_MEANS has an entry for {iid}, who is not AFFECTED")

    equivalent = case["EQUIVALENT_TO"]
    if equivalent is not None and (equivalent == key or equivalent not in REGISTRY):
        problems.append(f"EQUIVALENT_TO names {equivalent!r}, which is not another registered case")

    for name in PER_DEGREE:
        value = case[name]
        if value is not None and not set(value) <= set(degrees(key)):
            problems.append(f"{name} states degrees {sorted(set(value) - set(degrees(key)))} "
                            f"that are not run")

    return problems


def coding_dependence_xfail(key, degree):
    """The xfail for notes §13.7, where a case at a degree hits it: results that depend on which
    allele is counted.  Two independent sources, each checked from the case itself:

      - an unaffected sample is centered on a group containing an affected one, so raw and
        differenced genotypes are mixed, and recoding leaves a constant that does not cancel (§8.2)
      - an imputed family had fewer than 4 parental alleles observed, so the unobserved ones count
        as zero in whichever coding is used (§6.2.3a)
    """

    mixed = False
    affected = set(get(key, "AFFECTED"))
    if affected:
        R = get(key, "R_MATRICES", degree)
        R = R.toarray() if hasattr(R, "toarray") else np.asarray(R)
        is_affected = np.array([i in affected for i in get(key, "ANALYSIS_IIDS")])
        mixed = bool(np.any(R[np.ix_(~is_affected, is_affected)] != 0.0))

    imputed = REGISTRY[key]["IMPUTED_FILE"]
    incomplete = imputed is not None and bool(np.any(imputed["num_obs_par_alleles"] < 4.0))

    if not (mixed or incomplete):
        return None
    sources = [s for s, hit in (("raw and differenced genotypes centered together (§8.2)", mixed),
                                ("parental alleles left unobserved (§6.2.3a)", incomplete)) if hit]
    return dict(raises=AssertionError,
                reason=f"notes §13.7: results depend on allele coding -- {'; '.join(sources)}")


def equivalent_keys():
    """The cases that state EQUIVALENT_TO"""
    return [k for k in KEYS if REGISTRY[k]["EQUIVALENT_TO"] is not None]


# Generated cases ==================================================================================

def _snipar_variants():
    """Two imputed-mode versions of every case with fam-file trios: one parent imputed, and both

    Each is built from the case's true genotypes by helper.mock_snipar_from_trios(), so it must give
    results identical to the case itself (EQUIVALENT_TO) -- the invariance of notes §15.3.  Cases
    already in imputed mode, with no one affected, or with an XFAIL are skipped.  (Cases that filter
    samples were skipped too until §13.1 was fixed.)
    """

    variants = {}
    for key, case in REGISTRY.items():
        if is_imputed(key) or not case["AFFECTED"] or case["XFAIL"]:
            continue
        for mode, suffix in (("one", "SNIPAR_ONE_PARENT_IMPUTED"),
                             ("both", "SNIPAR_BOTH_PARENTS_IMPUTED")):
            overrides = dict(case.overrides)
            overrides.update(
                IMPUTED_FILE=helper.mock_snipar_from_trios(case["FAM_DF"], case["G"],
                                                           case["BIM_DF"], case["AFFECTED"], mode),
                PARENTAL_MEANS=get(key, "PARENTAL_MEANS"),
                RETAINED_RSIDS=get(key, "RETAINED_RSIDS"),
                EQUIVALENT_TO=key)
            variants[f"{key}__{suffix}"] = Case(case.module, **overrides)
    return variants


REGISTRY.update(_snipar_variants())
KEYS = list(REGISTRY)

