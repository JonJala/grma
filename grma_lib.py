"""
GRMA: planning and statistics.

Work happens in two stages:

    1. Planning, from everything except genotypes: the sample table, the relatedness graph, the
       groups, Omega and the four regressions.  Every input is validated here, so a problem is
       reported before any genotype is read rather than partway through a long run.
    2. Running: genotype blocks streamed through the finished plan.

So far: the sample table and each input's own checks; the relatedness graph and KING's checks;
duplicate samples; parents, from snipar's pedigree; the groups; and Omega, with the phenotypes
prepared for it; the SNPs to analyse, aligned to snipar's file; the four regressions; the covariance
of their estimates; their meta-analysis; and grma(), which runs it all: plan_analysis() then
run_analysis().  Reading the files is
grma_io's job.
"""

import dataclasses
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

import bedbimfam as bbf
import grma_io
import grma_util
import snipar_hdf5


# Sample table columns =============================================================================
#
# One row per sample in the fam file, in fam order.  Each sort of position a sample can have (its
# column in the bed file, and later its place in Omega) gets its own name, so that one can never be
# used where another was meant.

SAMPLE_FID_COL = bbf.FAM_FID_COL
SAMPLE_IID_COL = bbf.FAM_IID_COL
SAMPLE_BED_COL = "BED_COL"              # Column of the bed file holding this sample's genotypes
SAMPLE_SEX_COL = bbf.FAM_SEX_COL        # As the fam file records it, as text
SAMPLE_PHENO_COL = bbf.FAM_PHENO_COL    # The phenotype file's value where one is given
SAMPLE_HAS_PHENO_COL = "HAS_PHENOTYPE"
# Has a phenotype, complete covariates, and (where an ID list is given) is in it
SAMPLE_IN_ANALYSIS_COL = "IN_ANALYSIS"

SAMPLE_KEY = [SAMPLE_FID_COL, SAMPLE_IID_COL]


# Relatedness =====================================================================================
#
# The relationships GRMA distinguishes between a pair of samples, and how each of KING's InfType
# values maps to one.  PO and FS are separate relationships, though both are first-degree, because
# they lead to different regressions.  4th-degree and unrelated pairs play no part, and are left
# out of the relatedness graph.

# A duplicate sample, or an MZ twin: genotypes identical
RELATIONSHIP_DUPLICATE = "DUPLICATE"
RELATIONSHIP_PO = "PO"
RELATIONSHIP_FS = "FS"
RELATIONSHIP_2ND = "2ND"
RELATIONSHIP_3RD = "3RD"

INFTYPE_TO_RELATIONSHIP = {
    grma_io.INF_DUP_MZTWIN : RELATIONSHIP_DUPLICATE,
    grma_io.INF_PARENT_OFFSPRING : RELATIONSHIP_PO,
    grma_io.INF_FULLSIB : RELATIONSHIP_FS,
    grma_io.INF_2ND : RELATIONSHIP_2ND,
    grma_io.INF_3RD : RELATIONSHIP_3RD,
}
INFTYPES_IGNORED = (grma_io.INF_4TH, grma_io.INF_UNRELATED)

# The relatedness graph: one row per pair, each pair once, ROW_1 < ROW_2.  Rows are sample table
# rows.
GRAPH_ROW1_COL = "ROW_1"
GRAPH_ROW2_COL = "ROW_2"
GRAPH_RELATIONSHIP_COL = "RELATIONSHIP"

# Added to the sample table by resolve_duplicates(): for a dropped duplicate, the row of the sample
# kept in its place; -1 for every other sample
SAMPLE_DUPLICATE_OF_COL = "DUPLICATE_OF"


# Parents and groups ==============================================================================

# One row per child who will be residualized against their parents: in the analysis, with at
# least one parent genotyped.  Parent rows are sample table rows, -1 where that parent is not
# genotyped; the snipar family row indexes snipar's imputed data, -1 where it has none.
PARENTS_CHILD_COL = "CHILD_ROW"
PARENTS_FATHER_COL = "FATHER_ROW"
PARENTS_MOTHER_COL = "MOTHER_ROW"
PARENTS_SNIPAR_FAMILY_COL = "SNIPAR_FAMILY_ROW"
PARENTS_COLS = (PARENTS_CHILD_COL, PARENTS_FATHER_COL, PARENTS_MOTHER_COL,
                PARENTS_SNIPAR_FAMILY_COL)

# Each sample in the analysis is in exactly one group: the first it qualifies for, in this order
GROUP_PO = "PO"                 # At least one parent genotyped
GROUP_FS = "FS"                 # At least one full sib in the analysis
GROUP_2ND = "2ND"               # At least one 2nd-degree relative in the analysis
GROUP_3RD = "3RD"               # At least one 3rd-degree relative in the analysis
# In the analysis, but not the focal sample of any regression.  Such a sample may still be USED:
# as the genotyped parent of a child in the PO regression, for instance.
GROUP_NOT_FOCAL = "NOT_FOCAL"
SAMPLE_GROUP_COL = "GROUP"      # None for samples outside the analysis

# The relationship degree a regression needs.  --rel-degree runs every regression up to it, so 0
# and 1 both run PO and FS: the PO regression needs a snipar file, but no particular degree.
MAX_REL_DEGREE = 3
GROUP_MIN_REL_DEGREE = {GROUP_FS : 0, GROUP_2ND : 2, GROUP_3RD : 3}


# Omega ===========================================================================================

# Added to the sample table by select_omega_sample(): the sample's row and column in Omega, or -1
# for a sample not in it
SAMPLE_OMEGA_IDX_COL = "OMEGA_IDX"

# The relationships whose covariances fill Omega's off-diagonal.  Duplicates have been resolved by
# the time Omega is built, so none remain.
OMEGA_RELATIONSHIPS = (RELATIONSHIP_PO, RELATIONSHIP_FS, RELATIONSHIP_2ND, RELATIONSHIP_3RD)

# The floor Omega's smallest eigenvalue is shrunk to, so that it stays strictly positive definite.
# Absolute, as in the original implementation; whether it should instead scale with the phenotypic
# variance, so that a phenotype's units cannot affect it, is still to be decided.
DEFAULT_OMEGA_EPSILON = 1e-10
DEFAULT_EIGSH_TOL = 1e-8

# Matrices at or below this size use a dense eigendecomposition instead of ARPACK (exact,
# deterministic, and faster at this scale).  Relatedness blocks are nearly always well under this.
DENSE_LAMBDA_MIN_MAX_SIZE = 500

# Fixed seed for the ARPACK start vector, so repeated runs give identical results
EIGSH_START_VECTOR_SEED = 8675309


# Regressions =====================================================================================

# The regressions, in the order their estimates are combined.  Each is named after the group whose
# samples are its focal samples.
REGRESSION_ORDER = (GROUP_PO, GROUP_FS, GROUP_2ND, GROUP_3RD)

# Genotype residuals smaller than this are rounding noise, and are set to exactly zero.  A SNP with
# no variation among a focal sample and its relatives gives a residual of 0 in exact arithmetic,
# but about 1e-17 in floating point, since R's row (1 - 1/3, -1/3, -1/3, say) does not sum to
# exactly zero; left as it is, d would be tiny but positive and the "estimate" enormous.  Genotypes
# are always on the 0-2 scale, so a fixed tolerance is safe: a real residual is never this small.
GENOTYPE_RESIDUAL_TOLERANCE = 1e-9

# The relationship each relative regression residualizes against
RELATIVE_REGRESSION_RELATIONSHIP = {GROUP_FS : RELATIONSHIP_FS, GROUP_2ND : RELATIONSHIP_2ND,
                                    GROUP_3RD : RELATIONSHIP_3RD}


# SNPs ============================================================================================

ALLELE_COMPLEMENTS = {"A" : "T", "T" : "A", "C" : "G", "G" : "C"}

# Allele pairs that are their own complement, and so cannot have their strand determined from the
# letters alone
PALINDROMIC_ALLELE_PAIRS = ({"A", "T"}, {"C", "G"})

# snipar's imputed value, less the frequency where an allele was unobserved, is a whole number of
# alleles: one allele (0 or 1) where snipar observed one of the parent's two, the genotype (0, 1 or
# 2) where it observed both.  Each is rounded to that whole number, so that storage rounding cannot
# leave tiny nonzero residuals where the true residual is zero.  snipar stores these values as
# float16 -- about three significant figures -- so imputed - f can be off by up to about 0.0006;
# the tolerance leaves a wide margin over that.  A value further than this from any whole number
# does not fit that description at all, and is treated as missing.
SNIPAR_ALLELE_TOLERANCE = 0.01


# Running =========================================================================================

# Default number of SNPs to process at a time.  Memory in the genotype loop scales with it.
DEFAULT_SNPS_PER_BLOCK = 100


# Checks ===========================================================================================

def check_unique_ids(table: pd.DataFrame, name: str):
    """Raises if any FID/IID pair appears more than once

    Each sample must be identifiable by its IDs alone: a repeated pair would make every lookup by
    ID ambiguous, and silently picking one of the rows could pair a phenotype with the wrong
    genotypes.

    Args:
        table: Any input with FID and IID columns
        name: What the input is, for the error message

    Raises:
        ValueError: Naming every repeated pair (capped as describe_offenders caps it)
    """

    repeated = table.duplicated(subset=SAMPLE_KEY, keep=False)
    if repeated.any():
        pairs = (table.loc[repeated, SAMPLE_FID_COL] + " " +
                 table.loc[repeated, SAMPLE_IID_COL]).drop_duplicates()
        raise ValueError(f"The {name} lists some samples more than once: "
                         f"{grma_util.describe_offenders(pairs.tolist(), noun='sample(s)')}")


# The sample table =================================================================================

def build_sample_table(fam: pd.DataFrame, pheno: pd.DataFrame = None, covar: pd.DataFrame = None,
                       id_list: pd.DataFrame = None) -> pd.DataFrame:
    """One row per fam sample, saying where its genotypes are and whether it is in the analysis

    The inputs are as grma_io's readers return them, so their IDs are already text.  Checks run in
    order, and the first that fails raises, naming all of its own offenders: repeated IDs in each
    input, then an empty analysis.

    Args:
        fam: grma_io.read_fam()'s result
        pheno: grma_io.read_pheno()'s result, or None to use the fam file's phenotypes.  Where
            given, it is the only source: a sample it does not list has no phenotype.
        covar: grma_io.read_covar()'s result, or None.  A sample it does not list, or for which
            any covariate is missing, is not in the analysis.
        id_list: grma_io.read_id_list()'s result, or None.  Where given, only samples it lists can
            be in the analysis.

    Raises:
        ValueError: If any input repeats a sample, or no sample is left in the analysis

    Returns:
        One row per fam sample, in fam order: SAMPLE_FID_COL, SAMPLE_IID_COL, SAMPLE_BED_COL,
        SAMPLE_SEX_COL, SAMPLE_PHENO_COL, the covariates (COVAR_1 ... where covar is given),
        SAMPLE_HAS_PHENO_COL and SAMPLE_IN_ANALYSIS_COL
    """

    for table, name in ((fam, "fam file"), (pheno, "phenotype file"), (covar, "covariate file"),
                        (id_list, "ID list")):
        if table is not None:
            check_unique_ids(table, name)

    samples = fam[SAMPLE_KEY].copy()
    samples[SAMPLE_BED_COL] = np.arange(len(fam))
    samples[SAMPLE_SEX_COL] = fam[bbf.FAM_SEX_COL].to_numpy()
    keys = pd.MultiIndex.from_frame(samples[SAMPLE_KEY])

    if pheno is None:
        samples[SAMPLE_PHENO_COL] = fam[bbf.FAM_PHENO_COL].to_numpy()
    else:
        by_id = pheno.set_index(SAMPLE_KEY)[bbf.FAM_PHENO_COL]
        samples[SAMPLE_PHENO_COL] = by_id.reindex(keys).to_numpy()
    samples[SAMPLE_HAS_PHENO_COL] = samples[SAMPLE_PHENO_COL].notna()
    in_analysis = samples[SAMPLE_HAS_PHENO_COL].to_numpy().copy()

    if covar is not None:
        covar_cols = [c for c in covar.columns if c not in SAMPLE_KEY]
        aligned = covar.set_index(SAMPLE_KEY)[covar_cols].reindex(keys)
        for column in covar_cols:
            samples[column] = aligned[column].to_numpy()
        in_analysis &= aligned.notna().all(axis=1).to_numpy()

    if id_list is not None:
        listed = pd.MultiIndex.from_frame(id_list[SAMPLE_KEY])
        in_analysis &= keys.isin(listed)

    samples[SAMPLE_IN_ANALYSIS_COL] = in_analysis

    logging.info(f"\t{len(samples)} samples in the fam file; "
                 f"{int(samples[SAMPLE_HAS_PHENO_COL].sum())} with a phenotype; "
                 f"{int(in_analysis.sum())} in the analysis")

    if not in_analysis.any():
        raise ValueError("No sample is left in the analysis: none has a phenotype, complete "
                         "covariates and (where an ID list is given) a place in the ID list")

    return samples


# The relatedness graph ===========================================================================

def build_relatedness_graph(samples: pd.DataFrame, king: pd.DataFrame) -> pd.DataFrame:
    """KING's pairs between samples, as sample table rows, each pair once

    Only pairs that can matter are kept: both samples must be in the fam file, the relationship
    must be one GRMA uses, and at least one of the two must be in the analysis.  A pair between two
    samples outside the analysis can never be used -- even a parent outside it matters only
    through a child inside it -- so it is neither kept nor checked, and a contradiction between
    two unused samples never stops a run.

    Checks, in order, the first failure raising with all of its offenders:
        - an InfType KING does not define
        - a sample paired with itself
        - a pair given two different relationships, which cannot be resolved
    A pair listed more than once with the SAME relationship -- in either order -- is kept once, with
    a warning: the sparse matrices built from the graph would otherwise count it more than once.

    Args:
        samples: The sample table
        king: grma_io.read_king()'s result

    Raises:
        ValueError: As above

    Returns:
        GRAPH_ROW1_COL, GRAPH_ROW2_COL (ROW_1 < ROW_2) and GRAPH_RELATIONSHIP_COL
    """

    rel = king[grma_io.KING_REL_COL].to_numpy()
    unknown = sorted(set(rel) - set(INFTYPE_TO_RELATIONSHIP) - set(INFTYPES_IGNORED))
    if unknown:
        raise ValueError(f"The KING file has InfType values KING does not define: {unknown}")

    keys = pd.MultiIndex.from_frame(samples[SAMPLE_KEY])
    row1 = keys.get_indexer(pd.MultiIndex.from_arrays(
        [king[grma_io.KING_FID1_COL], king[grma_io.KING_IID1_COL]]))
    row2 = keys.get_indexer(pd.MultiIndex.from_arrays(
        [king[grma_io.KING_FID2_COL], king[grma_io.KING_IID2_COL]]))

    in_fam = (row1 != -1) & (row2 != -1)
    if not in_fam.all():
        logging.info(f"\t{int((~in_fam).sum())} KING pair(s) name a sample not in the fam file, "
                     f"and are ignored")
    used = np.isin(rel, list(INFTYPE_TO_RELATIONSHIP))
    in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
    keep = in_fam & used
    keep[keep] = in_analysis[row1[keep]] | in_analysis[row2[keep]]

    row1, row2, rel = row1[keep], row2[keep], rel[keep]
    iids = samples[SAMPLE_IID_COL].to_numpy()

    paired_with_self = row1 == row2
    if paired_with_self.any():
        raise ValueError(f"The KING file pairs some samples with themselves: "
                         f"{grma_util.describe_offenders(iids[row1[paired_with_self]])}")

    pairs = pd.DataFrame({GRAPH_ROW1_COL : np.minimum(row1, row2),
                          GRAPH_ROW2_COL : np.maximum(row1, row2),
                          GRAPH_RELATIONSHIP_COL : [INFTYPE_TO_RELATIONSHIP[r] for r in rel],
                          grma_io.KING_REL_COL : rel})

    by_pair = pairs.groupby([GRAPH_ROW1_COL, GRAPH_ROW2_COL])
    labels_per_pair = by_pair[grma_io.KING_REL_COL].unique()
    conflicting = labels_per_pair[labels_per_pair.map(len) > 1]
    if len(conflicting) > 0:
        details = [f"{iids[r1]}-{iids[r2]} ({', '.join(sorted(labels))})"
                   for (r1, r2), labels in conflicting.items()]
        raise ValueError(f"The KING file gives some pairs more than one relationship, which "
                         f"cannot be resolved: "
                         f"{grma_util.describe_offenders(details, noun='pair(s)')}.  KING writes "
                         f"one row per pair, so this usually comes from combining files.")

    repeated = pairs.duplicated(subset=[GRAPH_ROW1_COL, GRAPH_ROW2_COL]).to_numpy()
    if repeated.any():
        names = [f"{iids[r1]}-{iids[r2]}" for r1, r2 in
                 zip(pairs[GRAPH_ROW1_COL][repeated], pairs[GRAPH_ROW2_COL][repeated])]
        logging.warning(f"The KING file lists some pairs more than once, with the same "
                        f"relationship; each is counted once: "
                        f"{grma_util.describe_offenders(names, noun='pair(s)')}")

    graph = pairs.loc[~repeated, [GRAPH_ROW1_COL, GRAPH_ROW2_COL, GRAPH_RELATIONSHIP_COL]]
    logging.info(f"\tRelatedness graph: {len(graph)} pair(s); "
                 f"{graph[GRAPH_RELATIONSHIP_COL].value_counts().to_dict()}")
    return graph.reset_index(drop=True)


# Duplicate samples ===============================================================================

def resolve_duplicates(samples: pd.DataFrame, graph: pd.DataFrame):
    """Keeps one sample of each set of duplicates or MZ twins, and drops the rest

    Duplicate pairs can chain (A-B and B-C), so this works on clusters: the connected components of
    the duplicate pairs.  In each, the sample kept is the earliest in the fam file that is in the
    analysis, or the earliest overall if none is.  The others are dropped: taken out of the
    analysis, and every pair involving them is removed, so they are never used as a relative.
    Each dropped sample records which was kept in its place, so that a parent named as a dropped
    duplicate can be replaced by the one kept, whose genotypes are identical.

    Args:
        samples: The sample table
        graph: build_relatedness_graph()'s result

    Returns:
        A tuple (samples, graph): the sample table with SAMPLE_DUPLICATE_OF_COL added and the
        dropped samples out of the analysis; and the graph without duplicate pairs, without any
        pair involving a dropped sample, and without any pair left with neither sample in the
        analysis
    """

    samples = samples.copy()
    num_samples = len(samples)
    duplicate_of = np.full(num_samples, -1)

    is_duplicate = (graph[GRAPH_RELATIONSHIP_COL] == RELATIONSHIP_DUPLICATE).to_numpy()
    if is_duplicate.any():
        r1 = graph[GRAPH_ROW1_COL].to_numpy()[is_duplicate]
        r2 = graph[GRAPH_ROW2_COL].to_numpy()[is_duplicate]
        links = sp.coo_array((np.ones(len(r1)), (r1, r2)), shape=(num_samples, num_samples))
        _, labels = csgraph.connected_components(links, directed=False)

        in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
        for cluster in np.unique(labels[np.concatenate([r1, r2])]):
            members = np.flatnonzero(labels == cluster)     # Ascending: fam order
            eligible = members[in_analysis[members]]
            kept = eligible[0] if eligible.size else members[0]
            duplicate_of[members[members != kept]] = kept

    dropped = duplicate_of != -1
    samples[SAMPLE_DUPLICATE_OF_COL] = duplicate_of
    samples.loc[dropped, SAMPLE_IN_ANALYSIS_COL] = False
    if dropped.any():
        logging.info(f"\t{int(dropped.sum())} duplicate sample(s) or MZ twin(s) dropped, one of "
                     f"each set kept")
        dropped_iids = samples.loc[dropped, SAMPLE_IID_COL].tolist()
        logging.debug(f"\tDropped: {grma_util.describe_offenders(dropped_iids)}")

    # Removing every pair that involves a dropped sample also removes every duplicate pair, since
    # all but one sample of each cluster is dropped and a pair has two
    r1 = graph[GRAPH_ROW1_COL].to_numpy()
    r2 = graph[GRAPH_ROW2_COL].to_numpy()
    in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
    keep = ~dropped[r1] & ~dropped[r2] & (in_analysis[r1] | in_analysis[r2])
    return samples, graph.loc[keep].reset_index(drop=True)


# Parents =========================================================================================

def identify_parents(samples: pd.DataFrame, graph: pd.DataFrame,
                     snipar_metadata: dict = None) -> pd.DataFrame:
    """The children to be residualized against their parents, and who their parents are

    Parents come from snipar's pedigree, since fam files rarely carry them.  The pedigree names
    people by IID alone (its family IDs are snipar's own), so they are matched to the fam file by
    IID.  A parent named as a dropped duplicate is replaced by the sample kept in its place.

    Only children in the analysis are considered.  For each, checks run in order, the first failure
    raising with all of its offenders:
        - every IID the pedigree names must be unique in the fam file, or the match is ambiguous
        - snipar's record of which parents it had genotypes for must agree with which parents are
          in the fam file.  If they disagree, snipar was run on a different fileset, and its
          imputed data would be misread: a parent counted twice, or treated as absent.
        - every parent found must be a parent-offspring pair with the child in KING
    Two genotyped parents of the same recorded sex are only warned about: which parent is which
    does not affect the result, but a pedigree wrong in this way may be wrong in others.

    Args:
        samples: The sample table, after resolve_duplicates()
        graph: The relatedness graph, after resolve_duplicates()
        snipar_metadata: snipar_hdf5.read_snipar_metadata()'s result, or None without a snipar
            file, in which case nobody has parents and a warning is logged

    Raises:
        ValueError: As above

    Returns:
        PARENTS_COLS, one row per child in the analysis with at least one parent genotyped, in
        sample table order
    """

    empty = pd.DataFrame({column : np.array([], dtype=int) for column in PARENTS_COLS})
    if snipar_metadata is None:
        logging.warning("No imputed parental genotype file was given, so no parents are known "
                        "and the parent-offspring regression is omitted")
        return empty

    pedigree = snipar_metadata[snipar_hdf5.SNIPAR_PEDIGREE_KEY]
    ped_iids = pedigree[:, snipar_hdf5.SNIPAR_PED_IID_COL]
    ped_fathers = pedigree[:, snipar_hdf5.SNIPAR_PED_FATHER_COL]
    ped_mothers = pedigree[:, snipar_hdf5.SNIPAR_PED_MOTHER_COL]

    iids = samples[SAMPLE_IID_COL].to_numpy()
    in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
    duplicate_of = samples[SAMPLE_DUPLICATE_OF_COL].to_numpy()

    # Matching by IID is only safe where the IID names one sample.  Two unrelated samples may share
    # an IID in different families, which matters only if the pedigree names that IID.
    by_iid = pd.Series(np.arange(len(iids)), index=iids)
    shared = set(by_iid.index[by_iid.index.duplicated(keep=False)])
    named = set(ped_iids) | set(ped_fathers) | set(ped_mothers)
    ambiguous = sorted(shared & named)
    if ambiguous:
        raise ValueError(f"The imputed parental genotype file names people by IID alone, but "
                         f"some of the IIDs it names belong to more than one sample in the fam "
                         f"file: {grma_util.describe_offenders(ambiguous)}")
    by_iid = by_iid[~by_iid.index.duplicated(keep=False)]

    def rows_of(ped_ids):
        rows = by_iid.reindex(ped_ids).to_numpy()
        return np.where(np.isnan(rows), -1, rows).astype(int)

    def parent_rows_of(ped_ids):
        # "0" means no parent, as in a fam file, even if some sample happens to have the IID "0".
        # A parent dropped as a duplicate is replaced by the sample kept in its place.
        rows = rows_of(ped_ids)
        rows[ped_ids == bbf.FAM_MISSING_PARENT] = -1
        replaced = (rows != -1) & (duplicate_of[np.maximum(rows, 0)] != -1)
        rows[replaced] = duplicate_of[rows[replaced]]
        return rows

    # Children are NOT replaced: a child dropped as a duplicate is out of the analysis, and its
    # kept twin has its own record
    child = rows_of(ped_iids)
    father = parent_rows_of(ped_fathers)
    mother = parent_rows_of(ped_mothers)

    considered = (child != -1)
    considered[considered] = in_analysis[child[considered]]

    def flag(column):
        return np.char.lower(pedigree[:, column].astype(str)) == "true"
    disagree = considered & ((flag(snipar_hdf5.SNIPAR_PED_HAS_FATHER_COL) != (father != -1)) |
                             (flag(snipar_hdf5.SNIPAR_PED_HAS_MOTHER_COL) != (mother != -1)))
    if disagree.any():
        raise ValueError(f"The imputed parental genotype file and the fam file disagree about "
                         f"which parents are genotyped, for "
                         f"{grma_util.describe_offenders(ped_iids[disagree])}: snipar records "
                         f"whether it had each parent's genotypes, and a different set of those "
                         f"parents is in the fam file.  This happens when snipar was run on a "
                         f"different fileset.  Run snipar's impute.py on the same fileset given "
                         f"to GRMA.")

    usable = considered & ((father != -1) | (mother != -1))
    child, father, mother = child[usable], father[usable], mother[usable]
    ped_iids_used = ped_iids[usable]

    repeated = pd.Series(child).duplicated(keep=False).to_numpy()
    if repeated.any():
        raise ValueError(f"The imputed parental genotype file lists some children more than once: "
                         f"{grma_util.describe_offenders(sorted(set(ped_iids_used[repeated])))}")

    is_po = graph[GRAPH_RELATIONSHIP_COL] == RELATIONSHIP_PO
    po_pairs = graph.loc[is_po, [GRAPH_ROW1_COL, GRAPH_ROW2_COL]]
    po_pairs = set(zip(po_pairs[GRAPH_ROW1_COL], po_pairs[GRAPH_ROW2_COL]))
    unconfirmed = []
    for role, parent in (("father", father), ("mother", mother)):
        for c, p in zip(child, parent):
            if p != -1 and (min(c, p), max(c, p)) not in po_pairs:
                unconfirmed.append(f"{iids[c]} ({role} {iids[p]})")
    if unconfirmed:
        raise ValueError(f"The imputed parental genotype file names parents that KING does not "
                         f"classify as parent-offspring with the child: "
                         f"{grma_util.describe_offenders(unconfirmed, noun='child(ren)')}")

    sexes = samples[SAMPLE_SEX_COL].to_numpy()
    both = (father != -1) & (mother != -1)
    same_sex = both & np.isin(sexes[np.maximum(father, 0)], bbf.FAM_KNOWN_SEXES)
    same_sex &= sexes[np.maximum(father, 0)] == sexes[np.maximum(mother, 0)]
    if same_sex.any():
        logging.warning(f"Two genotyped parents of the same recorded sex are named for "
                        f"{grma_util.describe_offenders(iids[child[same_sex]])}.  Results are "
                        f"unaffected, since which parent is which does not matter, but a pedigree "
                        f"or a sex column wrong in this way may be wrong in others.")

    families = snipar_metadata[snipar_hdf5.SNIPAR_FAMILIES_KEY]
    family = pd.Index(families).get_indexer(
        pedigree[usable, snipar_hdf5.SNIPAR_PED_FID_COL])

    parents = pd.DataFrame({PARENTS_CHILD_COL : child, PARENTS_FATHER_COL : father,
                            PARENTS_MOTHER_COL : mother, PARENTS_SNIPAR_FAMILY_COL : family})
    parents = parents.sort_values(PARENTS_CHILD_COL).reset_index(drop=True)
    logging.info(f"\t{len(parents)} child(ren) with at least one parent genotyped; "
                 f"{int(both.sum())} with both")
    return parents


# Groups ==========================================================================================

def assign_groups(samples: pd.DataFrame, graph: pd.DataFrame, parents: pd.DataFrame,
                  rel_degree: int = MAX_REL_DEGREE) -> pd.DataFrame:
    """Places each sample in the analysis in exactly one group, the first it qualifies for

    In order: PO (a parent genotyped), FS, 2ND, 3RD (at least one such relative IN the analysis,
    since residualizing against a relative needs their phenotype), else NOT_FOCAL.  rel_degree
    cuts off the groups beyond it: those samples fall through to NOT_FOCAL.

    Args:
        samples: The sample table, after resolve_duplicates()
        graph: The relatedness graph, after resolve_duplicates()
        parents: identify_parents()'s result
        rel_degree: The highest relationship degree to run a regression for, 0 to MAX_REL_DEGREE

    Raises:
        ValueError: If rel_degree is out of range

    Returns:
        The sample table with SAMPLE_GROUP_COL added: a group for every sample in the analysis,
        None for every other
    """

    if not 0 <= rel_degree <= MAX_REL_DEGREE:
        raise ValueError(f"The relatedness degree must be between 0 and {MAX_REL_DEGREE}, "
                         f"but is {rel_degree}")

    samples = samples.copy()
    in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
    r1 = graph[GRAPH_ROW1_COL].to_numpy()
    r2 = graph[GRAPH_ROW2_COL].to_numpy()
    relationship = graph[GRAPH_RELATIONSHIP_COL].to_numpy()

    def has_relative_in_analysis(wanted):
        result = np.zeros(len(samples), dtype=bool)
        related = relationship == wanted
        result[r1[related & in_analysis[r2]]] = True
        result[r2[related & in_analysis[r1]]] = True
        return result

    group = np.full(len(samples), None, dtype=object)
    unassigned = in_analysis.copy()

    has_parent = np.zeros(len(samples), dtype=bool)
    has_parent[parents[PARENTS_CHILD_COL].to_numpy()] = True
    for name, qualifies in ((GROUP_PO, has_parent),
                            (GROUP_FS, has_relative_in_analysis(RELATIONSHIP_FS)),
                            (GROUP_2ND, has_relative_in_analysis(RELATIONSHIP_2ND)),
                            (GROUP_3RD, has_relative_in_analysis(RELATIONSHIP_3RD))):
        if rel_degree < GROUP_MIN_REL_DEGREE.get(name, 0):
            continue
        joining = unassigned & qualifies
        group[joining] = name
        unassigned &= ~joining
    group[unassigned] = GROUP_NOT_FOCAL

    # Kept as plain objects: pandas would otherwise turn this text column into its own string type,
    # which stores None as NaN and breaks the "None outside the analysis" sentinel
    samples[SAMPLE_GROUP_COL] = pd.Series(group, index=samples.index, dtype=object)
    logging.info(f"\tGroups: {pd.Series(group[in_analysis]).value_counts().to_dict()}")
    return samples


# Omega ===========================================================================================

def select_omega_sample(samples: pd.DataFrame, parents: pd.DataFrame) -> pd.DataFrame:
    """Chooses the samples Omega covers, and gives each its position in it

    Omega covers every sample in the analysis that some regression uses: each focal sample (any
    group but NOT_FOCAL), and each genotyped parent of a child in the PO regression.  Parents count
    because their genotypes enter their children's residuals, and because the PO covariance is
    estimated from parent-child pairs.  A relative used in a sibling, 2nd- or 3rd-degree
    residualization needs no separate rule: having a relative of that degree in the analysis makes
    them focal in some group themselves.  Everyone else in the analysis -- a sample related to
    nobody, for instance -- is left out, so that they affect neither Omega's variance nor the
    phenotypes' centering.

    Args:
        samples: The sample table, after assign_groups()
        parents: identify_parents()'s result

    Raises:
        ValueError: If no sample is used by any regression

    Returns:
        The sample table with SAMPLE_OMEGA_IDX_COL added: 0, 1, 2 ... in sample table order for the
        samples in Omega, -1 for every other
    """

    samples = samples.copy()
    in_analysis = samples[SAMPLE_IN_ANALYSIS_COL].to_numpy()
    group = samples[SAMPLE_GROUP_COL].to_numpy()
    focal = np.array([g is not None and g != GROUP_NOT_FOCAL for g in group])

    is_parent = np.zeros(len(samples), dtype=bool)
    for column in (PARENTS_FATHER_COL, PARENTS_MOTHER_COL):
        rows = parents[column].to_numpy()
        is_parent[rows[rows != -1]] = True

    member = in_analysis & (focal | is_parent)
    if not member.any():
        raise ValueError("No sample is used by any regression: no sample in the analysis has a "
                         "genotyped parent or a relative in the analysis")

    omega_idx = np.full(len(samples), -1)
    omega_idx[member] = np.arange(int(member.sum()))
    samples[SAMPLE_OMEGA_IDX_COL] = omega_idx

    logging.info(f"\tOmega covers {int(member.sum())} sample(s): "
                 f"{int((member & focal).sum())} focal, {int((member & ~focal).sum())} used only "
                 f"as parents; {int((in_analysis & ~member).sum())} sample(s) in the analysis are "
                 f"used by no regression")
    return samples


def prepare_phenotypes(samples: pd.DataFrame) -> np.ndarray:
    """The phenotypes of the samples in Omega, residualized on an intercept and any covariates

    Done within Omega's samples only, so that samples no regression uses cannot shift the centering.
    With no covariates this is centering on the mean.

    Args:
        samples: The sample table, after select_omega_sample()

    Raises:
        ValueError: If nothing is left to explain: every residualized phenotype is the same

    Returns:
        The residualized phenotypes, (omega samples,), in Omega order
    """

    in_omega = samples[SAMPLE_OMEGA_IDX_COL].to_numpy() != -1
    phenotypes = samples.loc[in_omega, SAMPLE_PHENO_COL].to_numpy(dtype=float)
    covar_cols = [c for c in samples.columns if str(c).startswith(grma_io.COVAR_PREFIX)]

    design = np.column_stack([samples.loc[in_omega, covar_cols].to_numpy(dtype=float),
                              np.ones(len(phenotypes))])
    coefficients, _, _, _ = np.linalg.lstsq(design, phenotypes, rcond=None)
    residualized = phenotypes - design @ coefficients

    scale = max(float(np.abs(phenotypes).max()), 1.0)
    if np.all(np.abs(residualized) <= 1e-12 * scale):
        raise ValueError("Every phenotype in Omega is the same once the intercept and any "
                         "covariates are accounted for, so there is nothing to estimate")
    logging.info(f"\tPhenotypes residualized on an intercept"
                 f"{f' and {len(covar_cols)} covariate(s)' if covar_cols else ''}, "
                 f"within the {len(phenotypes)} samples in Omega")
    return residualized


def get_omega_pairs(samples: pd.DataFrame, graph: pd.DataFrame):
    """The relatedness graph's pairs within Omega, as Omega positions

    Every relationship in OMEGA_RELATIONSHIPS is kept, whatever rel_degree is: a relative beyond the
    regressions' cut-off is still phenotypically correlated with samples in them.

    Args:
        samples: The sample table, after select_omega_sample()
        graph: The relatedness graph, after resolve_duplicates()

    Returns:
        A tuple (first, second, relationship), each (pairs,): the two samples' Omega positions and
        the pair's relationship
    """

    omega_idx = samples[SAMPLE_OMEGA_IDX_COL].to_numpy()
    first = omega_idx[graph[GRAPH_ROW1_COL].to_numpy()]
    second = omega_idx[graph[GRAPH_ROW2_COL].to_numpy()]
    relationship = graph[GRAPH_RELATIONSHIP_COL].to_numpy()
    keep = (first != -1) & (second != -1) & np.isin(relationship, OMEGA_RELATIONSHIPS)
    return first[keep], second[keep], relationship[keep]


def get_relationship_covariances(first: np.ndarray, second: np.ndarray, relationship: np.ndarray,
                                 phenotypes: np.ndarray) -> dict:
    """The phenotypic covariance for each relationship, from its pairs

    For each relationship, the mean is taken over the DISTINCT samples in its pairs, each counted
    once however many pairs they are in; the covariance is the average, over the pairs, of the
    product of the two samples' deviations from that mean.  Dividing by the number of pairs rather
    than one fewer: scaling Omega by a constant changes none of the estimates.

    Args:
        first, second: The pairs' Omega positions, each (pairs,)
        relationship: Each pair's relationship, (pairs,)
        phenotypes: The residualized phenotypes, (omega samples,)

    Returns:
        Relationship -> covariance, for each relationship that has at least one pair
    """

    covariances = {}
    for r in OMEGA_RELATIONSHIPS:
        of_r = relationship == r
        if not of_r.any():
            continue
        members = np.unique(np.concatenate([first[of_r], second[of_r]]))
        mean = phenotypes[members].mean()
        covariances[r] = float(np.mean((phenotypes[first[of_r]] - mean) *
                                       (phenotypes[second[of_r]] - mean)))
    logging.debug(f"\tPhenotypic covariance by relationship: {covariances}")
    return covariances


def get_omega_components(first: np.ndarray, second: np.ndarray, num_samples: int) -> np.ndarray:
    """Which connected component of Omega's off-diagonal each sample is in

    Args:
        first, second: The pairs' Omega positions, each (pairs,)
        num_samples: The number of samples in Omega

    Returns:
        Each sample's component label, (omega samples,)
    """

    links = sp.coo_array((np.ones(len(first)), (first, second)), shape=(num_samples, num_samples))
    _, labels = csgraph.connected_components(links, directed=False)
    return labels


def get_lambda_min(B: np.ndarray | sp.csr_array, tol=DEFAULT_EIGSH_TOL) -> float:

    N = B.shape[0]

    # For small matrices a dense eigendecomposition is exact, deterministic, and avoids ARPACK's
    # convergence failures entirely.  Relatedness blocks are almost always families of a handful
    # of people, so in practice this is the path taken.
    if N <= DENSE_LAMBDA_MIN_MAX_SIZE:
        return float(np.linalg.eigvalsh(B.toarray() if sp.issparse(B) else B).min())

    # ARPACK is iterative and starts from a random vector unless told otherwise, which makes
    # results differ between runs at the ULP level.  A fixed start vector makes them repeatable.
    # (A vector of ones is a poor choice here -- it is exactly orthogonal to the smallest
    # eigenvector of a bipartite-like block, which is precisely the structure that needs shrinking.)
    v0 = np.random.default_rng(EIGSH_START_VECTOR_SEED).standard_normal(N)

    for ncv_factor_exp in range(5):
        ncv = min(N, 20 * 2 ** (ncv_factor_exp + 1))

        try:
            eigenvalues, _ = spla.eigsh(B, k=1, which='SA', tol=tol, ncv=ncv, v0=v0)
            return eigenvalues[0]
        except spla.ArpackNoConvergence:
            continue

    raise RuntimeError("Eigsh failed to converge.  Not possible to compute omega.")


def get_block_alphas(off_diag: sp.csr_array, pheno_variance: float, labels: np.ndarray,
                     epsilon=DEFAULT_OMEGA_EPSILON) -> np.ndarray:
    """Computes a shrinkage factor for each connected component of off_diag.

    Omega's off-diagonal is block-diagonal over the connected components of the relatedness
    graph: a pair only ever contributes to the block containing both of its members.  The
    eigenvalues of a block-diagonal matrix are the union of its blocks', so positive-definiteness
    is a PER-BLOCK property and there is no mathematical requirement for alpha to be a single
    global scalar.

    With one global alpha, the single worst component sets the shrinkage for every individual in
    the sample.  Shrinking each component only as far as that component requires preserves
    positive-definiteness identically -- every block independently satisfies
    pheno_variance + alpha_c * lambda_min_c >= epsilon, so the union of the eigenvalues is
    positive -- while leaving unaffected blocks untouched.

    Args:
        off_diag: The unshrunk off-diagonal, as built by calculate_omega
        pheno_variance: Variance of the unresidualized phenotypes
        labels: Component index of each sample, from get_relatedness_components()
        epsilon: Small positive constant keeping the result strictly definite

    Returns:
        Numpy array alphas where alphas[c] is the shrinkage factor for component c
    """

    n_comp = labels.max() + 1 if len(labels) > 0 else 0
    alphas = np.ones(n_comp, dtype=float)
    sizes = np.bincount(labels, minlength=n_comp)

    # Gershgorin's circle theorem bounds every eigenvalue of a symmetric matrix by its largest
    # absolute row sum.  off_diag has a zero diagonal, so lambda_min >= -max_row_sum, and a block
    # whose bound is already below pheno_variance cannot break positive-definiteness -- no
    # eigendecomposition needed.  This skips the overwhelming majority of blocks (ordinary
    # families) and keeps the cost close to the single global call it replaces.  The threshold is
    # pheno_variance - epsilon rather than pheno_variance because alpha targets a smallest
    # eigenvalue of epsilon, not of zero.
    row_abs_sum = np.asarray(np.abs(off_diag).sum(axis=1)).ravel()
    block_bound = np.zeros(n_comp, dtype=float)
    np.maximum.at(block_bound, labels, row_abs_sum)

    candidates = np.nonzero((sizes > 1) & (block_bound >= pheno_variance - epsilon))[0]
    logging.debug(f"\t{n_comp} relatedness components; {len(candidates)} need an "
                  f"eigendecomposition after the Gershgorin screen")

    if len(candidates) > 0:
        order = np.argsort(labels, kind="stable")
        starts = np.searchsorted(labels[order], np.arange(n_comp))
        ends = np.searchsorted(labels[order], np.arange(n_comp), side="right")

        for c in candidates:
            idx = order[starts[c]:ends[c]]
            block_lambda_min = get_lambda_min(off_diag[idx][:, idx])

            if block_lambda_min < 0.0:
                alphas[c] = min((epsilon - pheno_variance) / block_lambda_min, 1.0)

                # |lambda_min| / max_row_sum separates a densely interconnected family (ratio near
                # 1/degree) from a triangle-free hub (ratio near 1).  The latter often indicates a
                # data problem rather than a real pedigree.
                logging.debug(f"\t\tComponent {c}: size={sizes[c]}, alpha={alphas[c]:.6f}, "
                              f"lambda_min={block_lambda_min:.6g}, "
                              f"max_row_sum={block_bound[c]:.6g}, "
                              f"ratio={abs(block_lambda_min) / block_bound[c]:.4f}")

    shrunk = alphas < 1.0
    if np.any(shrunk):
        logging.info(f"\tPer-block shrinkage: {int(shrunk.sum())} of {n_comp} components shrunk; "
                     f"smallest alpha {alphas.min():.6f}, {int(sizes[shrunk].sum())} of "
                     f"{off_diag.shape[0]} samples affected")
    else:
        logging.info("\tPer-block shrinkage: no component required shrinking")

    return alphas


def build_omega(samples: pd.DataFrame, graph: pd.DataFrame, phenotypes: np.ndarray,
                epsilon: float = DEFAULT_OMEGA_EPSILON) -> sp.csr_array:
    """Omega: the phenotypic covariance among the samples in it, regularized once

    The diagonal is the residualized phenotypes' variance.  Each related pair's off-diagonal entry
    is its relationship's covariance (get_relationship_covariances()); every other entry is zero.
    Each connected component of the off-diagonal is then shrunk only as far as needed to keep it
    positive definite (get_block_alphas()).  The one regularized Omega is used by every regression.

    Args:
        samples: The sample table, after select_omega_sample()
        graph: The relatedness graph, after resolve_duplicates()
        phenotypes: prepare_phenotypes()'s result, (omega samples,)
        epsilon: The floor for Omega's smallest eigenvalue

    Returns:
        Omega, (omega samples, omega samples)
    """

    num_samples = len(phenotypes)
    first, second, relationship = get_omega_pairs(samples, graph)
    covariances = get_relationship_covariances(first, second, relationship, phenotypes)

    values = np.array([covariances[r] for r in relationship], dtype=float)
    off_diag = sp.coo_array((np.concatenate([values, values]),
                             (np.concatenate([first, second]), np.concatenate([second, first]))),
                            shape=(num_samples, num_samples)).tocsr()

    variance = float(np.var(phenotypes))
    logging.info(f"\tOmega: {num_samples} samples, phenotypic variance {variance:.6g}, "
                 f"{len(first)} related pair(s)")

    labels = get_omega_components(first, second, num_samples)
    alphas = get_block_alphas(off_diag, variance, labels, epsilon)

    # Both members of a pair are in the same component, so an entry's column identifies its
    # component as well as its row does
    off_diag.data *= alphas[labels[off_diag.indices]]
    return sp.csr_array(variance * sp.eye_array(num_samples) + off_diag)


# Regressions: the plan ===========================================================================

@dataclasses.dataclass
class ParentColumns:
    """For each child in the PO regression, where its parents' data is

    Attributes:
        father, mother: The parent's column among the genotypes the loop reads, (children,), -1
            where that parent is not genotyped
        family: The child's family's row in snipar's imputed data, (children,), -1 where snipar
            has none
    """

    father: np.ndarray
    mother: np.ndarray
    family: np.ndarray


@dataclasses.dataclass
class Regression:
    """One regression, as the genotype loop needs it

    Attributes:
        name: Which regression: one of REGRESSION_ORDER
        focal: The focal samples' positions in Omega, (focal,)
        phenotype_map: The matrix A taking Omega's phenotypes to this regression's: a selection
            matrix for PO, a residualization matrix R for a relative regression,
            (focal, omega samples)
        phenotypes: This regression's phenotypes, A y, (focal,)
        gls_weights: The GLS weight matrix Q, the pseudo-inverse of A Omega A', (focal, focal)
        parents: For the PO regression, where each child's parents' data is; None otherwise
    """

    name: str
    focal: np.ndarray
    phenotype_map: sp.csr_array
    phenotypes: np.ndarray
    gls_weights: sp.csr_array
    parents: ParentColumns = None


def build_residualization_matrix(samples: pd.DataFrame, graph: pd.DataFrame, group: str):
    """The matrix R turning each focal sample's value into its deviation from its relatives' mean

    The focal samples are those in the group.  Each focal sample's relatives are the samples in
    Omega with the group's relationship to it, whatever group THEY are in; the mean is over the
    focal sample and all of them.  So a focal sample's row holds 1 - 1/(k+1) for itself and
    -1/(k+1) for each of its k relatives, and sums to zero -- which is what makes the regression
    unaffected by adding a constant to every phenotype or to every genotype.

    Args:
        samples: The sample table, after select_omega_sample()
        graph: The relatedness graph, after resolve_duplicates()
        group: GROUP_FS, GROUP_2ND or GROUP_3RD

    Returns:
        A tuple (focal, R): the focal samples' Omega positions, (focal,), in Omega order; and R,
        (focal, omega samples)
    """

    omega_idx = samples[SAMPLE_OMEGA_IDX_COL].to_numpy()
    is_focal = (samples[SAMPLE_GROUP_COL] == group).to_numpy() & (omega_idx != -1)
    focal = omega_idx[is_focal]
    num_omega = int(omega_idx.max()) + 1

    first, second, relationship = get_omega_pairs(samples, graph)
    of_group = relationship == RELATIVE_REGRESSION_RELATIONSHIP[group]
    relatives = {f : [] for f in focal}
    for a, b in zip(first[of_group], second[of_group]):
        if a in relatives:
            relatives[a].append(b)
        if b in relatives:
            relatives[b].append(a)

    rows, cols, values = [], [], []
    for row, f in enumerate(focal):
        members = [f] + relatives[f]
        share = 1.0 / len(members)
        rows.extend([row] * len(members))
        cols.extend(members)
        values.extend([1.0 - share] + [-share] * (len(members) - 1))
    R = sp.csr_array((values, (rows, cols)), shape=(len(focal), num_omega))
    return focal, R


def calculate_gls_weights(vcv: sp.csr_array) -> sp.csr_array:
    """The GLS weight matrix: the pseudo-inverse of a regression's phenotypic covariance

    The covariance is block-diagonal over its connected components, so it is inverted block by
    block.  Each block's cutoff for treating a direction as null scales with the block's size, as
    the rounding noise in those directions does: a residualization matrix makes every block
    singular by construction, and a fixed cutoff would invert that noise in large blocks, giving
    weights around 1e14.

    Args:
        vcv: The regression's phenotypic covariance, A Omega A', (focal, focal)

    Returns:
        Its pseudo-inverse, (focal, focal), with the same block structure
    """

    size = vcv.shape[0]
    _, labels = csgraph.connected_components(vcv, directed=False)
    rows, cols, values = [], [], []
    for c in range(labels.max() + 1 if size else 0):
        idx = np.flatnonzero(labels == c)
        block = vcv[idx][:, idx].toarray()
        inverse = np.linalg.pinv(block, rcond=max(block.shape) * np.finfo(float).eps,
                                 hermitian=True)
        r, k = np.meshgrid(idx, idx, indexing="ij")
        rows.append(r.ravel())
        cols.append(k.ravel())
        values.append(inverse.ravel())
    if not rows:
        return sp.csr_array((size, size))
    return sp.csr_array((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(size, size))


def build_relative_regressions(samples: pd.DataFrame, graph: pd.DataFrame, phenotypes: np.ndarray,
                               omega: sp.csr_array, rel_degree: int = MAX_REL_DEGREE) -> list:
    """The sibling, 2nd- and 3rd-degree regressions that rel_degree allows

    A regression whose group is empty is left out, with a warning: estimates that would include it
    are still reported, so that the output's layout depends only on the options, but they equal
    the estimates without it.

    Args:
        samples: The sample table, after select_omega_sample()
        graph: The relatedness graph, after resolve_duplicates()
        phenotypes: prepare_phenotypes()'s result, (omega samples,)
        omega: build_omega()'s result
        rel_degree: As for assign_groups()

    Returns:
        A list of Regression, in REGRESSION_ORDER
    """

    regressions = []
    for group, relationship in RELATIVE_REGRESSION_RELATIONSHIP.items():
        if rel_degree < GROUP_MIN_REL_DEGREE[group]:
            continue
        focal, R = build_residualization_matrix(samples, graph, group)
        if len(focal) == 0:
            logging.warning(f"No sample qualifies for the {group} regression, so it is omitted; "
                            f"every reported estimate that would include it equals the one "
                            f"without it")
            continue
        vcv = sp.csr_array(R @ omega @ R.T)
        regressions.append(Regression(name=group, focal=focal, phenotype_map=R,
                                      phenotypes=R @ phenotypes,
                                      gls_weights=calculate_gls_weights(vcv)))
        logging.info(f"\t{group} regression: {len(focal)} focal sample(s)")
    return regressions


def build_cross_covariances(regressions: list, omega: sp.csr_array) -> dict:
    """The phenotypic covariance between every two regressions' phenotypes, A_i Omega A_j'

    Computed once, so that the genotype loop can find the covariance of the estimates from each
    regression's own quantities alone, never spreading weights over all of Omega.

    Args:
        regressions: The regressions
        omega: build_omega()'s result

    Returns:
        (name_i, name_j) -> A_i Omega A_j', (focal_i, focal_j), for every ordered pair i <= j in
        the list's order
    """

    cross = {}
    for i, first in enumerate(regressions):
        weighted = first.phenotype_map @ omega
        for second in regressions[i:]:
            cross[(first.name, second.name)] = sp.csr_array(weighted @ second.phenotype_map.T)
    return cross


def reported_estimates(has_po: bool, rel_degree: int = MAX_REL_DEGREE) -> list:
    """The estimates reported, each as the regressions it combines

    Cumulative, in REGRESSION_ORDER: PO alone, then PO+FS, PO+FS+2ND, PO+FS+2ND+3RD, as far as
    rel_degree allows.  Without PO (no imputed parental genotype file), the same without it: FS,
    FS+2ND, FS+2ND+3RD.  This depends only on the options, never on the data, so the output's
    columns do too.

    Args:
        has_po: Whether there is a PO regression to include, i.e. an imputed parental file
        rel_degree: As for assign_groups()

    Returns:
        A list of tuples of regression names
    """

    allowed = [g for g in REGRESSION_ORDER[1:] if rel_degree >= GROUP_MIN_REL_DEGREE[g]]
    included = ([GROUP_PO] if has_po else []) + allowed
    return [tuple(included[:n + 1]) for n in range(len(included))]


# SNPs ============================================================================================

@dataclasses.dataclass
class SnpPlan:
    """Which SNPs are analysed, and where each is in snipar's file

    Attributes:
        bim_rows: The analysed SNPs' rows in the bim file, (snps,), increasing
        snipar_positions: Each SNP's position in snipar's file, (snps,), -1 where it has no usable
            data; None without a snipar file
        flip: Whether each SNP's alleles are swapped in snipar's file relative to the bim file,
            (snps,); None without a snipar file
    """

    bim_rows: np.ndarray
    snipar_positions: np.ndarray = None
    flip: np.ndarray = None


def classify_allele_orientation(grma_alleles: np.ndarray, snipar_alleles: np.ndarray):
    """How each SNP's alleles in snipar's file relate to the bim file's

    Args:
        grma_alleles: The bim file's A1 and A2, (snps, 2)
        snipar_alleles: snipar's allele1 and allele2, (snps, 2)

    Returns:
        A tuple (agrees, swapped, strand_differs), each (snps,): the same alleles in the same
        order; the same alleles in the other order; and the alleles of the opposite strand, in
        either order
    """

    complemented = np.vectorize(ALLELE_COMPLEMENTS.get)(snipar_alleles)
    agrees = (grma_alleles == snipar_alleles).all(axis=1)
    swapped = (grma_alleles == snipar_alleles[:, ::-1]).all(axis=1)
    strand_differs = ((grma_alleles == complemented).all(axis=1) |
                      (grma_alleles == complemented[:, ::-1]).all(axis=1))
    return agrees, swapped, strand_differs


def align_snipar_to_bim(bim: pd.DataFrame, snipar_metadata: dict):
    """Where each bim SNP is in snipar's file, and whether its alleles are swapped there

    SNPs are matched by rsid.  A SNP snipar does not have, or has with alleles that cannot be
    reconciled with the bim file's, has no usable imputed data: it is still analysed, but children
    with one parent genotyped have no residual at it.  Palindromic SNPs (A/T, C/G) can be used only
    if no non-palindromic SNP is recorded on opposite strands in the two files, since their strand
    cannot be read from the letters.

    Raises:
        ValueError: If an rsid the bim file names appears more than once in snipar's file, so the
            match is ambiguous; or if an rsid is at different positions in the two files, so they
            are not the same variant

    Args:
        bim: The bim rows to align, as grma_io.read_bim() returns them
        snipar_metadata: snipar_hdf5.read_snipar_metadata()'s result

    Returns:
        A tuple (positions, flip), each (snps,): the position in snipar's file, -1 where unusable;
        and whether the alleles are swapped
    """

    values = snipar_metadata[snipar_hdf5.SNIPAR_BIM_VALUES_KEY]
    columns = snipar_metadata[snipar_hdf5.SNIPAR_BIM_COLUMNS_KEY]

    def column(names):
        return values[:, snipar_hdf5.get_snipar_bim_column_index(columns, names)].astype(str)

    snipar_ids = pd.Series(np.arange(len(values)), index=column(snipar_hdf5.SNIPAR_BIM_ID_NAMES))
    rsids = bim[bbf.BIM_RSID_COL].to_numpy()
    repeated = set(snipar_ids.index[snipar_ids.index.duplicated(keep=False)])
    ambiguous = sorted(repeated & set(rsids))
    if ambiguous:
        raise ValueError(f"The imputed parental genotype file lists some rsids more than once, "
                         f"so they cannot be matched to the bim file: "
                         f"{grma_util.describe_offenders(ambiguous, noun='rsid(s)')}")
    snipar_ids = snipar_ids[~snipar_ids.index.duplicated(keep=False)]

    found = snipar_ids.reindex(rsids).to_numpy()
    matched = ~np.isnan(found)
    positions = np.where(matched, found, -1).astype(int)

    at = positions[matched]
    same_place = np.ones(len(bim), dtype=bool)
    same_place[matched] = (
        (column(snipar_hdf5.SNIPAR_BIM_CHR_NAMES)[at] == bim[bbf.BIM_CHR_COL].to_numpy()[matched]) &
        (column(snipar_hdf5.SNIPAR_BIM_BP_NAMES)[at].astype(np.int64) ==
         bim[bbf.BIM_BP_COL].to_numpy()[matched]))
    if not same_place.all():
        raise ValueError(f"Some rsids are at different positions in the bim file and the imputed "
                         f"parental genotype file, so they are not the same variant: "
                         f"{grma_util.describe_offenders(rsids[~same_place], noun='rsid(s)')}")

    grma_alleles = bim[[bbf.BIM_A1_COL, bbf.BIM_A2_COL]].to_numpy().astype(str)
    snipar_alleles = np.column_stack([column(snipar_hdf5.SNIPAR_BIM_A1_NAMES),
                                      column(snipar_hdf5.SNIPAR_BIM_A2_NAMES)]
                                     )[np.maximum(positions, 0)]
    agrees, swapped, strand_differs = classify_allele_orientation(grma_alleles, snipar_alleles)
    palindromic = np.array([set(pair) in PALINDROMIC_ALLELE_PAIRS for pair in grma_alleles])

    strand_evidence = int((matched & ~palindromic & strand_differs).sum())
    if strand_evidence:
        logging.warning(f"{strand_evidence} SNP(s) are recorded on opposite strands in the bim "
                        f"file and the imputed parental genotype file.  Their imputed data, and "
                        f"every palindromic SNP's, cannot be used.")
    usable = matched & (agrees | swapped) & (~palindromic | (strand_evidence == 0))

    logging.info(f"\t{int(usable.sum())} of {len(bim)} SNP(s) have usable imputed parental data; "
                 f"{int((~matched).sum())} are not in the file, "
                 f"{int((matched & ~usable).sum())} have alleles that cannot be reconciled")
    return np.where(usable, positions, -1), usable & swapped


def plan_snps(bim: pd.DataFrame, snp_list: pd.DataFrame = None,
              snipar_metadata: dict = None) -> SnpPlan:
    """The SNPs to analyse, and where each is in snipar's file

    Every bim SNP, or only those in the SNP list where one is given.  With a snipar file, each SNP
    is aligned to it.  A SNP without usable imputed data is still analysed: the sibling, 2nd- and
    3rd-degree regressions and children with both parents genotyped do not need it.  Only children
    with one parent genotyped lose their residual at it, and a warning says how many SNPs this
    affects, since many would suggest the files were built from different SNP sets.

    Args:
        bim: grma_io.read_bim()'s result
        snp_list: grma_io.read_snp_list()'s result, or None
        snipar_metadata: snipar_hdf5.read_snipar_metadata()'s result, or None

    Raises:
        ValueError: If no SNP is left to analyse; if snipar's file lists the analysed SNPs in a
            different order from the bim file; or as align_snipar_to_bim() raises

    Returns:
        A SnpPlan
    """

    rows = np.arange(len(bim))
    if snp_list is not None:
        listed = set(snp_list[bbf.BIM_RSID_COL])
        in_list = bim[bbf.BIM_RSID_COL].isin(listed).to_numpy()
        unknown = len(listed - set(bim[bbf.BIM_RSID_COL]))
        if unknown:
            logging.warning(f"{unknown} SNP(s) in the SNP list are not in the bim file")
        rows = rows[in_list]
    if len(rows) == 0:
        raise ValueError("No SNP is left to analyse: the SNP list names none of the bim file's")
    logging.info(f"\t{len(rows)} SNP(s) to analyse")

    if snipar_metadata is None:
        return SnpPlan(bim_rows=rows)

    analysed = bim.iloc[rows].reset_index(drop=True)
    positions, flip = align_snipar_to_bim(analysed, snipar_metadata)

    # The reader moves forward through snipar's file, so the positions must never decrease.  A
    # repeated position is fine: it can only come from the bim file listing one SNP twice.
    present = np.flatnonzero(positions != -1)
    backwards = present[1:][np.diff(positions[present]) < 0]
    if backwards.size:
        rsids = analysed[bbf.BIM_RSID_COL].to_numpy()[backwards]
        raise ValueError(f"The imputed parental genotype file lists SNPs in a different order "
                         f"from the bim file: "
                         f"{grma_util.describe_offenders(rsids, noun='SNP(s)')} come earlier in "
                         f"the imputed file than the SNP before them in the bim file.  GRMA reads "
                         f"the imputed file in the bim file's order, so the two must match.  Run "
                         f"snipar's impute.py on a fileset whose variants are in the same order "
                         f"as the one given to GRMA.")

    missing = int((positions == -1).sum())
    if missing:
        logging.warning(f"{missing} of {len(rows)} SNP(s) have no usable imputed parental data.  "
                        f"They are still analysed, but children with one parent genotyped "
                        f"contribute nothing at them.")
    return SnpPlan(bim_rows=rows, snipar_positions=positions, flip=flip)


# The PO regression ===============================================================================

def plan_genotype_samples(samples: pd.DataFrame, parents: pd.DataFrame):
    """The samples whose genotypes the loop reads, and in what order

    The Omega samples first, in Omega order, so that the first columns of a genotype block are
    exactly Omega's; then each genotyped parent outside Omega, in sample table order.  A parent
    outside Omega has no phenotype, but their genotype is still needed for their child's residual.

    Args:
        samples: The sample table, after select_omega_sample()
        parents: identify_parents()'s result

    Returns:
        A tuple (bed_cols, column_of_row): the bed file columns to read, (loaded samples,), in
        that order; and each sample table row's column among them, (sample table rows,), -1 for
        a sample not read
    """

    omega_idx = samples[SAMPLE_OMEGA_IDX_COL].to_numpy()
    bed = samples[SAMPLE_BED_COL].to_numpy()
    in_omega = np.flatnonzero(omega_idx != -1)
    in_omega = in_omega[np.argsort(omega_idx[in_omega])]

    parent_rows = np.concatenate([parents[PARENTS_FATHER_COL].to_numpy(),
                                  parents[PARENTS_MOTHER_COL].to_numpy()])
    extra = np.unique(parent_rows[(parent_rows != -1)])
    extra = extra[omega_idx[extra] == -1]

    loaded_rows = np.concatenate([in_omega, extra]).astype(int)
    column_of_row = np.full(len(samples), -1)
    column_of_row[loaded_rows] = np.arange(len(loaded_rows))
    return bed[loaded_rows], column_of_row


def build_po_regression(samples: pd.DataFrame, parents: pd.DataFrame, phenotypes: np.ndarray,
                        omega: sp.csr_array, column_of_row: np.ndarray):
    """The PO regression: children with a genotyped parent, regressed on their own phenotypes

    Its phenotype map is a selection matrix, so its phenotypic covariance is the children's block
    of Omega.  It also records where each child's parents' data is.

    Args:
        samples: The sample table, after select_omega_sample()
        parents: identify_parents()'s result
        phenotypes: prepare_phenotypes()'s result, (omega samples,)
        omega: build_omega()'s result
        column_of_row: plan_genotype_samples()'s second result

    Returns:
        A Regression, or None if there are no such children (without a snipar file, for instance)
    """

    if len(parents) == 0:
        return None
    omega_idx = samples[SAMPLE_OMEGA_IDX_COL].to_numpy()
    focal = omega_idx[parents[PARENTS_CHILD_COL].to_numpy()]
    num_omega = len(phenotypes)
    A = sp.csr_array((np.ones(len(focal)), (np.arange(len(focal)), focal)),
                     shape=(len(focal), num_omega))

    def column(rows):
        return np.where(rows != -1, column_of_row[np.maximum(rows, 0)], -1)

    columns = ParentColumns(father=column(parents[PARENTS_FATHER_COL].to_numpy()),
                            mother=column(parents[PARENTS_MOTHER_COL].to_numpy()),
                            family=parents[PARENTS_SNIPAR_FAMILY_COL].to_numpy())
    logging.info(f"\tPO regression: {len(focal)} focal sample(s)")
    return Regression(name=GROUP_PO, focal=focal, phenotype_map=A, phenotypes=A @ phenotypes,
                      gls_weights=calculate_gls_weights(sp.csr_array(A @ omega @ A.T)),
                      parents=columns)


def po_genotype_residuals(genotypes: np.ndarray, snipar_block,
                          regression: Regression) -> np.ndarray:
    """The PO regression's genotype residuals, for a block of SNPs

    For each child at each SNP:
        - both parents genotyped:                 g_c - (g_father + g_mother) / 2
        - one parent genotyped, snipar k = 3:     g_c - g_obs / 2 - t, where t = imputed - f is
                                                  the allele the other parent transmitted
        - one parent genotyped, snipar k = 4:     g_c - (g_obs + g_other) / 2, where g_other =
                                                  imputed is the other parent's genotype
        - anything else -- k = 2 or missing, a SNP snipar lacks, a missing genotype -- is missing,
          and a missing residual is zero.
    t and g_other are rounded to whole numbers (SNIPAR_ALLELE_TOLERANCE).

    Args:
        genotypes: The loaded samples' genotypes, (snps, loaded samples), NaN where missing; the
            first columns are Omega's
        snipar_block: SniparReader.read()'s result for the block, or None without a snipar file
        regression: The PO regression

    Returns:
        The residuals, (snps, children)
    """

    columns = regression.parents
    num_snps, num_children = genotypes.shape[0], len(regression.focal)

    def genotypes_of(cols):
        out = np.full((num_snps, num_children), np.nan)
        has = cols != -1
        out[:, has] = genotypes[:, cols[has]]
        return out

    child = genotypes[:, regression.focal]
    father, mother = genotypes_of(columns.father), genotypes_of(columns.mother)
    both = (columns.father != -1) & (columns.mother != -1)
    observed = np.where(columns.father != -1, father, mother)

    residuals = np.full((num_snps, num_children), np.nan)
    residuals[:, both] = (child - (father + mother) / 2)[:, both]

    one = ~both & (columns.family != -1)
    if snipar_block is not None and one.any():
        imputed, counts, f = snipar_block
        family = columns.family[one]
        k, value = counts[:, family], imputed[:, family]
        g_c, g_obs = child[:, one], observed[:, one]

        transmitted = value - f[:, None]
        other_genotype = value
        result = np.full(k.shape, np.nan)
        unfit = 0
        for count, quantity, formula in (
                (3, transmitted, lambda q: g_c - g_obs / 2 - q),
                (4, other_genotype, lambda q: g_c - (g_obs + q) / 2)):
            at = k == count
            whole = np.rint(quantity)
            fits = at & (np.abs(quantity - whole) <= SNIPAR_ALLELE_TOLERANCE)
            unfit += int((at & ~fits & ~np.isnan(quantity)).sum())
            result = np.where(fits, formula(whole), result)
        if unfit:
            logging.warning(f"{unfit} imputed parental value(s) in this block are not a whole "
                            f"number of alleles as their observed-allele count implies, and are "
                            f"treated as missing")
        residuals[:, one] = result

    # Every residual is an exact multiple of 1/2 -- whole-number genotypes, and snipar's values
    # rounded to whole numbers above -- so unlike a relative regression's, none carries rounding
    # noise needing GENOTYPE_RESIDUAL_TOLERANCE
    return np.nan_to_num(residuals, nan=0.0)


# Regressions: the genotype loop ==================================================================

def relative_genotype_residuals(genotypes: np.ndarray, regression: Regression) -> np.ndarray:
    """A relative regression's genotype residuals, R g, for a block of SNPs

    A missing genotype makes missing every residual it enters -- the focal sample's own, and those
    of the focal samples it is a relative of -- and a missing residual is zero.  Residuals that are
    only rounding noise are zero too (GENOTYPE_RESIDUAL_TOLERANCE).

    Args:
        genotypes: The Omega samples' genotypes, (snps, omega samples), NaN where missing
        regression: The regression

    Returns:
        The residuals, (snps, focal)
    """

    # Only R's stored entries are multiplied, so a NaN reaches only the residuals it enters
    residuals = np.nan_to_num(np.asarray((regression.phenotype_map @ genotypes.T).T), nan=0.0)
    residuals[np.abs(residuals) < GENOTYPE_RESIDUAL_TOLERANCE] = 0.0
    return residuals


def regression_statistics(residuals: np.ndarray, regression: Regression):
    """A regression's estimate for each SNP, and the quantities the covariance of estimates needs

    With Z the genotype residuals, Q the GLS weights and y the regression's phenotypes:
    u = Z Q, d = Z'QZ per SNP, and beta = u y / d.  A SNP with no information in this regression
    (d = 0, e.g. no variation among its samples) has no estimate: NaN.

    Args:
        residuals: The genotype residuals Z, (snps, focal)
        regression: The regression

    Returns:
        A tuple (u, d, beta): (snps, focal), (snps,) and (snps,)
    """

    u = np.asarray((regression.gls_weights @ residuals.T).T)
    d = np.einsum("sf,sf->s", residuals, u)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(d > 0, (u @ regression.phenotypes) / d, np.nan)
    return u, d, beta


def covariance_of_estimates(us: list, ds: list, names: list, cross: dict) -> np.ndarray:
    """The covariance matrix C of the regressions' estimates, for each SNP

    C_ij = u_i (A_i Omega A_j') u_j' / (d_i d_j), the same as W Omega W' with each regression's
    weights w = A'QZ/d spread over all of Omega, but without forming them.  C_ii equals 1/d_i,
    each regression's own GLS variance.  Rows and columns of a regression without an estimate at a
    SNP are NaN there.

    Args:
        us, ds: Each regression's u, (snps, focal_i), and d, (snps,), from regression_statistics()
        names: The regressions' names, in the same order
        cross: build_cross_covariances()'s result

    Returns:
        C, (snps, regressions, regressions)
    """

    num_snps, num = len(ds[0]), len(ds)
    C = np.full((num_snps, num, num), np.nan)
    for i in range(num):
        for j in range(i, num):
            weighted = np.asarray((cross[(names[i], names[j])].T @ us[i].T).T)
            with np.errstate(divide="ignore", invalid="ignore"):
                value = np.einsum("sf,sf->s", weighted, us[j]) / (ds[i] * ds[j])
            valid = (ds[i] > 0) & (ds[j] > 0)
            C[:, i, j] = C[:, j, i] = np.where(valid, value, np.nan)
    return C


def meta_analyze(betas: np.ndarray, C: np.ndarray):
    """The optimally weighted combination of estimates, for each SNP

    Weights a = C^+ 1 / (1' C^+ 1), estimate a'b, standard error sqrt(a' C a), with C^+ the
    pseudo-inverse in case C is singular.  An estimate missing at a SNP is left out there: zeroing
    its row and column of C makes its pseudo-inverse weight exactly zero.

    Args:
        betas: The estimates to combine, (snps, k), NaN where missing
        C: Their covariance, (snps, k, k), NaN where missing

    Returns:
        A tuple (estimate, standard error), each (snps,), NaN where every estimate is missing
    """

    present = ~np.isnan(betas)
    b = np.where(present, betas, 0.0)
    mask = present[:, :, None] & present[:, None, :]
    C = np.where(mask, C, 0.0)

    weights = np.linalg.pinv(C, hermitian=True) @ np.ones(betas.shape[1])
    total = weights.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = weights / total[:, None]
        estimate = np.where(present.any(axis=1), np.einsum("sk,sk->s", a, b), np.nan)
        variance = np.einsum("sk,skl,sl->s", a, C, a)
    se = np.where(present.any(axis=1), np.sqrt(variance), np.nan)
    return estimate, se


def estimate_block(genotypes: np.ndarray, regressions: list, cross: dict, estimates: list,
                   snipar_block=None):
    """Every reported estimate and its standard error, for a block of SNPs

    Args:
        genotypes: The loaded samples' genotypes (plan_genotype_samples()), (snps, loaded
            samples), NaN where missing; the first columns are Omega's
        regressions: The regressions
        cross: build_cross_covariances()'s result
        estimates: reported_estimates()'s result
        snipar_block: SniparReader.read()'s result for the block, or None without a snipar file

    Returns:
        A tuple (betas, ses), each (snps, estimates)
    """

    names = [r.name for r in regressions]
    us, ds, betas = [], [], []
    for regression in regressions:
        if regression.parents is not None:
            residuals = po_genotype_residuals(genotypes, snipar_block, regression)
        else:
            num_omega = regression.phenotype_map.shape[1]
            residuals = relative_genotype_residuals(genotypes[:, :num_omega], regression)
        u, d, beta = regression_statistics(residuals, regression)
        us.append(u)
        ds.append(d)
        betas.append(beta)
    num_snps = genotypes.shape[0]
    all_betas = np.column_stack(betas) if betas else np.empty((num_snps, 0))
    C = (covariance_of_estimates(us, ds, names, cross) if regressions
         else np.empty((num_snps, 0, 0)))

    out_betas = np.full((num_snps, len(estimates)), np.nan)
    out_ses = np.full((num_snps, len(estimates)), np.nan)
    for e, combined in enumerate(estimates):
        idx = [names.index(n) for n in combined if n in names]
        if idx:
            out_betas[:, e], out_ses[:, e] = meta_analyze(all_betas[:, idx],
                                                          C[:, idx][:, :, idx])
    return out_betas, out_ses


# Planning and running ============================================================================

@dataclasses.dataclass
class AnalysisPlan:
    """Everything the genotype loop needs, and nothing else

    Built by plan_analysis() from every input except the genotypes.  Every table used to build it
    is released when plan_analysis() returns, so that memory is left for the genotype data.

    Attributes:
        regressions: The regressions, in REGRESSION_ORDER
        cross: build_cross_covariances()'s result
        estimates: reported_estimates()'s result
        bed_cols: The bed file columns to read, (loaded samples,)
        snps: The SNP plan
        bim: The analysed SNPs' bim rows, for the output
        num_file_snps, num_file_samples: The bed file's dimensions
        imputed_parental_file: snipar's file, or None
        standard_f: snipar's allele frequency for each SNP in its file, or None
    """

    regressions: list
    cross: dict
    estimates: list
    bed_cols: np.ndarray
    snps: SnpPlan
    bim: pd.DataFrame
    num_file_snps: int
    num_file_samples: int
    imputed_parental_file: str = None
    standard_f: np.ndarray = None


def plan_analysis(*, rel_file, bim_file, fam_file, rel_degree, pheno_file=None, covar_file=None,
                  id_list=None, snp_list=None, imputed_parental_file: str = None) -> AnalysisPlan:
    """Reads and checks every input except the genotypes, and builds the analysis plan

    Every check runs here, so a problem with any input is reported before any genotype is read.

    Args:
        As for grma()

    Raises:
        ValueError: If any input fails a check

    Returns:
        The AnalysisPlan
    """

    rel_degree = int(rel_degree)

    logging.info("Reading the samples")
    samples = build_sample_table(
        grma_io.read_fam(fam_file),
        pheno=None if pheno_file is None else grma_io.read_pheno(pheno_file),
        covar=None if covar_file is None else grma_io.read_covar(covar_file),
        id_list=None if id_list is None else grma_io.read_id_list(id_list))

    logging.info("Reading the relatedness file")
    graph = build_relatedness_graph(samples, grma_io.read_king(rel_file))
    samples, graph = resolve_duplicates(samples, graph)

    logging.info("Identifying parents and groups")
    snipar_metadata = (None if imputed_parental_file is None
                       else snipar_hdf5.read_snipar_metadata(imputed_parental_file))
    parents = identify_parents(samples, graph, snipar_metadata)
    samples = assign_groups(samples, graph, parents, rel_degree=rel_degree)

    logging.info("Building Omega")
    samples = select_omega_sample(samples, parents)
    phenotypes = prepare_phenotypes(samples)
    omega = build_omega(samples, graph, phenotypes)

    logging.info("Setting up the regressions")
    bed_cols, column_of_row = plan_genotype_samples(samples, parents)
    po = build_po_regression(samples, parents, phenotypes, omega, column_of_row)
    if snipar_metadata is not None and po is None:
        logging.warning("No sample qualifies for the PO regression, so it is omitted; every "
                        "reported estimate that would include it equals the one without it")
    regressions = (([po] if po is not None else []) +
                   build_relative_regressions(samples, graph, phenotypes, omega,
                                              rel_degree=rel_degree))
    cross = build_cross_covariances(regressions, omega)
    estimates = reported_estimates(has_po=snipar_metadata is not None, rel_degree=rel_degree)

    logging.info("Reading the SNPs")
    bim = grma_io.read_bim(bim_file)
    snps = plan_snps(bim, snp_list=None if snp_list is None else grma_io.read_snp_list(snp_list),
                     snipar_metadata=snipar_metadata)

    return AnalysisPlan(
        regressions=regressions, cross=cross, estimates=estimates, bed_cols=bed_cols, snps=snps,
        bim=bim.iloc[snps.bim_rows].reset_index(drop=True), num_file_snps=len(bim),
        num_file_samples=len(samples), imputed_parental_file=imputed_parental_file,
        standard_f=(None if snipar_metadata is None
                    else snipar_metadata[snipar_hdf5.SNIPAR_STANDARD_F_KEY]))


def run_analysis(plan: AnalysisPlan, bed_file,
                 snps_per_block: int = DEFAULT_SNPS_PER_BLOCK) -> pd.DataFrame:
    """Streams the genotypes through the plan, a block of SNPs at a time

    Args:
        plan: plan_analysis()'s result
        bed_file: As for grma()
        snps_per_block: How many SNPs to process at a time

    Returns:
        GRMA's results, as grma_io.build_output() describes them
    """

    num_snps = len(plan.snps.bim_rows)
    betas = np.full((num_snps, len(plan.estimates)), np.nan)
    ses = np.full((num_snps, len(plan.estimates)), np.nan)
    reader = (grma_io.SniparReader(plan.imputed_parental_file, plan.standard_f)
              if plan.imputed_parental_file is not None else None)
    try:
        for start in range(0, num_snps, snps_per_block):
            block = slice(start, min(start + snps_per_block, num_snps))
            genotypes = grma_io.read_genotype_block(bed_file, plan.num_file_snps,
                                                     plan.num_file_samples,
                                                     plan.snps.bim_rows[block], plan.bed_cols)
            snipar_block = (None if reader is None
                            else reader.read(plan.snps.snipar_positions[block],
                                             plan.snps.flip[block]))
            betas[block], ses[block] = estimate_block(genotypes, plan.regressions, plan.cross,
                                                      plan.estimates, snipar_block=snipar_block)
            logging.debug(f"\tSNPs {block.start} to {block.stop - 1} done")
    finally:
        if reader is not None:
            reader.close()
    logging.info(f"\t{num_snps} SNP(s) analysed")
    return grma_io.build_output(plan.bim, plan.estimates, betas, ses)


def grma(*, rel_file, bed_file, bim_file, fam_file, rel_degree, pheno_file=None,
          covar_file=None, snps_per_block: int = DEFAULT_SNPS_PER_BLOCK, id_list=None,
          snp_list=None, imputed_parental_file: str = None) -> pd.DataFrame:
    """GRMA: plans the analysis from every input but the genotypes, then runs it

    Takes the same keyword arguments as the original implementation's grma(), so the command line
    can switch to it with no other change.  Each input may be a path or, for testing, its contents.

    Args:
        rel_file: KING's relatedness file
        bed_file: The bed file, or its genotypes as an array, (file snps, file samples)
        bim_file, fam_file: The bim and fam files
        rel_degree: The highest relationship degree to run a regression for, 0 to MAX_REL_DEGREE
        pheno_file, covar_file, id_list, snp_list: Optional inputs, as their readers describe
        snps_per_block: How many SNPs to process at a time
        imputed_parental_file: snipar's imputed parental genotype file, or None

    Raises:
        ValueError: If any input fails a check, before any genotype is read

    Returns:
        GRMA's results, as grma_io.build_output() describes them
    """

    plan = plan_analysis(rel_file=rel_file, bim_file=bim_file, fam_file=fam_file,
                         rel_degree=rel_degree, pheno_file=pheno_file, covar_file=covar_file,
                         id_list=id_list, snp_list=snp_list,
                         imputed_parental_file=imputed_parental_file)
    return run_analysis(plan, bed_file, snps_per_block=snps_per_block)


#################################
if __name__ == "__main__":

    print("This script is not meant to be called directly.")
