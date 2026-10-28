#!/usr/bin/env python3

"""
Library / core code for GRMA method
"""

import contextlib
import logging
import operator
import textwrap
import time
from typing import NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import scipy.sparse.linalg as spla
from scipy.stats import norm

from bedbimfam import (
    BIM_A1_COL,
    BIM_A2_COL,
    BIM_BP_COL,
    BIM_CHR_COL,
    BIM_COLS,
    BIM_RSID_COL,
    FAM_COLS,
    FAM_FID_COL,
    FAM_IID_COL,
    FAM_IIDF_COL,
    FAM_IIDM_COL,
    FAM_PHENO_COL,
    FAM_SEX_COL,
    read_bed_file
)
import snipar_hdf5 as snipar

# Copy-on-Write will become the default behaviour in Pandas 3.0 and is turned on to increase clarity
# about whether objects are views or copies
# (https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#)
pd.options.mode.copy_on_write = True

# Calculate constants used in determination of P values
ln = np.log  # pylint: disable=invalid-name
LN_2 = ln(2.0)
RECIP_LN_10 = np.reciprocal(ln(10.0))



# Columns used from the King output (this will need to be adjusted if King output is changed)
KING_FID1_COL = "FID1"
KING_IID1_COL = "ID1"
KING_FID2_COL = "FID2"
KING_IID2_COL = "ID2"
KING_KINSHIP_COL = "Kinship"
KING_REL_COL = "InfType"


NEEDED_KING_COLS = [
    KING_FID1_COL,
    KING_IID1_COL,
    KING_FID2_COL,
    KING_IID2_COL,
    KING_KINSHIP_COL,
    KING_REL_COL,
]


# Two parents, two alleles each: the total snipar's allele counts are measured against
SNIPAR_TOTAL_PARENTAL_ALLELES = 4

# A parent has two alleles, so flipping which allele a genotype counts gives 2 - g
SNIPAR_ALLELES_PER_PARENT = 2

# Complementary bases, for detecting two files recorded on opposite strands
ALLELE_COMPLEMENTS = {"A" : "T", "T" : "A", "C" : "G", "G" : "C"}

# Allele pairs that are their own complement, and so cannot have their strand determined from the
# letters alone
PALINDROMIC_ALLELE_PAIRS = ({"A", "T"}, {"C", "G"})

# Column positions within snipar's pedigree records, re-exported for readability here
SNIPAR_PED_IID_COL = snipar.SNIPAR_PED_IID_COL
SNIPAR_PED_FID_COL = snipar.SNIPAR_PED_FID_COL
SNIPAR_PED_FATHER_COL = snipar.SNIPAR_PED_FATHER_COL
SNIPAR_PED_MOTHER_COL = snipar.SNIPAR_PED_MOTHER_COL

FAM_INDEX = "FAM_INDEX"

FAM_KEY = [FAM_FID_COL, FAM_IID_COL]
KING_KEY_1 = [KING_FID1_COL, KING_IID1_COL]
KING_KEY_2 = [KING_FID2_COL, KING_IID2_COL]

# PLINK uses "0" in a parental ID column to mean that parent is not in the dataset
FAM_MISSING_PARENT = "0"

# PLINK sex codes.  Anything else (0, -9) means unknown.
FAM_MALE_SEX = "1"
FAM_FEMALE_SEX = "2"
FAM_UNKNOWN_SEX = "0"
FAM_KNOWN_SEXES = (FAM_MALE_SEX, FAM_FEMALE_SEX)

FAM_INDEX_1 = "FAM_INDEX_1"
FAM_INDEX_2 = "FAM_INDEX_2"

# For error messages
MAX_REPORTED_OFFENDERS = 10


# Values seen in KING file as InfTypes
INF_DUP_MZTWIN = "Dup/MZTwin"
INF_FULLSIB = "FS"
INF_PARENT_OFFSPRING = "PO"
INF_2ND = "2nd"
INF_3RD = "3rd"
INF_4TH = "4th"
INF_UNRELATED = "UN"

# Values to map InfTypes to
DEG_DUP_MZTWIN = 0
DEG_FULLSIB = 0
DEG_PARENT_OFFSPRING = 1
DEG_2ND = 2
DEG_3RD = 3
DEG_4TH = 4
DEG_UNRELATED = 5

# Map of possible InfTypes in a King output file to degree values 
INFTYPE_TO_DEG_MAP = {
    INF_DUP_MZTWIN: DEG_DUP_MZTWIN,
    INF_FULLSIB: DEG_FULLSIB,
    INF_PARENT_OFFSPRING: DEG_PARENT_OFFSPRING,
    INF_2ND: DEG_2ND,
    INF_3RD: DEG_3RD,
    INF_4TH: DEG_4TH,
    INF_UNRELATED: DEG_UNRELATED
    }


MAX_GRMA_RELATEDNESS = 3  # Maximum degree of relatedness GRMA currently handles
SE_RELATEDNESS = 3  # Relatedness to include in SE calculations


# Default number of SNPs to process at a time
DEFAULT_SNPS_PER_BLOCK = 100

# Default values in omega shrinking procedure
DEFAULT_OMEGA_EPSILON = 1e-10
DEFAULT_EIGSH_TOL = 1e-8

# Matrices at or below this size use a dense eigendecomposition instead of ARPACK (exact,
# deterministic, and faster at this scale).  Relatedness blocks are nearly always well under this.
DENSE_LAMBDA_MIN_MAX_SIZE = 500

# Fixed seed for the ARPACK start vector, so repeated runs give identical results
EIGSH_START_VECTOR_SEED = 8675309

# Output columns
OUTPUT_BETA_COL = 'BETA'
OUTPUT_SE_COL = 'SE'
OUTPUT_P_COL = 'P'
OUTPUT_SUMSQX_COL = 'SUM_SQ_X'


# -------------------------

class ParentalSources(NamedTuple):
    """Describes whose genotypes are residualized against their parents

    Produced by either branch of parental processing -- fam file trio determination or imputed
    parent processing -- and consumed downstream without regard to which branch ran.

    Sentinels operate at two levels.  imputed_rows is None entirely when no imputed parental
    genotype file was supplied; a -1 within an array means that particular parent, or that
    particular family's imputed row, does not exist.

    Attributes:
        affected_samples: (n,) indices into the original, unfiltered sample space
        observed_parent_cols: (n, 2) sample indices of each individual's genotyped parents, with
            -1 where that parent is not genotyped
        imputed_rows: (n,) row index into the imputed parental genotype arrays, -1 where the
            family has no imputed row because both parents are genotyped.  None entirely in fam
            file trio mode
        imputed_file: Path to the imputed parental genotype file, reopened around the block loop
            rather than held open here.  None in fam file trio mode
        snipar_metadata: Output of read_snipar_metadata(), or None in fam file trio mode
    """

    affected_samples: np.ndarray
    observed_parent_cols: np.ndarray
    imputed_rows: np.ndarray
    imputed_file: str
    snipar_metadata: dict


# -------------------------

def get_df(filename: Union[str, pd.DataFrame], df_name: str, read_csv_params: dict = None,
           mi_key: list = None) -> Tuple[pd.DataFrame, pd.MultiIndex]:

    """Gets DataFrame from file (along with optional MultiIndex)

    Helper function to retrieve data from a file and return it in Pandas Dataframe format.  Uses
    default settings but allows for overrides and the possibility of creating a returning a 
    MultiIndex.  Also allows for the possibilty of the Dataframe already being passed in,
    essentially as a no-op, which is useful for testing or library purposes.

    Args:
        filename: String path to the file, or the Dataframe itself (already read in)
        df_name: Name of the Dataframe, only for error / logging purposes
        read_csv_params: Dictionary for specification / override of read_csv() function parameters
        mi_key: Optional column names to allow for a MultiIndex to be returned along with the frame

    Returns:
        A tuple (df, mi) where df is the DataFrame in question and mi is a MultiIndex on the 
        columns specified by mi_key (or None if no columns were specified)

    """

    read_csv_defaults = {
        "sep" : r"\s+",
        "index_col" : False,
        "header" : None
    }
    if read_csv_params:
        read_csv_defaults.update(read_csv_params) 

    if isinstance(filename, str):
        if not filename:
            return None, None
        df = pd.read_csv(filename, **read_csv_defaults)
    elif isinstance(filename, pd.DataFrame):
        df = filename
    elif filename is None:
        return None, None
    else:
        raise TypeError(f"Expected str or Pandas DataFrame for parameter {df_name}, but "
                        f"received {type(filename)}")

    mi = pd.MultiIndex.from_frame(df[mi_key]) if mi_key else None

    return df, mi


# TODO(jonbjala) Check various files for duplicates?
def process_phenotypes(fam_file: Union[str, pd.DataFrame], *,
                       pheno_file: Union[str, pd.DataFrame]=None,
                       sample_id_file: Union[str, pd.DataFrame]=None,
                       covar_file: Union[str, pd.DataFrame]=None
    ) -> Tuple[int, int, np.ndarray, pd.DataFrame, np.ndarray]:
    """Performs processing related to raw phenotypes and sample IDs

    Function to read in a fam file along with various optional files that are relevant to this
    portion of the GRMA processing, including filtering by sample ID, overriding the fam phenotypes,
    and residualizing the phenotype values on covariates.  There is also checking for missing/NaN
    values.  No input file should contain any column name headers.

    Args:
        fam_file: String path to the fam file, or the Dataframe itself (already read in).  This
            needs to contain the columns specified by Plink.
            (see https://www.cog-genomics.org/plink/1.9/formats#fam)
        pheno_file: String path to the pheno file, or the Dataframe itself (already read in).  This
            needs to contain the columns for a simple, phenotype file as specified by Plink.
            (see https://www.cog-genomics.org/plink/1.9/input#pheno)
        sample_id_file: String path to the sample ID file, or the Dataframe itself (already read in)
            This needs to contain two columns, FID and IID.
        covar_file: String path to the sample ID file, or the Dataframe itself (already read in).  
            This needs to contain two columns, FID and IID, along with N other columns (where N > 0)
            that contain the covariate values to use for residualizing the phenotypes.

    Returns:
        A tuple (N_orig, N, sample_filter, fam_id_cols, phenotypes) where:

            N_orig is the initial number of samples in the fam file
            N is the final number of samples after intersecting with any of the
                optional inputs (pheno file, sample ID file, covariates file) and dropping samples with 
                NaN/missing values.
            sample_filter is the list of numerical (values from 0 to N_orig-1) indices to use when
                filtering, say, the values in the bed file by sample
            fam_id_cols is what is left of the fam DataFrame after all the sample filtering is 
                complete and the columns are restricted down to FID and IID
            phenotypes is array of phenotypes not yet residualized on relatedness
    """

    # Read in fam file and take note of starting number of samples (before any filtering)
    fam_name = fam_file if isinstance(fam_file, str) else "fam file"
    fam_df, fam_mi = get_df(fam_file, "fam_file", {"names" : FAM_COLS}, FAM_KEY)
    fam_df = fam_df.reset_index(names=FAM_INDEX).set_index(FAM_KEY, drop=False)
    N_orig = len(fam_df)
    total_intersection = fam_mi.copy()
    logging.info(f"\tThere are {N_orig} samples in {fam_name}")
    logging.debug(f"\tThere are {len(total_intersection)} samples left in intersection\n")


    # Read in id_list file (if specified) and find intersection of sample IDs
    _, sample_mi = get_df(sample_id_file, "sample_id_file",
                          {"usecols" : (0,1), "names" : FAM_KEY}, FAM_KEY)
    if sample_id_file is not None:
        sample_name = sample_id_file if isinstance(sample_id_file, str) else "sample id file"
        total_intersection = total_intersection.intersection(sample_mi)
        logging.info(f"\tThere are {len(sample_mi)} samples in {sample_name}")
        logging.debug(f"\tThere are {len(total_intersection)} samples left in intersection\n")


    # Read in phenotype file (if specified) and find intersection of sample IDs
    pheno_df, pheno_mi = get_df(pheno_file, "pheno_file",
                                {"names":[FAM_FID_COL, FAM_IID_COL, FAM_PHENO_COL]}, FAM_KEY)
    if pheno_file is not None:
        pheno_name = pheno_file if isinstance(pheno_file, str) else "phenotype file"
        total_intersection = total_intersection.intersection(pheno_mi)
        logging.info(f"\tThere are {len(pheno_mi)} samples in {pheno_name}")
        logging.debug(f"\tThere are {len(total_intersection)} samples left in intersection\n")


    # Read in covariates file (if specified) and find intersection of sample IDs
    covar_df, _ = get_df(covar_file, "covar_file")
    if covar_file is not None:
        covar_name = covar_file if isinstance(covar_file, str) else "covariates file"
        n_covars = covar_df.shape[1] - 2 # Since there should be two initial columns, FID and IID
        if n_covars < 1:
            raise ValueError(f"Covariates file should contain at least 3 columns")
        if covar_df.iloc[:, 2:].isna().values.any():
            num_orig_covar_rows = len(covar_df)
            logging.info(f"\tThere are {num_orig_covar_rows} samples in {covar_name}")
            covar_df = covar_df[covar_df.iloc[:, 2:].notna().all(axis=1)]
            num_nan_rows = num_orig_covar_rows - len(covar_df)
            logging.warning(f"Covariates file contains NaN values.  Dropped {num_nan_rows} samples.\n")
        covar_df.columns = FAM_KEY + [f"COVAR_{i+1}" for i in range(n_covars)]
        covar_mi = pd.MultiIndex.from_frame(covar_df[FAM_KEY])

        total_intersection = total_intersection.intersection(covar_mi)
        logging.info(f"\tThere are {len(covar_mi)} (non-NaN-containing) samples in {covar_name}")
        logging.debug(f"\tThere are {len(total_intersection)} samples left in intersection\n")


    # Restrict down to the intersection of available samples
    if any(f is not None for f in (sample_id_file, pheno_file, covar_file)):
        logging.info(f"\tThere are {len(total_intersection)} samples in the intermediate intersection")
        if len(total_intersection) < N_orig:
            logging.info(f"\tRestricting to these samples\n")
            fam_df = fam_df.reindex(total_intersection)
            fam_mi = total_intersection

    # Reassign the phenotype values to the ones from the phenotype file (override fam values)
    if pheno_file is not None:
        logging.info(f"\tExtracting phenotype values from the provided phenotype file")
        pheno_s = pheno_df.set_index(FAM_KEY)[FAM_PHENO_COL]
        fam_df[FAM_PHENO_COL] = pheno_s.reindex(fam_df.index).to_numpy()


    # Drop any NaN / missing values
    fam_df = fam_df.dropna(subset=[FAM_PHENO_COL])
    num_nan = len(total_intersection) - len(fam_df)
    if num_nan > 0:
        logging.info(f"\tDropped {num_nan} samples with missing/NaN phenotype values.")
    fam_mi = fam_df.index

    # Residualize phenotypes on an intercept plus any covariates.  With no covariates this is
    # centering on the mean, which matters: an affected sample's row of R is an identity row, so its
    # phenotype passes through unchanged, and an uncentered one would make the betas depend on the
    # phenotype's origin (notes §13.5)
    design = np.ones((len(fam_df), 1))
    if covar_file is not None:
        logging.info(f"\tRunning regression to residualize phenotypes on covariates")
        covar_df = covar_df.set_index(FAM_KEY, drop=False).reindex(fam_mi)
        design = np.column_stack([covar_df.iloc[:, 2:].to_numpy(dtype=float), design])
    else:
        logging.info(f"\tCentering phenotypes on their mean")

    orig_phenotypes = fam_df[FAM_PHENO_COL].to_numpy(dtype=float)
    x, _, _, _ = np.linalg.lstsq(a = design, b = orig_phenotypes, rcond = None)
    fam_df[FAM_PHENO_COL] = orig_phenotypes - design @ x


    # Make sure fam_df is in sorted order based on original index
    fam_df = fam_df.sort_values(FAM_INDEX)

    # Grab the phenotype values and the final value for N
    N = len(fam_df)
    phenotypes = fam_df[FAM_PHENO_COL].to_numpy()

    # Debug logging
    sample_ids = fam_df[FAM_KEY].reset_index(drop=True)
    logging.debug(f"\t{N_orig=} {N=} {len(sample_ids)=}")

    return (N_orig, N, fam_df[FAM_INDEX].to_numpy(), sample_ids, phenotypes)


# -------------------------
def get_rel_report(rel_df: pd.DataFrame) -> str:
    rel_counts = rel_df[KING_REL_COL].value_counts(sort=False).sort_index().to_dict()
    return "{" + ", ".join(f"{deg}->{count}" for deg, count in rel_counts.items()) + "}"


def get_lambda_min(B: Union[np.ndarray, sp.csr_array], tol=DEFAULT_EIGSH_TOL) -> float:

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


def get_rel_covariances(rel_df: pd.DataFrame,
                        unresidualized_phenotypes: np.ndarray) -> dict[int, float]:
    rel_to_cov = {rel_deg : 
        np.cov(unresidualized_phenotypes[np.concatenate([df_view[FAM_INDEX_1].to_numpy(np.int64),
                                                         df_view[FAM_INDEX_2].to_numpy(np.int64)])],
               unresidualized_phenotypes[np.concatenate([df_view[FAM_INDEX_2].to_numpy(np.int64),
                                                         df_view[FAM_INDEX_1].to_numpy(np.int64)])],
               ddof=0)[0, 1] if len(df_view) > 0 else np.nan
                  for rel_deg in range(SE_RELATEDNESS+1)
                  for df_view in [rel_df[rel_df[KING_REL_COL] == rel_deg]]}

    logging.debug(f"\t{rel_to_cov=}\n")

    return rel_to_cov


# -------------------------
def get_relatedness_components(rel_df: pd.DataFrame, N: int) -> np.ndarray:
    """Assigns each sample to a connected component of the relatedness graph

    Both the per-block shrinkage in calculate_omega() and the block-wise inversion in
    calculate_gls_weights() need the same partition, and both must use omega's components rather
    than R's.  Computing it once here makes that a single decision instead of a constraint two
    functions have to independently honour.

    The partition depends only on WHICH pairs exist, not on their covariances, so it is available
    before any phenotype data is involved.

    Args:
        rel_df: Relatedness pairs, mapped to sample indices and integer degrees
        N: Number of samples

    Returns:
        Component index of each sample
    """

    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)

    adjacency = sp.coo_array((np.ones(2 * len(i1)),
                              (np.concatenate([i1, i2]), np.concatenate([i2, i1]))),
                             shape=(N, N)).tocsr()

    n_components, labels = connected_components(adjacency, directed=False)
    logging.debug(f"\t{n_components} connected components in the relatedness graph")

    return labels


# -------------------------
def calculate_gls_weights(R: sp.csr_array, omega: sp.csr_array, labels: np.ndarray,
                          epsilon=DEFAULT_OMEGA_EPSILON) -> sp.csr_array:
    """Builds the GLS weight matrix

    The regression weights observations by the inverse of their covariance, so that correlated
    relatives -- whose observations overlap and therefore carry less independent information than
    the same number of unrelated people -- are discounted accordingly.  The covariance of the
    residualized phenotypes is R @ omega @ R.T, and the weight is its pseudo-inverse.

    A pseudo-inverse rather than an inverse because R is rank-deficient, so that product is
    singular by construction.  It is the correct object rather than a workaround: since omega is
    positive definite, the product's range equals R's, and the regressor is itself R-residualized
    and therefore lies within it -- which is the condition under which pseudo-inverse GLS is the
    best linear unbiased estimator rather than an approximation.

    The inversion is done block by block.  R and omega are both block diagonal over the
    relatedness components, so their product is too, and the pseudo-inverse of a block diagonal
    matrix is the block diagonal matrix of pseudo-inverses -- exactly, not approximately.  That
    keeps the result as sparse as the input, where a dense inverse would be unusable at scale.

    Note that where every pair within a component shares one relatedness degree, R @ omega @ R.T
    reduces to a scalar multiple of R, its pseudo-inverse weights every direction equally, and
    the estimates coincide exactly with ordinary least squares.  The weighting only does work
    where a component contains pairs at differing degrees.

    Args:
        R: The residualization matrix
        omega: The phenotypic covariance matrix, already shrunk
        labels: Component index of each sample, from get_relatedness_components()
        epsilon: Smallest eigenvalue omega was shrunk to retain

    Returns:
        The GLS weight matrix, block diagonal over the relatedness components
    """

    resid_pheno_cov = R @ omega @ R.T

    N = R.shape[0]
    n_components = labels.max() + 1 if len(labels) > 0 else 0

    # No components means no samples, and the assembly below cannot concatenate an empty list of
    # blocks.  Returning here keeps that from being a crash.
    if n_components == 0:
        return sp.csr_array((N, N))

    order = np.argsort(labels, kind="stable")
    starts = np.searchsorted(labels[order], np.arange(n_components))
    ends = np.searchsorted(labels[order], np.arange(n_components), side="right")

    rows, cols, data = [], [], []
    for component in range(n_components):
        idx = order[starts[component]:ends[component]]
        block = resid_pheno_cov[idx][:, idx].toarray()

        # numpy's default cutoff for treating a direction as null space, which separates float
        # noise from everything else.  An explicit one was tried and removed: expressing it
        # relative to epsilon drove rcond to 1.0 for any block whose largest entry fell below
        # epsilon, which discards every direction including the largest and silently gives every
        # sample in that block zero weight.
        block_pinv = np.linalg.pinv(block)

        nonzero = np.nonzero(block_pinv)
        rows.append(idx[nonzero[0]])
        cols.append(idx[nonzero[1]])
        data.append(block_pinv[nonzero])


    gls_weights = sp.coo_array((np.concatenate(data),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(N, N)).tocsr()

    logging.debug(f"\tGLS weight matrix has {gls_weights.nnz} nonzeros over "
                  f"{n_components} components")

    return gls_weights


# -------------------------
def calculate_gls_estimates(genotypes: np.ndarray, gls_weights: sp.csr_array,
                            weighted_phenotypes: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the GLS effect estimates and their standard errors

    Betas and standard errors share their expensive term, so they are produced together rather
    than by separate functions that would each have to compute it.

    A SNP carrying no information -- monomorphic, entirely missing, or residualized away to
    nothing -- yields XtWX = 0, hence a NaN beta and an infinite standard error.  Both are left
    as they are rather than substituted: an infinite standard error means zero weight in any
    downstream inverse-variance weighting, which is the safe direction to fail in.

    Args:
        genotypes: Residualized genotypes for this block, (snps, samples)
        gls_weights: The GLS weight matrix
        weighted_phenotypes: gls_weights @ resid_phenotypes, computed once outside the block loop

    Returns:
        A tuple (betas, ses, XtWX)
    """

    weighted_genotypes = gls_weights @ genotypes.T
    XtWX = np.einsum('ij,ij->j', weighted_genotypes, genotypes.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        betas = np.einsum('ij,j->i', genotypes, weighted_phenotypes) / XtWX
        ses = np.reciprocal(np.sqrt(XtWX))

    return betas, ses, XtWX

def calculate_omega(rel_df: pd.DataFrame, N: int, unresidualized_phenotypes: np.ndarray,
                    labels: np.ndarray, epsilon=DEFAULT_OMEGA_EPSILON) -> sp.csr_array:

    rel_to_cov = get_rel_covariances(rel_df=rel_df,
                                     unresidualized_phenotypes=unresidualized_phenotypes)


    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)
    rel = rel_df[KING_REL_COL]

    rows = np.concatenate([i1, i2])
    cols = np.concatenate([i2, i1])
    data = np.tile(rel.map(rel_to_cov).to_numpy(), 2)

    # get_rel_covariances() returns NaN for any degree with no pairs, so a degree present in
    # rel_df but absent from rel_to_cov maps to NaN here.  That should be unreachable (rel_df is
    # filtered to <= SE_RELATEDNESS before this point, and rel_to_cov is keyed over that same
    # range), but it must not pass silently: the Gershgorin screen in get_block_alphas() compares
    # against a row sum, and every comparison against NaN is False, so a NaN block would fail the
    # candidate test, keep alpha=1 without an eigendecomposition, and propagate NaN into omega.
    if not np.all(np.isfinite(data)):
        bad_degrees = sorted(rel[~np.isfinite(rel.map(rel_to_cov))].unique())
        raise ValueError(f"No covariance could be computed for relatedness degree(s) "
                         f"{bad_degrees}, which are present in the relatedness data.  This "
                         f"would silently produce a NaN omega.")

    off_diag = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()

    pheno_variance = np.var(unresidualized_phenotypes)
    logging.debug(f"\t{pheno_variance=}")

    alphas = get_block_alphas(off_diag, pheno_variance, labels, epsilon)

    # Scale each entry by the alpha of the component it belongs to.  Both members of a pair are in
    # the same component by construction, so a nonzero's column index identifies its component
    # just as well as its row index does.
    off_diag.data *= alphas[labels[off_diag.indices]]

    omega = pheno_variance * sp.eye(N) + off_diag

    return omega


# -------------------------
def empty_parental_sources() -> ParentalSources:
    """
    Returns a ParentalSources describing no affected individuals, so that callers need not
    distinguish "no parental residualization was requested" from "parental residualization was
    requested but applied to nobody".
    """

    return ParentalSources(affected_samples=np.array([], dtype=np.int64),
                           observed_parent_cols=np.empty((0, 2), dtype=np.int64),
                           imputed_rows=None,
                           imputed_file=None,
                           snipar_metadata=None)


# -------------------------
def as_id_strings(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Returns a copy of the given columns with every value as a string.

    Pandas infers the dtype of each column when reading, so a fam file whose IDs happen to be
    purely numeric yields int64 where one with alphanumeric IDs yields str.  Comparisons then
    behave differently between the two: 0 == "0" is False, so a missing-parent test written
    against the string marker would treat every row of a numeric file as listing both parents.
    The same inference applies to FID and IID, so sample matching is exposed as well.

    Forcing the dtype at read time is not sufficient, because a DataFrame passed directly into
    GRMA (as the tests do) never goes through read_csv at all.

    Args:
        df: Frame containing the columns
        columns: Names of the columns to convert

    Returns:
        The requested columns, as strings
    """

    return df[columns].astype(str)


# -------------------------
def describe_offenders(ids, noun: str = "individual(s)",
                       limit: int = MAX_REPORTED_OFFENDERS) -> str:
    """Summarises a list of offending identifiers for an error message

    The count comes first because it is what distinguishes a handful of data entry errors from
    something systematic, and the list is capped because a systematic problem can name every
    individual in the cohort -- which is unreadable in a terminal and unhelpful in a log.

    Args:
        ids: The offending identifiers, as any sequence.  They are rendered as plain strings, so
            that a numpy array of them does not print its element type alongside each one
        noun: What they are, so that the result reads as a phrase
        limit: How many to name before summarising the rest

    Returns:
        A string naming how many there are and up to limit of them.  Where there are more
        than limit, which ones are named is arbitrary
    """

    count = len(ids)
    if count <= limit:
        return f"{count} {noun}: {sorted(str(identifier) for identifier in ids)}"

    # Take the sample before sorting rather than after.  Sorting the whole list to show ten of
    # them would be the expensive part of handling an error, and the cohort-sized case is exactly
    # the one worth not making slow.  The ten are therefore arbitrary rather than the first ten
    # in order, which is fine for examples.
    sample = sorted(str(identifier) for identifier in ids[:limit])

    # Built by hand rather than from the list's own repr, so that it can be left visibly open:
    # a list printed whole reads as the full set whatever the wording in front of it says.
    named = ", ".join(repr(identifier) for identifier in sample)

    return f"{count} {noun}, including: [{named}, ...]"

# -------------------------
def validate_pedigree_sanity(candidates: pd.DataFrame):
    """Checks the fam file's parental ID columns for a malformed pedigree

    All conditions are fatal: each means the input is wrong rather than merely incomplete, and
    detecting bad input cheaply is preferable to producing results that are quietly wrong.  All
    offending IDs are reported, since the user will want to fix them in one pass.

    Args:
        candidates: Rows of the fam file that list both parents, with ID columns as strings

    Raises:
        ValueError: If any malformed condition is found
    """

    self_referential = candidates[(candidates[FAM_IID_COL] == candidates[FAM_IIDF_COL]) |
                                  (candidates[FAM_IID_COL] == candidates[FAM_IIDM_COL])]
    if len(self_referential) > 0:
         raise ValueError(f"The fam file lists "
                         f"{describe_offenders(self_referential[FAM_IID_COL])} as their own "
                         f"parent")

    # Rows where both parents are absent are not a pedigree error, and callers are not relied on
    # to have filtered them out.  Note this catches only PLINK's marker: a file using something
    # else for "no parent here" will arrive with those rows looking like real IDs, which is what
    # the error below mentions.
    both_parents_absent = ((candidates[FAM_IIDF_COL] == FAM_MISSING_PARENT) &
                           (candidates[FAM_IIDM_COL] == FAM_MISSING_PARENT))

    identical_parents = candidates[~both_parents_absent &
                                   (candidates[FAM_IIDF_COL] == candidates[FAM_IIDM_COL])]
    if len(identical_parents) > 0:
        raise ValueError(f"The fam file lists the same individual as both parents of "
                         f"{describe_offenders(identical_parents[FAM_IID_COL])}.  Note that GRMA "
                         f"treats \"{FAM_MISSING_PARENT}\" as meaning a parent is absent; a file "
                         f"using something else for that would reach this check with those rows "
                         f"looking like real IDs.")


# -------------------------
def validate_parent_sexes(full_fam_df: pd.DataFrame, candidates: pd.DataFrame,
                          father_pos: np.ndarray, mother_pos: np.ndarray):
    """Confirms that no individual has two genetic parents of the same recorded sex

    This is a data-quality signal rather than a correctness issue, and so warns rather than
    stopping the run.  The parental mean is symmetric in the two parents, so mislabelling which is
    which changes no result -- but two same-sex genetic parents mean the pedigree or the sex
    coding is wrong, which probably affects other things too.  The other pedigree checks are fatal
    because what they catch has no sensible reading at all; this one has.

    A trio where either parent's sex is unknown cannot be checked at all, which is common: GRMA
    does not otherwise use the sex column, and many fam files leave it unpopulated.  Those are
    reported as a count rather than silently passed over, since silence would otherwise mean both
    "checked and fine" and "could not be checked".

    Args:
        full_fam_df: The unfiltered fam file
        candidates: Rows of the fam file that list both parents
        father_pos: Positions of their fathers in full_fam_df, -1 where not genotyped
        mother_pos: Positions of their mothers in full_fam_df, -1 where not genotyped

    """

    sexes = full_fam_df[FAM_SEX_COL].astype(str).to_numpy()
    both_genotyped = (father_pos != -1) & (mother_pos != -1)

    father_sex = np.where(both_genotyped, sexes[father_pos], FAM_UNKNOWN_SEX)
    mother_sex = np.where(both_genotyped, sexes[mother_pos], FAM_UNKNOWN_SEX)

    known = np.isin(father_sex, FAM_KNOWN_SEXES) & np.isin(mother_sex, FAM_KNOWN_SEXES)

    # Saying nothing would mean two different things at once -- that the check passed, and that
    # it could not run -- and the user has no way to tell which they got.  One line covers it
    # however many there are.
    unchecked = both_genotyped & ~known
    if np.any(unchecked):
        logging.info(f"\t\tSex is not recorded for one or both parents of "
                     f"{describe_offenders(candidates[FAM_IID_COL].to_numpy()[unchecked])}, so "
                     f"they could not be checked for same-sex parents")

    same_sex = known & (father_sex == mother_sex)
    if np.any(same_sex):
        logging.warning(f"The fam file lists two parents of the same recorded sex as the genetic "
                        f"parents of "
                        f"{describe_offenders(candidates[FAM_IID_COL].to_numpy()[same_sex])}.  "
                        f"Results are unaffected -- the parental mean is symmetric in the two "
                        f"parents -- but a pedigree or a sex column wrong in this way may well "
                        f"be wrong in others.")


# -------------------------
def validate_parents_against_king(iids: np.ndarray, child_pos: np.ndarray,
                                  father_pos: np.ndarray, mother_pos: np.ndarray,
                                  usable: np.ndarray, rel_df: pd.DataFrame,
                                  source: str = "the fam file"):
    """Confirms that every stated parent-child relationship is classified as PO by KING

    Used by both parental processing branches, since a contradicted pedigree is the same class of
    error whichever file asserted it.  The fam file and snipar's pedigree are potentially
    independent artifacts -- a user may have received one from a collaborator and produced the
    other themselves -- so agreement between either and KING is real evidence rather than a
    restatement.

    A pair KING classifies as something else is a contradiction.  So is a pair KING does not
    mention at all: KING reports the relationships it detects, so the absence of a
    parent-offspring pair means it did not detect one.  Both kill the run, and every offending
    pair is collected before raising so the user can fix them in one pass.

    Only relationships that will actually be used are checked.  Both sources describe individuals
    beyond those contributing to this analysis, and failing over a pedigree error involving
    someone who affects no result would be unhelpful.

    Args:
        iids: Individual IDs, for the error message
        child_pos: Sample positions of those individuals, -1 where not a sample
        father_pos: Sample positions of their fathers, -1 where not genotyped
        mother_pos: Sample positions of their mothers, -1 where not genotyped
        usable: Which entries will actually be used
        rel_df: Relatedness pairs, mapped to sample indices and integer degrees
        source: Where the relationships came from, for the error message

    Raises:
        ValueError: If any stated relationship is contradicted by KING
    """

    # Keyed both ways round, since the stated direction (child to parent) need not match the
    # order in which KING recorded the pair
    po_df = rel_df[rel_df[KING_REL_COL] == DEG_PARENT_OFFSPRING]
    king_po_pairs = set(zip(po_df[FAM_INDEX_1], po_df[FAM_INDEX_2]))
    king_po_pairs |= set(zip(po_df[FAM_INDEX_2], po_df[FAM_INDEX_1]))

    contradicted = []
    for row in np.flatnonzero(usable):
        for parent_pos, label in ((father_pos[row], "father"), (mother_pos[row], "mother")):
            # A parent that was not genotyped states no relationship KING could confirm
            if parent_pos == -1:
                continue
            if (child_pos[row], parent_pos) not in king_po_pairs:
                contradicted.append(f"{iids[row]} ({label})")

    if contradicted:
        raise ValueError(f"{source} states parent-offspring relationships that the KING data does "
                         f"not confirm.  KING either classifies these pairs as something other "
                         f"than parent-offspring, or does not report them at all, which means it "
                         f"did not detect the relationship.  Affected: "
                         f"{describe_offenders(contradicted, noun='relationship(s)')}")


# -------------------------
def determine_fam_file_trios(fam_file: Union[str, pd.DataFrame], fam_df: pd.DataFrame,
                             rel_df: pd.DataFrame) -> ParentalSources:
    """Identifies full trios from the fam file's parental ID columns

    The ORIGINAL fam file is read rather than the already-filtered frame, for two reasons: the
    parental ID columns do not survive process_phenotypes(), and a parent dropped for a missing
    phenotype still has genotypes that are usable here.  An individual must be in the analysis
    sample to be affected, but their parents need not be.

    A listed parent who is not a sample in the fam file simply means that individual is not a full
    trio, and ordinary within-group residualization continues to apply to them.  A listed parent
    who IS a sample but whom KING does not classify as parent-offspring is a contradiction between
    the two inputs, and kills the run.

    Args:
        fam_file: String path to the fam file, or the DataFrame itself, UNFILTERED
        fam_df: The already-filtered sample IDs, as returned by process_phenotypes()
        rel_df: Relatedness pairs, mapped to sample indices and integer degrees

    Returns:
        A ParentalSources describing the full trios found

    Raises:
        ValueError: On a malformed pedigree, or a stated parent contradicted by KING
    """

    logging.info("\tDetermining trios from the fam file's parental ID columns")

    full_fam_df, _ = get_df(fam_file, "fam_file", {"names" : FAM_COLS, "dtype" : str})
    id_cols = [FAM_FID_COL, FAM_IID_COL, FAM_IIDF_COL, FAM_IIDM_COL]
    full_fam_ids = as_id_strings(full_fam_df, id_cols)

    has_father = full_fam_ids[FAM_IIDF_COL] != FAM_MISSING_PARENT
    has_mother = full_fam_ids[FAM_IIDM_COL] != FAM_MISSING_PARENT
    candidates = full_fam_ids[has_father & has_mother]

    logging.info(f"\t\t{len(candidates)} individual(s) list both parents in the fam file")
    if len(candidates) == 0:
        return empty_parental_sources()

    validate_pedigree_sanity(candidates)

    # Parental IDs are within-family, so a parent's key is (this row's FID, the parental ID)
    full_fam_mi = pd.MultiIndex.from_frame(full_fam_ids[FAM_KEY])
    child_pos = full_fam_mi.get_indexer(pd.MultiIndex.from_frame(candidates[FAM_KEY]))
    father_pos = full_fam_mi.get_indexer(
        pd.MultiIndex.from_arrays([candidates[FAM_FID_COL], candidates[FAM_IIDF_COL]]))
    mother_pos = full_fam_mi.get_indexer(
        pd.MultiIndex.from_arrays([candidates[FAM_FID_COL], candidates[FAM_IIDM_COL]]))

    validate_parent_sexes(full_fam_df, candidates, father_pos, mother_pos)

    # Only individuals in the analysis sample can be affected, though their parents need not be
    analysis_mi = pd.MultiIndex.from_frame(as_id_strings(fam_df, FAM_KEY))
    in_analysis = analysis_mi.get_indexer(pd.MultiIndex.from_frame(candidates[FAM_KEY])) != -1

    genotyped_parents = (father_pos != -1) & (mother_pos != -1)
    usable = in_analysis & genotyped_parents

    logging.info(f"\t\t{int((in_analysis & ~genotyped_parents).sum())} not usable as trios "
                 f"(a listed parent is not genotyped)")

    validate_parents_against_king(iids=candidates[FAM_IID_COL].to_numpy(), child_pos=child_pos,
                                  father_pos=father_pos, mother_pos=mother_pos, usable=usable,
                                  rel_df=rel_df, source="The fam file")

    logging.info(f"\t\t{int(usable.sum())} full trio(s) validated against KING")

    if not np.any(usable):
        return empty_parental_sources()

    return ParentalSources(
        affected_samples=child_pos[usable],
        observed_parent_cols=np.column_stack([father_pos[usable], mother_pos[usable]]),
        imputed_rows=None,
        imputed_file=None,
        snipar_metadata=None)

# -------------------------
def map_samples_to_imputed_families(pedigree: np.ndarray, families: np.ndarray,
                                    full_fam_ids: pd.DataFrame
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolves each pedigree record to its parents' bed positions and its imputed row

    snipar's pedigree supplies the relationships; the genotypes of any parent that WAS genotyped
    still come from the bed file, so both have to be looked up.  Note families is sorted
    lexicographically while pedigree is not, so the two must be matched by value rather than by
    position.

    Args:
        pedigree: (n, 6) snipar pedigree records, header row already removed
        families: (f,) family IDs, indexing the rows of the imputed parental genotype arrays
        full_fam_ids: The unfiltered fam file's ID columns, as strings

    Returns:
        A tuple (child_pos, parent_pos, imputed_rows) of (n,), (n, 2) and (n,) arrays holding
        positions into the unfiltered fam file and into the imputed arrays, with -1 wherever
        there is no match

    Raises:
        ValueError: If the fam file repeats an IID that the pedigree refers to
    """

    # snipar records relationships by IID alone rather than by FID and IID together, so a repeated
    # IID is ambiguous -- but only where the pedigree actually refers to it.  Two unrelated
    # samples sharing an IID under different FIDs have nothing to do with the families snipar
    # imputed for, and should not stop the run.
    positions = pd.Series(np.arange(len(full_fam_ids)), index=full_fam_ids[FAM_IID_COL])
    ambiguous = positions.index.duplicated(keep=False)

    referenced = (set(pedigree[:, SNIPAR_PED_IID_COL]) |
                  set(pedigree[:, SNIPAR_PED_FATHER_COL]) |
                  set(pedigree[:, SNIPAR_PED_MOTHER_COL]))
    clashing = sorted(set(positions.index[ambiguous]) & referenced)
    if clashing:
        raise ValueError(f"The imputed parental genotype file records relationships by IID alone, "
                         f"so an IID appearing more than once in the fam file cannot be resolved. "
                         f"The fam file repeats {describe_offenders(clashing, noun='IID(s)')} that "
                         f"the file's pedigree refers to.")

    # The guard above means no ambiguous IID is ever looked up, so this only has to leave them
    # out of the way.  Dropping every copy rather than keeping the first is belt and braces: if
    # the guard were ever loosened, resolving to nothing is the safer failure.
    positions = positions[~ambiguous]

    def resolve(column):
        return positions.reindex(pedigree[:, column]).fillna(-1).astype(np.int64).to_numpy()

    child_pos = resolve(SNIPAR_PED_IID_COL)
    parent_pos = np.column_stack([resolve(SNIPAR_PED_FATHER_COL),
                                  resolve(SNIPAR_PED_MOTHER_COL)])

    # families is snipar's own list and is unique by construction, so it needs no such care
    imputed_rows = pd.Index(families).get_indexer(pedigree[:, SNIPAR_PED_FID_COL])

    return child_pos, parent_pos, imputed_rows

# -------------------------
def process_imputed_parents(imputed_parental_file: str, fam_file: Union[str, pd.DataFrame],
                            fam_df: pd.DataFrame, rel_df: pd.DataFrame) -> ParentalSources:
    """Identifies the individuals whose genotypes can be residualized against their parents,
    using snipar's imputed parental genotype file

    The file describes more families than can be used.  Three populations appear in it:

        the family has an imputed row                 -- use it, plus any genotyped parent
        no imputed row, but both parents genotyped    -- use the bed file for both
        no imputed row and no genotyped parent        -- NOT usable; ordinary within-group
                                                         residualization applies instead

    The third is easy to overlook, since the obvious reading of "absent from families" is that
    both parents were genotyped and there was nothing to impute.  That holds for some families but
    not all: snipar also omits families it could not impute for.

    The fam file's parental ID columns are not consulted at all -- the relationships come from
    snipar's own pedigree, which is validated against KING in the same way (see
    validate_parents_against_king).

    Args:
        imputed_parental_file: String path to the imputed parental genotype file
        fam_file: String path to the fam file, or the DataFrame itself, UNFILTERED
        fam_df: The already-filtered sample IDs, as returned by process_phenotypes()
        rel_df: Relatedness pairs, mapped to sample indices and integer degrees

    Returns:
        A ParentalSources describing the individuals with imputed or observed parents

    Raises:
        ValueError: If a relationship snipar states is contradicted by KING
    """

    logging.info(f"\tTaking parental relationships from {imputed_parental_file}.  The fam file's "
                 f"parental ID columns will not be consulted.")

    metadata = snipar.read_snipar_metadata(imputed_parental_file)
    pedigree = metadata[snipar.SNIPAR_PEDIGREE_KEY]
    families = metadata[snipar.SNIPAR_FAMILIES_KEY]

    full_fam_df, _ = get_df(fam_file, "fam_file", {"names" : FAM_COLS, "dtype" : str})
    full_fam_ids = as_id_strings(full_fam_df, [FAM_FID_COL, FAM_IID_COL])

    child_pos, parent_pos, imputed_rows = map_samples_to_imputed_families(
        pedigree=pedigree, families=families, full_fam_ids=full_fam_ids)

    # An individual is usable if there is anything to residualize them against: either the family
    # has an imputed row, or at least one parent was genotyped
    has_imputed = imputed_rows != -1
    has_observed_parent = (parent_pos != -1).any(axis=1)

    # ...and only if they are themselves in the analysis sample
    analysis_index = pd.Index(as_id_strings(fam_df, FAM_KEY)[FAM_IID_COL])
    in_analysis = analysis_index.get_indexer(pedigree[:, SNIPAR_PED_IID_COL]) != -1

    usable = in_analysis & (child_pos != -1) & (has_imputed | has_observed_parent)

    logging.info(f"\t\t{len(pedigree)} pedigree record(s) in the imputed parental genotype file")
    logging.info(f"\t\t{int((in_analysis & ~(has_imputed | has_observed_parent)).sum())} not "
                 f"usable (no imputed row and no genotyped parent)")
    logging.info(f"\t\t{int(usable.sum())} individual(s) to residualize against their parents")

    validate_parents_against_king(
        iids=pedigree[:, SNIPAR_PED_IID_COL], child_pos=child_pos,
        father_pos=parent_pos[:, 0], mother_pos=parent_pos[:, 1], usable=usable, rel_df=rel_df,
        source="the imputed parental genotype file's pedigree")

    if not np.any(usable):
        return empty_parental_sources()

    return ParentalSources(
        affected_samples=child_pos[usable],
        observed_parent_cols=parent_pos[usable],
        imputed_rows=imputed_rows[usable],
        imputed_file=imputed_parental_file,
        snipar_metadata=metadata)


# -------------------------
def calculate_R(rel_df: pd.DataFrame, N: int, affected_samples: np.ndarray = None
                ) -> sp.csr_array:
    """Builds the residualization matrix

    Each sample's row centers them within their relatedness group, so that multiplying by R
    subtracts that group's mean.

    Samples listed in affected_samples instead receive an IDENTITY row, passing their values
    through unchanged.  Those are the individuals whose genotypes have already been residualized
    against their parents, so centering them a second time would be wrong.  Note this differs from
    what a sample with no relatives receives, which is a ZERO row: an unrelated sample has no
    within-family contrast and so contributes nothing, whereas an affected sample contributes its
    own Mendelian residual.

    Affected samples are excluded only from their OWN rows.  They remain as columns in the rows of
    their relatives, who continue to center against them, and they still count when determining
    those relatives' minimum relatedness degree.

    Args:
        rel_df: Relatedness pairs, mapped to sample indices and integer degrees
        N: Number of samples
        affected_samples: Indices of samples to be given identity rows, or None for none

    Returns:
        The residualization matrix
    """

    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)
    rel_vals = rel_df[KING_REL_COL].to_numpy(np.int8)  # Values should fit in a single byte

    min_rel_per_index = np.full(N, max(INFTYPE_TO_DEG_MAP.values()) + 1, dtype=np.int8)
    np.minimum.at(min_rel_per_index, i1, rel_vals)
    np.minimum.at(min_rel_per_index, i2, rel_vals)

    mask_i1 = (rel_vals == min_rel_per_index[i1])  # Where rel is min for INDEX 1
    mask_i2 = (rel_vals == min_rel_per_index[i2])  # Where rel is min for INDEX 2

    # Affected samples contribute no entries to their own rows, which leaves those rows empty
    # through the normalization below, so that the final identity term yields an identity row
    is_affected = np.zeros(N, dtype=bool)
    if affected_samples is not None:
        is_affected[affected_samples] = True
        mask_i1 = mask_i1 & ~is_affected[i1]
        mask_i2 = mask_i2 & ~is_affected[i2]

    unaffected = np.flatnonzero(~is_affected)
    rows = np.concatenate([i1[mask_i1], i2[mask_i2], unaffected])
    cols = np.concatenate([i2[mask_i1], i1[mask_i2], unaffected])
    data = np.ones(rows.size, dtype=float)

    R = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()

    # Normalize only the rows that have entries.  Affected rows are empty here, and an unguarded
    # reciprocal of their zero row sum emits a divide-by-zero warning -- harmless as it happens,
    # since multiplying an empty sparse row by the resulting infinity touches nothing, but not
    # something to leave in the output of an ordinary run.
    row_sums = R.sum(axis=1).reshape((N, 1))
    scale = np.zeros_like(row_sums)
    np.divide(-1.0, row_sums, out=scale, where=(row_sums != 0.0))
    R *= scale
    R += sp.identity(N, dtype=float)

    return R

def process_relatedness(
    rel_file: Union[str, pd.DataFrame],
    fam_file: Union[str, pd.DataFrame],
    fam_df: pd.DataFrame,
    rel_degree: int,
    unresidualized_phenotypes: np.ndarray,
    imputed_parental_file: str = None
    ) -> Tuple[sp.csr_array, sp.csr_array, np.ndarray, "ParentalSources"]:
    """Performs processing of the KING-formatted relatedness / pedigree file

    Function to read in the relatedness / pedigree file, filter based on the already-processed
    fam file / sample IDs and the user-specified relatedness threshold, and generate the 
    residualization matrix and the the matrix used to calculate standard errors.

    Args:
        rel_file: String path to the pedigree file, or the Dataframe itself (already read in).
            (see https://www.kingrelatedness.com/manual.shtml)
        fam_df: Contains the already-filtered sample IDs (two columns, FID and IID)
        rel_degree: The maximum relatedness degree desired by the user, either an integer from
            0 to MAX_GRMA_RELATEDNESS or those same values in string form.  NOT the KING InfTypes.
        unresidualized_phenotypes: Phenotypes that have been processed up to the point of
            (but not including!) residualization based on relatedness groups.

    Returns:
        A tuple (R, gls_weights, resid_phenotypes, parental_sources) where:

            R is the residualization matrix (multiplication by this matrix residualizes within
                relatedness groups)
            gls_weights is the matrix the regression weights observations by: the pseudo-inverse
                of R @ Omega @ R.T, the covariance of the residualized phenotypes
            resid_phenotypes are the relatedness-residualized phenotypes
            parental_sources describes whose genotypes were residualized against their parents
    """


    # Make sure the relatedness threshold is of the correct type
    if isinstance(rel_degree, str):
        try:
            rel_degree = int(rel_degree)
        except ValueError as exc:
            raise ValueError(f"Could not interpret rel_degree ({rel_degree!r}) as an integer.  "
                             f"Expected a value from 0 to {MAX_GRMA_RELATEDNESS} (note that KING "
                             f"InfTypes such as \"FS\" are not valid for this parameter).") from exc
    else:
        try:
            rel_degree = operator.index(rel_degree)
        except TypeError as exc:
            raise TypeError(f"Expected str or int for parameter rel_degree, but "
                            f"received {type(rel_degree)}") from exc

    # Make sure the relatedness threshold is in range.
    if not 0 <= rel_degree <= MAX_GRMA_RELATEDNESS:
        raise ValueError(f"Received rel_degree = {rel_degree}, but it must be between 0 and "
                         f"{MAX_GRMA_RELATEDNESS} (inclusive).")


    # Read in relatedness file (should be in KING format)
    rel_df, _ = get_df(rel_file, "rel_file", {"header":0})
    rel_df = rel_df[NEEDED_KING_COLS]
    logging.info(f"\tRead in {len(rel_df)} rows from relatedness file")
    logging.info(f"\tRelatedness counts: {get_rel_report(rel_df)}\n")

    # Take note of the sample indices for each FID, IID pair to map to bed file and phenotypes
    N = len(fam_df)
    fam_mi = pd.MultiIndex.from_frame(fam_df[FAM_KEY])
    fam_index_vals = fam_df.index.to_numpy()

    pos1 = fam_mi.get_indexer(pd.MultiIndex.from_frame(rel_df[KING_KEY_1]))
    pos2 = fam_mi.get_indexer(pd.MultiIndex.from_frame(rel_df[KING_KEY_2]))
    mask = (pos1 != -1) & (pos2 != -1)

    rel_df = rel_df.loc[mask].drop(columns=KING_KEY_1+KING_KEY_2).reset_index(drop=True)
    rel_df[FAM_INDEX_1] = fam_index_vals[pos1[mask]]
    rel_df[FAM_INDEX_2] = fam_index_vals[pos2[mask]]
    logging.debug(f"\tRestricted the KING file to relevant samples")
    logging.debug(f"\tRelatedness degree counts: {get_rel_report(rel_df)}\n")


    # Make sure the InfType column is a number rather than a string
    rel_df[KING_REL_COL] = rel_df[KING_REL_COL].map(INFTYPE_TO_DEG_MAP)
    logging.debug(f"\tMapping KING relatedness to relatedness degree")
    logging.debug(f"\tRelatedness degree counts: {get_rel_report(rel_df)}\n")


    # Filter out relatedness that's too far away
    # TODO(jonbjala) This will need to change if we bring back higher degree thresholds
    rel_df = rel_df[rel_df[KING_REL_COL] <= SE_RELATEDNESS]
    logging.debug(f"\tAfter restricting to SE relatedness threshold, "
                  f"{len(rel_df)} rows remaining in relatedness file")
    logging.debug(f"\tRelatedness degree counts: {get_rel_report(rel_df)}\n")


    # Determine whose genotypes are residualized against their parents, and where those parents'
    # genotypes come from.  Needs the KING data parsed and mapped to sample indices, which is why
    # it sits here rather than as a separate step above.
    parental_sources = (
        process_imputed_parents(imputed_parental_file=imputed_parental_file, fam_file=fam_file,
                                fam_df=fam_df, rel_df=rel_df)
        if imputed_parental_file is not None
        else determine_fam_file_trios(fam_file=fam_file, fam_df=fam_df, rel_df=rel_df))

    # Assign samples to relatedness components once.  Both the shrinkage inside calculate_omega()
    # and the block-wise inversion in calculate_gls_weights() need this same partition, and both
    # must use omega's components rather than R's.
    labels = get_relatedness_components(rel_df=rel_df, N=N)

    # Create omega matrix
    omega = calculate_omega(rel_df=rel_df, N=N,
                            unresidualized_phenotypes=unresidualized_phenotypes, labels=labels)

    # Create R matrix
    rel_df = rel_df[rel_df[KING_REL_COL] <= rel_degree]
    logging.debug(f"\tAfter restricting to user-specified relatedness threshold, "
                  f"{len(rel_df)} rows remaining in relatedness file")
    logging.debug(f"\tRelatedness degree counts: {get_rel_report(rel_df)}")
    R = calculate_R(rel_df=rel_df, N=N, affected_samples=parental_sources.affected_samples)


    # Residualize the phenotypes using R
    resid_phenotypes = R @ unresidualized_phenotypes

    # Build the GLS weight matrix.  The covariance it inverts, R @ omega @ R.T, is a local
    # intermediate: everything downstream wants the weights rather than the covariance.
    gls_weights = calculate_gls_weights(R=R, omega=omega, labels=labels)
    del omega


    return R, gls_weights, resid_phenotypes, parental_sources


# -------------------------
class SnpAlignment(NamedTuple):
    """Maps the imputed parental genotype file's SNPs onto GRMA's retained ones

    Both arrays are aligned to the SNPs that survive filtering, in the same order as snp_filter.

    Attributes:
        hdf5_indices: Position within the imputed file of each retained SNP's data
        flip_mask: True where the imputed genotypes are coded to the other allele and must be
            subtracted from 2 before use
    """

    hdf5_indices: np.ndarray
    flip_mask: np.ndarray


# -------------------------
def classify_allele_orientation(grma_alleles: np.ndarray, snipar_alleles: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compares two sets of allele pairs and determines what each requires

    Args:
        grma_alleles: (n, 2) array of A1 and A2 from GRMA's bim file
        snipar_alleles: (n, 2) array of allele1 and allele2 from the imputed file

    Returns:
        A tuple (agrees, swapped, strand_differs) of (n,) boolean masks.  They are not mutually
        exclusive: a palindromic SNP's strand cannot be told from its letters, so it is both
        strand_differs and whichever of agrees or swapped applies.  A SNP matching none of the
        three has incompatible alleles and is a different variant altogether.
    """

    complemented = np.vectorize(ALLELE_COMPLEMENTS.get)(snipar_alleles)

    agrees = (grma_alleles == snipar_alleles).all(axis=1)
    swapped = (grma_alleles == snipar_alleles[:, ::-1]).all(axis=1)
    strand_differs = ((grma_alleles == complemented).all(axis=1) |
                      (grma_alleles == complemented[:, ::-1]).all(axis=1))

    return agrees, swapped, strand_differs


# -------------------------
def align_imputed_file_to_bim(bim_df: pd.DataFrame, metadata: dict
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matches the imputed parental genotype file's SNPs onto GRMA's, and resolves their
    orientation

    The bim file is the authority: every SNP GRMA analyses comes from it, in its order, and a
    variant present only in the imputed file is never looked at.  What this produces is, for each
    retained bim SNP, where to find its imputed data -- not a reordering of the analysis around
    the imputed file's variant set.

    Matching is by variant ID, with chromosome and position confirmed afterwards so that two files
    reusing an ID for different variants are caught rather than silently combined.

    Orientation matters because a disagreement means the imputed genotypes count the other allele,
    and subtracting an oppositely-coded parental mean produces plausible but wrong numbers.  Only
    heterozygotes are unaffected, so the error would not be obvious in aggregate.

    PALINDROMIC SNPs (A/T and C/G) cannot be resolved from the allele letters at all: on the
    opposite strand, (A,T) still reads as (A,T), with only the roles of the two alleles exchanged.
    Both hypotheses -- same strand, or opposite strand with the alleles swapped -- produce
    identical letters.  They are resolved here by looking at the UNAMBIGUOUS SNPs instead: if no
    non-palindromic SNP shows a strand difference, the two files share a strand, and palindromic
    SNPs can then be read literally.  If any does, the files disagree about strand and every
    palindromic SNP becomes genuinely ambiguous.

    TODO(jonbjala): rescue ambiguous palindromic SNPs by allele frequency.  Comparing standard_f
    against GRMA's own frequency distinguishes the two hypotheses wherever the frequency is far
    enough from 0.5 -- simulation puts that at ~93% of them at a threshold of 0.05, with no
    errors.  It needs a targeted pre-pass over the bed file, which this function has no access to,
    and a threshold calibrated from the non-palindromic SNPs rather than hardcoded.  Until then
    they are dropped, which is conservative but wasteful.

    Args:
        bim_df: GRMA's bim file, restricted to the SNPs that survive other filtering
        metadata: Output of read_snipar_metadata()

    Returns:
        A tuple (keep_mask, hdf5_indices, flip_mask), each aligned to bim_df's rows.  hdf5_indices
        and flip_mask are meaningful only where keep_mask is True.
    """

    bim_values = metadata[snipar.SNIPAR_BIM_VALUES_KEY]
    bim_columns = metadata[snipar.SNIPAR_BIM_COLUMNS_KEY]

    def snipar_column(names):
        return bim_values[:, snipar.get_snipar_bim_column_index(bim_columns, names)]

    hdf5_indices = pd.Index(snipar_column(snipar.SNIPAR_BIM_ID_NAMES)).get_indexer(
        bim_df[BIM_RSID_COL].astype(str))
    matched = hdf5_indices != -1

    # Confirm the matched variants really are the same ones.  Positions are stringified in the
    # imputed file, so both sides are compared as integers rather than as text.
    position_agrees = np.zeros(len(bim_df), dtype=bool)
    position_agrees[matched] = (
        (snipar_column(snipar.SNIPAR_BIM_CHR_NAMES)[hdf5_indices[matched]].astype(str) ==
         bim_df[BIM_CHR_COL].to_numpy().astype(str)[matched]) &
        (snipar_column(snipar.SNIPAR_BIM_BP_NAMES)[hdf5_indices[matched]].astype(np.int64) ==
         bim_df[BIM_BP_COL].to_numpy().astype(np.int64)[matched]))

    if np.any(matched & ~position_agrees):
        offenders = bim_df[BIM_RSID_COL].to_numpy()[matched & ~position_agrees]
        raise ValueError(f"{len(offenders)} variant ID(s) appear in both the bim file and the "
                         f"imputed parental genotype file but describe different positions, so "
                         f"they are not the same variant.  First few: {sorted(offenders)[:5]}")

    grma_alleles = bim_df[[BIM_A1_COL, BIM_A2_COL]].to_numpy().astype(str)
    snipar_alleles = np.column_stack([snipar_column(snipar.SNIPAR_BIM_A1_NAMES),
                                      snipar_column(snipar.SNIPAR_BIM_A2_NAMES)]
                                     )[np.where(matched, hdf5_indices, 0)]

    agrees, swapped, strand_differs = classify_allele_orientation(grma_alleles, snipar_alleles)
    palindromic = np.array([set(pair) in PALINDROMIC_ALLELE_PAIRS for pair in grma_alleles])

    # The unambiguous SNPs decide whether the two files share a strand
    strand_evidence = int((matched & ~palindromic & strand_differs).sum())
    strands_agree = strand_evidence == 0
    if not strands_agree:
        logging.warning(f"\t{strand_evidence} SNP(s) are recorded on opposite strands in the bim "
                        f"file and the imputed parental genotype file.  Those, and all "
                        f"palindromic SNPs, cannot be used.")

    usable_palindromic = palindromic & strands_agree
    keep_mask = matched & position_agrees & (agrees | swapped) & (~palindromic | usable_palindromic)
    flip_mask = keep_mask & swapped

    log_dropped_snps(bim_df=bim_df, matched=matched, keep_mask=keep_mask,
                     palindromic=palindromic, strand_differs=strand_differs,
                     strands_agree=strands_agree)

    return keep_mask, hdf5_indices, flip_mask


# -------------------------
def log_dropped_snps(bim_df: pd.DataFrame, matched: np.ndarray, keep_mask: np.ndarray,
                     palindromic: np.ndarray, strand_differs: np.ndarray, strands_agree: bool):
    """Reports how many SNPs could not be aligned, and why

    The counts matter more than the identities: a handful is ordinary data noise, while a large
    fraction means the two files were built from materially different variant sets and the user
    should know before trusting anything downstream.

    Args:
        bim_df: GRMA's bim file, restricted to the SNPs that survive other filtering
        matched: Which SNPs were found in the imputed file at all
        keep_mask: Which SNPs survive alignment
        palindromic: Which SNPs have palindromic alleles
        strand_differs: Which SNPs' alleles are complements
        strands_agree: Whether the two files were found to share a strand, which decides how a
            palindromic drop is described
    """

    dropped = ~keep_mask
    if not np.any(dropped):
        return

    # A palindromic SNP is dropped for one of two quite different reasons, and saying which
    # saves the reader working it out.  Where the two files share a strand it is kept, so a
    # palindromic drop here means they do not -- and that is a property of the whole file rather
    # than of this SNP, which the separate warning about opposite strands also reports.
    palindromic_reason = ("palindromic, and the two files disagree about strand" if not
                          strands_agree else "palindromic, so their orientation is ambiguous")

    reasons = {
        "not present in the imputed parental genotype file" : dropped & ~matched,
        palindromic_reason : dropped & matched & palindromic,
        "recorded on the opposite strand" :
            dropped & matched & ~palindromic & strand_differs,
        "incompatible alleles, so not the same variant" :
            dropped & matched & ~palindromic & ~strand_differs
    }

    total = int(dropped.sum())
    logging.info(f"\t{total} SNP(s) ({total / len(bim_df):.1%}) dropped -- they cannot be "
                 f"aligned to the imputed parental genotype file")
    for reason, mask in reasons.items():
        if np.any(mask):
            logging.info(f"\t\t{int(mask.sum())} {reason}")
            logging.debug(f"\t\t\t{sorted(bim_df[BIM_RSID_COL].to_numpy()[mask])}")


# -------------------------
def process_bim_file(bim_file: Union[str, pd.DataFrame], snp_list: Union[str, pd.DataFrame],
                     parental_sources: ParentalSources = None
                     ) -> Tuple[int, int, np.ndarray, SnpAlignment]:
    """Performs processing of the bim file

    Function to read in the bim file and possibly filter based on the optional user-provided snp
    list, and on aligning the imputed parental genotype file where one was supplied.

    Alignment belongs here rather than alongside the rest of parental processing because dropping
    an unalignable SNP is just another filtering criterion, and this function is what produces the
    SNP filter.  Doing it elsewhere would mean computing the filter and then revising it.

    Args:
        bim_file: String path to the bim file, or the Dataframe itself (already read in).
            (see https://www.cog-genomics.org/plink/1.9/formats#bim)
        snp_list: Optional file that contains the rsIDs (one per line) that should be included
            in the analysis
        parental_sources: Describes parental processing, or None.  Alignment happens only when it
            carries imputed parental genotypes.

    Returns:
        A tuple (M_orig, M, snp_filter, snp_alignment) where:

            M_orig is the initial number of SNPs in the bim file
            M is the final number of SNPs after filtering
            snp_filter is the list of numerical (values from 0 to M_orig-1) indices to use when
                filtering, say, the values in the bed file by SNP.  None when nothing was filtered
            snp_alignment maps the imputed parental genotype file onto the retained SNPs, or is
                None when no such file was supplied
    """

    aligning = (parental_sources is not None and parental_sources.snipar_metadata is not None)

    # Alignment needs the position and allele columns as well, so read the whole file for it
    read_params = ({"names" : list(BIM_COLS)} if aligning
                   else {"usecols" : [BIM_COLS.index(BIM_RSID_COL)], "names" : [BIM_RSID_COL]})
    bim_df, _ = get_df(bim_file, "bim_file", read_params)

    M_orig = len(bim_df)
    bim_name = bim_file if isinstance(bim_file, str) else "bim file"
    logging.info(f"\tThere are {M_orig} SNPs in {bim_name}")

    snp_filter = None

    # Read in snp list (if it exists)
    if snp_list is not None:
        snp_df, _ = get_df(snp_list, "snp_list", {"names" : [BIM_RSID_COL]})
        snplist_name = snp_list if isinstance(snp_list, str) else "SNP list"
        logging.info(f"\tThere are {len(snp_df)} SNPs in {snplist_name}")

        bim_index = pd.Index(bim_df[BIM_RSID_COL])
        positions = bim_index.get_indexer(snp_df[BIM_RSID_COL])
        missing_mask = (positions == -1)
        num_missing = int(missing_mask.sum())
        logging.debug(f"\tThere are {num_missing} SNPs listed in SNP list that are not in "
                      f"the bim file")
        logging.info(f"\tThere are {len(snp_df) - num_missing} SNPs remaining after filtering")

        snp_filter = np.sort(positions[~missing_mask])

    if not aligning:
        return M_orig, M_orig if snp_filter is None else len(snp_filter), snp_filter, None

    # Alignment needs an explicit filter to narrow, even when nothing has been filtered yet
    retained = np.arange(M_orig) if snp_filter is None else snp_filter

    keep_mask, hdf5_indices, flip_mask = align_imputed_file_to_bim(
        bim_df=bim_df.iloc[retained], metadata=parental_sources.snipar_metadata)

    snp_filter = retained[keep_mask]
    snp_alignment = SnpAlignment(hdf5_indices=hdf5_indices[keep_mask],
                                 flip_mask=flip_mask[keep_mask])

    logging.info(f"\tThere are {len(snp_filter)} SNPs remaining after aligning the imputed "
                 f"parental genotype file")

    return M_orig, len(snp_filter), snp_filter, snp_alignment
        

# -------------------------
def calculate_parental_means(G: np.ndarray, parental_sources: ParentalSources,
                             imputed_gts: np.ndarray = None,
                             num_obs_par_alleles: np.ndarray = None,
                             standard_f: np.ndarray = None) -> np.ndarray:
    """Calculates the value to subtract from each affected individual's genotypes

    What is subtracted depends on how many of an individual's parents were genotyped:

        both observed:    (g_father + g_mother) / 2
        one observed:     g_observed / 2, plus the transmitted allele from the unobserved parent
        neither observed: the corrected imputed value, which snipar has already averaged


    The asymmetry in the middle case is deliberate.  For the OBSERVED parent the genotype is
    known, so g/2 is the expected transmitted allele and what remains is that parent's Mendelian
    deviation -- the actual randomness.  For the UNOBSERVED parent only the transmitted allele is
    known, so subtracting it exactly makes that parent's contribution vanish rather than
    contribute a deviation.  Subtracting half the transmitted allele instead would leave a
    family-level quantity behind, which is precisely what this is meant to remove.

    G must be the FULL-WIDTH genotype array, before any sample filtering, since a parent may have
    been dropped from the analysis for a missing phenotype while their genotypes remain usable.
    It is only ever read: the means are returned for the caller to subtract, so that an affected
    individual who is themselves a parent is differenced against raw genotypes rather than
    against values that have already been adjusted.

    Every affected individual is expected to have at least one parental source -- a genotyped
    parent, an imputed row, or both.  GRMA code guarantees that by excluding
    anyone who has neither, and both of its branches report how many were excluded.  That is not
    re-checked here: this function runs once per block, so a check would repeat thousands of
    times what was settled once before the block loop began.  An individual with no source at all
    would simply have zero subtracted.

    Args:
        G: Genotypes for this block, (snps, N_orig), unfiltered by sample
        parental_sources: Describes the affected individuals and where their parents are
        imputed_gts: Imputed parental genotypes for this block, (families, snps), or None
        num_obs_par_alleles: Observed parental allele counts for this block, same shape, or None
        standard_f: Per-SNP allele frequencies for this block, or None

    Returns:
        The values to subtract, (snps, n_affected)
    """

    father_cols = parental_sources.observed_parent_cols[:, 0]
    mother_cols = parental_sources.observed_parent_cols[:, 1]

    # Sum of the observed parents' genotypes, counting only parents that were genotyped.  A NaN
    # in an observed parent propagates, which makes that SNP missing for the child.
    parental_sum = np.zeros((G.shape[0], len(father_cols)), dtype=float)
    for parent_cols in (father_cols, mother_cols):
        observed = parent_cols != -1
        if np.any(observed):
            parental_sum[:, observed] += G[:, parent_cols[observed]]

    # Call the imputed branch only when somebody actually has an imputed row.  A -1 throughout
    # means every affected family had both parents genotyped, and the imputed arrays are then not
    # merely unused but potentially absent, so this is a guard rather than an optimisation.
    if (parental_sources.imputed_rows is not None
            and np.any(parental_sources.imputed_rows != -1)):
        parental_sum += calculate_imputed_contribution(
            parental_sources=parental_sources, imputed_gts=imputed_gts,
            num_obs_par_alleles=num_obs_par_alleles, standard_f=standard_f)

    return parental_sum / 2.0


# -------------------------
def calculate_imputed_contribution(parental_sources: ParentalSources, imputed_gts: np.ndarray,
                                   num_obs_par_alleles: np.ndarray,
                                   standard_f: np.ndarray) -> np.ndarray:
    """Calculates the imputed parents' contribution to the parental sum

    snipar folds the population allele frequency into its imputed genotypes, standing in for the
    parental alleles it could not observe.  That term carries population structure, so it is
    removed here and only genuinely observed alleles are kept.

    The divisor depends on what the imputed value represents, which differs by parental status:
    where ONE parent was observed it is that single missing parent's genotype, and where NEITHER
    was it is the AVERAGE of the two.  An average has half the frequency contribution of a sum,
    so the correction must be halved for it.  Without that, the neither-observed case
    over-subtracts and produces negative observed-allele counts, which is impossible by
    construction.

    Args:
        parental_sources: Describes the affected individuals and their imputed rows
        imputed_gts: Imputed parental genotypes for this block, (families, snps)
        num_obs_par_alleles: Observed parental allele counts for this block, same shape
        standard_f: Per-SNP allele frequencies for this block

    Returns:
        The imputed contribution to the parental sum, (snps, n_affected)
    """

    rows = parental_sources.imputed_rows
    has_imputed = rows != -1

    contribution = np.zeros((imputed_gts.shape[1], len(rows)), dtype=float)
    if not np.any(has_imputed):
        return contribution

    # Transposed to (snps, affected) to match the genotype block's orientation
    imputed = imputed_gts[rows[has_imputed], :].T.astype(float)
    counts = num_obs_par_alleles[rows[has_imputed], :].T.astype(float)
    frequency = standard_f[:, None]

    # Where neither parent was observed, the imputed value is an average of two parents rather
    # than one parent's genotype, so its frequency contribution is halved
    neither_observed = (parental_sources.observed_parent_cols[has_imputed] == -1).all(axis=1)
    divisor = np.where(neither_observed[None, :], 2.0, 1.0)

    corrected = imputed - (frequency * (SNIPAR_TOTAL_PARENTAL_ALLELES - counts)) / divisor

    # An average of two parents stands for both of them, so it contributes twice to the sum
    corrected *= np.where(neither_observed[None, :], 2.0, 1.0)

    contribution[:, has_imputed] = corrected

    return contribution


# -------------------------
def get_block_snp_bounds(snp_filter: np.ndarray, M_start: int, num_to_read: int
                         ) -> Tuple[int, int]:
    """Finds which of the retained SNPs fall within a block

    Returns positions into snp_filter, not SNP indices.  Both the block loop and
    get_residualized_genotype_data() need this, and they must agree exactly: the first uses it to
    fetch the matching imputed data, the second to slice the genotypes, and a disagreement would
    pair each SNP with another SNP's parental values.

    Args:
        snp_filter: Indices of the SNPs retained for analysis, sorted
        M_start: First SNP of the block, as an index into the unfiltered bim
        num_to_read: Number of SNPs in the block

    Returns:
        A tuple (lower, upper) of positions into snp_filter.  The SNPs they select are
        snp_filter[lower:upper], so lower == upper means no retained SNP falls in this block.
    """

    lower = np.searchsorted(snp_filter, M_start, side="left")
    upper = np.searchsorted(snp_filter, M_start + num_to_read, side="left")

    return int(lower), int(upper)


# -------------------------
def read_imputed_block(imputed_hdf5, metadata: dict, block_hdf5_indices: np.ndarray,
                       block_flip_mask: np.ndarray, chunk: dict) -> dict:
    """Fetches one block's imputed parental data, maintaining a sliding chunk cache

    HDF5 cannot serve a partial chunk: asking for a block's worth of SNPs reads and discards the
    whole chunk containing them, and the next block does it again.  Chunk widths run from a few
    hundred to several thousand SNPs while a block is typically a hundred, so reading per block
    costs an order of magnitude more than necessary.  This keeps ONE chunk and slides it forward,
    copying out each part of the block as it goes, which is valid because blocks are processed in
    ascending SNP order and so the frontier never moves back.

    The flip is applied HERE, to the raw genotypes and frequencies, rather than to the corrected
    values further downstream.  A genotype always flips as 2 - g and a frequency as 1 - f, whereas
    flipping a corrected value needs a constant that varies with how many parental alleles were
    observed -- see calculate_imputed_contribution().

    Args:
        imputed_hdf5: Open handle to the imputed parental genotype file
        metadata: Output of read_snipar_metadata()
        block_hdf5_indices: Positions within the imputed file of this block's SNPs
        block_flip_mask: Which of this block's SNPs need their imputed genotypes flipped
        chunk: The cache, mutated in place.  Keys "cached_start" and "cached_stop" bound the SNP
            range currently held, as a half-open interval into the imputed file's own ordering,
            and "gts" and "counts" hold it.  cached_start is -1 before anything has been read

    Returns:
        A dict of the keyword arguments calculate_parental_means() needs for this block
    """

    chunk_width = snipar.get_snipar_chunk_width(imputed_hdf5)
    snp_count = snipar.get_snipar_snp_count(imputed_hdf5)

    imputed_gts = None
    num_obs_par_alleles = None

    # Fill from the cached chunk, advancing whenever this block runs past its end.  Only ONE
    # chunk is held at a time: the next is read, the part of the block it covers is copied out,
    # and it then replaces the current one.  Caching a range spanning several chunks instead
    # would be simpler but keeps them all resident, where this stays bounded at a single chunk
    # however snps_per_block and the file's chunk width happen to line up.
    #
    # Advancing is only ever forward, which is what makes discarding the previous chunk safe:
    # blocks are processed in ascending SNP order and hdf5_indices follows the bim's order, so
    # the frontier never moves back.
    filled = 0
    while filled < len(block_hdf5_indices):
        if chunk["cached_start"] == -1 or block_hdf5_indices[filled] >= chunk["cached_stop"]:
            start = (int(block_hdf5_indices[filled]) // chunk_width) * chunk_width
            stop = min(start + chunk_width, snp_count)

            # Rebinding drops the previous chunk, so the two coexist only between this read and
            # the assignment completing
            chunk["gts"], chunk["counts"] = snipar.read_snipar_snp_range(
                imputed_hdf5, start, stop)
            chunk["cached_start"], chunk["cached_stop"] = start, stop
            logging.debug(f"\t\tRead imputed parental SNPs [{start}, {stop})")

        if imputed_gts is None:
            imputed_gts = np.empty((chunk["gts"].shape[0], len(block_hdf5_indices)), dtype=float)
            num_obs_par_alleles = np.empty_like(imputed_gts)

        # Take every SNP of this block that the current chunk covers, then go round again
        covered = filled + int(np.searchsorted(block_hdf5_indices[filled:],
                                               chunk["cached_stop"], side="left"))
        offsets = block_hdf5_indices[filled:covered] - chunk["cached_start"]
        imputed_gts[:, filled:covered] = chunk["gts"][:, offsets]
        num_obs_par_alleles[:, filled:covered] = chunk["counts"][:, offsets]
        filled = covered

    standard_f = metadata[snipar.SNIPAR_STANDARD_F_KEY][block_hdf5_indices].astype(float)

    if np.any(block_flip_mask):
        imputed_gts[:, block_flip_mask] = (SNIPAR_ALLELES_PER_PARENT
                                           - imputed_gts[:, block_flip_mask])
        standard_f[block_flip_mask] = 1.0 - standard_f[block_flip_mask]

    return {"imputed_gts" : imputed_gts,
            "num_obs_par_alleles" : num_obs_par_alleles,
            "standard_f" : standard_f}

# -------------------------
def get_residualized_genotype_data(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int,
                                   snp_filter: np.ndarray, sample_filter: np.ndarray,
                                   R: sp.csr_array, M_start: int,
                                   num_snps: int, parental_sources: "ParentalSources" = None,
                                   imputed_block: dict = None) -> Tuple[np.ndarray, int]:
    """Performs processing of the bed file

    Function to read in the bed file, possibly filter based on either user-provided snp or sample
    lists as well as samples or SNPs dropped due to NaN / missing values, and then residualize
    based on relatedness grouping.  Using the M_start and num_snps parameter, a subset / chunk
    of the data can be read in (for all samples but only some of the SNPs).  Any values that are
    NaN are set to 0.0 after residualization.

    Args:
        bed_file: String path to the bed file, or the ndarray itself (already read in).
            (see https://www.cog-genomics.org/plink/1.9/formats#bed)
        M_orig: Number of SNPs provided in the bim/bed file (prior to filtering)
        N_orig: Number of samples provided in the fam/bed file (prior to filtering)
        snp_filter: Indices (valued 0 to M_orig-1) of SNPs to include in analysis.  If None,
            all SNPs are included
        sample_filter: Indices (valued 0 to N_orig-1) of samples to include in analysis.  If None,
            all samples are included
        R: Matrix used to residualize based on relatedness groups
        M_start: The index of the SNP to start at when reading in the data.  Should be between 0 and
            M_orig - 1, inclusive
        num_snps: The number of SNPs (prior to filtering) that should be read in.  If this would
            cause more SNPs to be read in than are available (given the value of M_start), as many
            as possible are read in

    Returns:
        A tuple (G, M) where:

            G is the residualized genotype array
            M is the number of SNPs in G / the number of SNPs actually read in
    """

    if M_start < 0 or M_start >= M_orig:
        raise ValueError(f"Invalid value for M_start ({M_start}), which should be between "
                         f"0 and {M_orig-1}")

    if num_snps <= 0:
        raise ValueError(f"Invalid value for num_snps ({num_snps}), which should be > 0")


    # Figure out if a full block should be read or if we're near the end and it's a partial block
    num_to_read = min(num_snps, M_orig - M_start)

    # If we're filtering SNPs, find out what portion of the filter array (if any) is relevant
    # and adjust for the offset.  Return early if the filter means skipping this whole block
    if snp_filter is not None:
        lower_index, upper_index = get_block_snp_bounds(snp_filter, M_start, num_to_read)

        logging.debug(f"\t\t{lower_index=} {upper_index=}")
        if lower_index == len(snp_filter) or lower_index == upper_index:
            return None, 0

        mod_snp_filter = snp_filter[lower_index:upper_index] - M_start


    # Read the bed file (or do nothing if we already have the genotype array)
    G = read_bed_file(bed_filename=bed_file, M=M_orig, N=N_orig, M_start=M_start,
                      num_snps=num_to_read) if isinstance(bed_file, str) else \
        bed_file[M_start:M_start+num_to_read]


    # Filter by SNP first.  The parental means below are computed from the FULL-WIDTH array,
    # before any sample filtering, because a parent may have been dropped from the analysis for a
    # missing phenotype while their genotypes remain perfectly usable.
    G = G[mod_snp_filter] if snp_filter is not None else G

    # Subtract the parental means BEFORE multiplying by R, not after.  For the affected
    # individuals themselves the order makes no difference, since their identity rows pass them
    # through either way.  It differs for everyone else in their relatedness groups: subtracting
    # first means a relative centers against the differenced value, which is the intended
    # behaviour.  Note also that every mean is computed before any subtraction is applied, so a
    # multi-generation pedigree differences against raw parental genotypes rather than against
    # values that have already been adjusted.
    if parental_sources is not None and len(parental_sources.affected_samples) > 0:
        # When bed_file is passed in as an array rather than a path, the slice above is a VIEW of
        # the caller's data, and subtracting in place would modify it.  Fancy indexing by SNP
        # already copies, so this only matters when there is no SNP filter.
        if snp_filter is None:
            G = G.copy()

        parental_means = calculate_parental_means(G=G, parental_sources=parental_sources,
                                                  **(imputed_block or {}))
        G[:, parental_sources.affected_samples] -= parental_means

    # Filter by sample now that the parental means have been taken
    G = G[:, sample_filter] if sample_filter is not None else G

    # Residualize the genotypes
    residualized_genotypes = G @ R.T

    # Replace NaN values with 0.0
    np.nan_to_num(residualized_genotypes, copy=False)

    return residualized_genotypes, residualized_genotypes.shape[0]



# -------------------------

def process_genotypes(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int, M: int, N: int,
                      resid_phenotypes: np.ndarray, snp_filter: np.ndarray,
                      sample_filter: np.ndarray,
                      R: sp.csr_array, gls_weights: sp.csr_array, snps_per_block: int,
                      parental_sources: ParentalSources = None,
                      snp_alignment: SnpAlignment = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs the main GRMA processing (calculating betas and SEs)

    Function that combines all the pre-processing and filtering, and then uses that to read in
    the appropriate chunks of the bed file, and then calculate the betas and SEs.

    Args:
        bed_file: String path to the bed file, or the ndarray itself (already read in).
            (see https://www.cog-genomics.org/plink/1.9/formats#bed)
        M_orig: Number of SNPs provided in the bim/bed file (prior to filtering)
        N_orig: Number of samples provided in the fam/bed file (prior to filtering)
        M: Number of SNPs used in the analysis (post-filtering)
        N: Number of samples used in the analysis (post-filtering)
        resid_phenotypes: Phenotypes that have been residualized by relatedness groups
        snp_filter: Indices (valued 0 to M_orig-1) of SNPs to include in analysis.  If None,
            all SNPs are included
        sample_filter: Indices (valued 0 to N_orig-1) of samples to include in analysis.  If None,
            all samples are included
        R: Matrix used to residualize based on relatedness groups
        gls_weights: Matrix the regression weights observations by (see calculate_gls_weights)
        snps_per_block: The number of SNPs to process at a time (prior to filtering, so some blocks
            will potentially include less than this).  If more are specified here than are
            available, then all SNPs will be processed together in one block.
        parental_sources: Describes whose genotypes are residualized against their parents, or
            None when no parental residualization applies
        snp_alignment: Maps the retained SNPs onto the imputed parental genotype file, or None
            when no such file was supplied

    Returns:
        A tuple (betas, ses, XtWX) where:

            betas are the per-SNP effect estimates
            ses are their standard errors
            XtWX is the per-SNP weighted sum of squares, reported as SUM_SQ_X
    """

    # Error checking
    if not isinstance(bed_file, (str, np.ndarray)):
        raise TypeError(f"Expected str or Numpy array for parameter bed_file, but "
                        f"received {type(bed_file)}")

    if snps_per_block <= 0:
        raise ValueError(f"Invalid value for snps_per_block ({snps_per_block}).  "
                         f"Must be > 0")
    
    # Debug logging
    logging.debug(f"\t{M_orig=} {M=} {N_orig=} {N=} {snps_per_block=}")
    logging.debug(f"\t{snp_filter=}")
    logging.debug(f"\t{sample_filter=}")

    # Create a place for results to be stored
    betas, ses, XtWX = np.zeros(M), np.zeros(M), np.zeros(M)

    # The phenotypes do not change between blocks, so weight them once rather than every pass
    weighted_phenotypes = gls_weights @ resid_phenotypes

    # Process data in blocks of SNPs
    num_blocks = int(np.ceil(M_orig / snps_per_block))
    logging.info(f"\tGenotype data will be processed in {num_blocks} "
                  f"block{"s" if num_blocks > 1 else ""}\n")
    current_position = 0

    # The imputed parental genotype file is held open for the whole loop rather than reopened per
    # block, and its data is cached a chunk at a time (see read_imputed_block).  nullcontext lets
    # a single with-statement cover both modes.
    reading_imputed = parental_sources is not None and parental_sources.imputed_file is not None
    file_context = (snipar.open_snipar_file(parental_sources.imputed_file) if reading_imputed
                    else contextlib.nullcontext())

    with file_context as imputed_hdf5:
        chunk = {"cached_start" : -1, "cached_stop" : -1, "gts" : None, "counts" : None}

        for block_num in range(num_blocks):
            logging.debug(f"\tProcessing block {block_num}")

            M_start = block_num * snps_per_block
            num_to_read = min(snps_per_block, M_orig - M_start)

            # Fetch this block's imputed parental data before the genotypes, so that the two are
            # sliced by the same bounds and each SNP is paired with its own parental values
            imputed_block = None
            if reading_imputed:
                lower, upper = get_block_snp_bounds(snp_filter, M_start, num_to_read)
                if lower == upper:
                    logging.debug(f"\tAll SNPs filtered out in this block!")
                    continue

                imputed_block = read_imputed_block(
                    imputed_hdf5=imputed_hdf5,
                    metadata=parental_sources.snipar_metadata,
                    block_hdf5_indices=snp_alignment.hdf5_indices[lower:upper],
                    block_flip_mask=snp_alignment.flip_mask[lower:upper],
                    chunk=chunk)

            block_genotypes, snps_read = get_residualized_genotype_data(
                bed_file=bed_file,
                M_orig=M_orig,
                N_orig=N_orig,
                snp_filter=snp_filter,
                sample_filter=sample_filter,
                R=R,
                M_start=M_start,
                num_snps=snps_per_block,
                parental_sources=parental_sources,
                imputed_block=imputed_block
            )
            if snps_read == 0:
                logging.debug(f"\tAll SNPs filtered out in this block!")
                continue

            # Calculate betas and SEs for this block of SNPs
            block_betas, block_ses, block_XtWX = calculate_gls_estimates(
                genotypes=block_genotypes,
                gls_weights=gls_weights,
                weighted_phenotypes=weighted_phenotypes
            )

            # Record the values and increment the current result position
            betas[current_position:current_position+snps_read] = block_betas
            ses[current_position:current_position+snps_read] = block_ses
            XtWX[current_position:current_position+snps_read] = block_XtWX
            current_position += snps_read

    logging.info("\tDone processing bed file blocks.\n")
    logging.info(f"\tNumber of SNPs used: {M}")
    logging.info(f"\tNumber of samples used: {N}")
    return betas, ses, XtWX

# -------------------------
def calculate_pvals(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    z_scores = betas / ses

    # Since P = 2 * normal_cdf(-|Z|), P = e ^ (log_normal_cdf(-|Z|) + ln 2)
    # This can be changed to base 10 as P = 10 ^ ((log_normal_cdf(-|Z|) + ln 2) / ln 10)
    log_10_p = RECIP_LN_10 * (norm.logcdf(-np.abs(z_scores)) + LN_2)

    # Break up the log based 10 of P values into the integer and fractional part
    # To handle the case of Z = 0 (and not result in "10e-1"), set initial values to (-1.0, 1.0)
    frac_part, int_part = np.full_like(z_scores, -1.0), np.full_like(z_scores, 1.0)
    np.modf(log_10_p, out=(frac_part, int_part), where=(z_scores != 0.0))

    # A SNP carrying no information gives XtWX = 0, hence a NaN beta and an infinite standard
    # error (see calculate_gls_estimates).  Those cannot be formatted as a number and must not be
    # cast to int, so they are reported as NaN and excluded from the arithmetic below.
    usable = np.isfinite(z_scores)
    frac_part = np.where(usable, frac_part, 0.0)
    int_part = np.where(usable, int_part, 1.0)

    # Construct strings for the P values
    # 1) Add one to the fractional part to ensure that the result mantissa is between 1 and 10
    # 2) Subtract one from the integer part to compensate and keep the overall value correct
    result = np.char.add(np.char.add(np.power(10.0, (frac_part + 1.0)).astype(str), 'e'),
                         (int_part - 1).astype(int).astype(str))

    return np.where(usable, result, str(np.nan))


def calculate_chisqs(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    return np.square(betas / ses)


def create_output(bim_file: Union[str, pd.DataFrame], snp_filter: np.ndarray,
                  betas: np.ndarray, ses: np.ndarray, sum_sq_x: np.ndarray) -> pd.DataFrame:

    # Read in bim file
    bim_df, _ = get_df(bim_file, "bim_file", {"usecols" : list(range(len(BIM_COLS))), 
                                              "names" : BIM_COLS})

    # Filter dataframe if need be
    bim_df = bim_df.iloc[snp_filter].reset_index(drop=True) if snp_filter is not None else bim_df

    # Calculate P values
    p_values = calculate_pvals(betas=betas, ses=ses)

    # Calculate chi squared values
    chisqs = calculate_chisqs(betas=betas, ses=ses)

    # Create new dataframe with output columns
    flip_betas = -betas  # Betas should be reported for A2, so flip sign
    extra_cols_df = pd.DataFrame(data={
            OUTPUT_BETA_COL : flip_betas, 
            OUTPUT_SE_COL : ses,
            OUTPUT_P_COL: p_values,
            OUTPUT_SUMSQX_COL: sum_sq_x
        }
    )

    # Combine dataframes
    results_df = pd.concat([bim_df, extra_cols_df], axis=1)


    # Log some results
    summary_df = pd.DataFrame(
        {
            "Beta" : [flip_betas.min(), flip_betas.mean(), flip_betas.std(), flip_betas.max()],
            "SE" : [ses.min(), ses.mean(), ses.std(), ses.max()],
            "CHI_SQ" : [chisqs.min(), chisqs.mean(), chisqs.std(), chisqs.max()]
        },
        index=["MIN", "MEAN", "STD_DEV", "MAX"]
    )
    logging.info(f"\tSummary table:\n{textwrap.indent(summary_df.to_string(), "\t")}")


    return results_df


# -------------------------
def grma(
    *,
    rel_file: Union[str, pd.DataFrame],
    bed_file: Union[str, np.ndarray],
    bim_file: Union[str, pd.DataFrame],
    fam_file: Union[str, pd.DataFrame],
    rel_degree: Union[str, int],
    pheno_file: Union[str, pd.DataFrame] = None,
    covar_file: Union[str, pd.DataFrame] = None,
    snps_per_block: int = DEFAULT_SNPS_PER_BLOCK,
    id_list: Union[str, pd.DataFrame] = None,
    snp_list: Union[str, pd.DataFrame] = None,
    imputed_parental_file: str = None
) -> pd.DataFrame:

    logging.debug(f"GRMA called with: {locals()}\n\n")

    grma_time = time.time()
    logging.info("Beginning grma() processing\n")

    # Process fam file and phenotype data
    fam_time = time.time()
    logging.info("Processing fam file / phenotype data")
    N_orig, N, sample_filter, fam_df, unresidualized_phenotypes = process_phenotypes(
        fam_file=fam_file,
        pheno_file=pheno_file,
        sample_id_file=id_list,
        covar_file=covar_file
    )
    if len(sample_filter) == 0:
        raise ValueError("Resulting sample filter is empty.")
    logging.info(f"Processing fam file / phenotype data took {time.time() - fam_time} seconds\n")



    # Process relatedness file (generate R and the SE matrix)
    rel_time = time.time()
    logging.info("Processing relatedness info")
    R, gls_weights, resid_phenotypes, parental_sources = process_relatedness(
        rel_file=rel_file,
        fam_file=fam_file,
        fam_df=fam_df,
        rel_degree=rel_degree,
        unresidualized_phenotypes=unresidualized_phenotypes,
        imputed_parental_file=imputed_parental_file
    )
    del unresidualized_phenotypes
    logging.info(f"Processing relatedness info took {time.time() - rel_time} seconds\n")


    # Process bim file (generate SNP filter)
    bim_time = time.time()
    logging.info("Processing bim file")
    M_orig, M, snp_filter, snp_alignment = process_bim_file(
        bim_file=bim_file, snp_list=snp_list, parental_sources=parental_sources)
    if snp_filter is not None and len(snp_filter) == 0:
        raise ValueError("Resulting SNP filter is empty.")
    logging.info(f"Processing bim file took {time.time() - bim_time} seconds\n")


    # Process bed file (generate betas and SEs)
    bed_time = time.time()
    logging.info("Processing bed file / running regressions / calculating standard errors")
    betas, ses, sum_sq_x = process_genotypes(
        bed_file=bed_file,
        M_orig=M_orig,
        N_orig=N_orig,
        M=M,
        N=N,
        resid_phenotypes=resid_phenotypes,
        snp_filter=snp_filter,
        sample_filter=sample_filter,
        R=R,
        gls_weights=gls_weights,
        snps_per_block=snps_per_block,
        parental_sources=parental_sources,
        snp_alignment=snp_alignment
    )
    logging.info(f"Processing bed file took {time.time() - bed_time} seconds\n")
 
    # Collate results and output them
    output_time = time.time()
    logging.info("Creating output (combining results with values from bim file)")
    results = create_output(bim_file=bim_file, snp_filter=snp_filter,
                            betas=betas, ses=ses, sum_sq_x=sum_sq_x)
    logging.info(f"Creating output took {time.time() - output_time} seconds\n")


    logging.info(f"Finished grma() processing.  It took {time.time() - grma_time} seconds.\n")
    return results


#################################
if __name__ == "__main__":
    print("This script is not meant to be called directly.")

    