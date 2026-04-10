#!/usr/bin/env python3

"""
Library / core code for GRMA method
"""

import logging
import textwrap
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm

from bedbimfam import (
    BED_SUFFIX,
    BIM_COLS,
    BIM_RSID_COL,
    BIM_SUFFIX,
    FAM_COLS,
    FAM_FID_COL,
    FAM_IID_COL,
    FAM_IIDF_COL,
    FAM_IIDM_COL,
    FAM_SEX_COL,
    FAM_PHENO_COL,
    FAM_SUFFIX,
    get_num_snps_from_bim_file,
    get_sample_size_from_fam_file,
    read_bed_file,
)

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


FAM_INDEX = "FAM_INDEX"

FAM_KEY = [FAM_FID_COL, FAM_IID_COL]
KING_KEY_1 = [KING_FID1_COL, KING_IID1_COL]
KING_KEY_2 = [KING_FID2_COL, KING_IID2_COL]

FAM_INDEX_1 = "FAM_INDEX_1"
FAM_INDEX_2 = "FAM_INDEX_2"



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


# Kinship thresholds
MIN_KINSHIP_THRESH = 0.0
MAX_KINSHIP_THRESH = 0.0625


# Default number of SNPs to process at a time
DEFAULT_SNPS_PER_BLOCK = 100

# Default values in omega shrinking procedure
DEFAULT_OMEGA_EPSILON = 1e-10
DEFAULT_EIGSH_TOL = 1e-8


# Output columns
OUTPUT_BETA_COL = 'BETA'
OUTPUT_SE_COL = 'SE'
OUTPUT_P_COL = 'P'
OUTPUT_SUMSQX_COL = 'SUM_SQ_X'

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

    # Residualize phenotypes on covariates
    if covar_file is not None:
        logging.info(f"\tRunning regression to residualize phenotypes on covariates")
        covar_df = covar_df.set_index(FAM_KEY, drop=False).reindex(fam_mi)
        covar_df["Intercept"] = 1.0

        covars = covar_df.iloc[:, 2:].to_numpy(dtype=float)
        orig_phenotypes = fam_df[FAM_PHENO_COL].to_numpy(dtype=float)
        x, _, _, _ = np.linalg.lstsq(a = covars, b = orig_phenotypes, rcond = None)
        fam_df[FAM_PHENO_COL] = orig_phenotypes - covars @ x


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

    for ncv_factor_exp in range(5):
        ncv = min(N, 20 * 2 ** (ncv_factor_exp + 1))

        try:
            eigenvalues, _ = spla.eigsh(B, k=1, which='SA', tol=tol, ncv=ncv)
            return eigenvalues[0]
        except spla.ArpackNoConvergence:
            continue

    raise RuntimeError("Eigsh failed to converge.  Not possible to compute omega.")


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


def calculate_omega(rel_df: pd.DataFrame, N: int, unresidualized_phenotypes: np.ndarray,
                    epsilon=DEFAULT_OMEGA_EPSILON) -> sp.csr_array:

    rel_to_cov = get_rel_covariances(rel_df=rel_df,
                                     unresidualized_phenotypes=unresidualized_phenotypes)


    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)
    rel = rel_df[KING_REL_COL]

    rows = np.concatenate([i1, i2])
    cols = np.concatenate([i2, i1])
    data = np.tile(rel.map(rel_to_cov).to_numpy(), 2)

    off_diag = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()

    lambda_min = get_lambda_min(off_diag)
    logging.debug(f"\t{lambda_min=}")

    pheno_variance = np.var(unresidualized_phenotypes)
    logging.debug(f"\t{pheno_variance=}")
    alpha = min((epsilon - pheno_variance) / lambda_min, 1.0) if lambda_min < 0.0 else 1.0
    logging.debug(f"\t{alpha=}\n")

    omega = pheno_variance * sp.eye(N) + alpha * off_diag

    return omega


def calculate_R(rel_df: pd.DataFrame, N: int) -> sp.csr_array:
    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)
    rel_vals = rel_df[KING_REL_COL].to_numpy(np.int8)  # Values should fit in a single byte

    min_rel_per_index = np.full(N, max(INFTYPE_TO_DEG_MAP.values()) + 1, dtype=np.int8)
    np.minimum.at(min_rel_per_index, i1, rel_vals)
    np.minimum.at(min_rel_per_index, i2, rel_vals)

    mask_i1 = (rel_vals == min_rel_per_index[i1])  # Where rel is min for INDEX 1
    mask_i2 = (rel_vals == min_rel_per_index[i2])  # Where rel is min for INDEX 2

    rows = np.concatenate([i1[mask_i1], i2[mask_i2], np.arange(N, dtype=np.int64)])
    cols = np.concatenate([i2[mask_i1], i1[mask_i2], np.arange(N, dtype=np.int64)])
    data = np.ones(rows.size, dtype=float)

    R = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    R *= -np.reciprocal(R.sum(axis=1)).reshape((N,1))
    R += sp.identity(N, dtype=float)

    return R

def process_relatedness(
    rel_file: Union[str, pd.DataFrame],
    fam_df: pd.DataFrame,
    rel_degree: Union[str, int],
    unresidualized_phenotypes: np.ndarray
    ) -> Tuple[sp.csr_array, sp.csr_array, np.ndarray]:
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
        A tuple (R, se_matrix, phenotypes) where:

            R is the residualization matrix (multiplication by this matrix residualizes within 
                relatedness groups)
            se_matrix is the matrix used to calculate standard errors (see GRMA paper for details,
                but this is essentially R @ Omega @ R.T)
            phenotypes are the relatedness-residualized phenotypes
    """


    # Make sure the relatedness threshold is of the correct type
    if isinstance(rel_degree, str):
        rel_degree = int(rel_degree)
    elif not isinstance(rel_degree, int):
        raise TypeError(f"Expected str or int for parameter rel_degree, but "
                        f"received {type(rel_degree)}")

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


    # Create omega matrix
    omega = calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=unresidualized_phenotypes)

    # Create R matrix
    rel_df = rel_df[rel_df[KING_REL_COL] <= rel_degree]
    logging.debug(f"\tAfter restricting to user-specified relatedness threshold, "
                  f"{len(rel_df)} rows remaining in relatedness file")
    logging.debug(f"\tRelatedness degree counts: {get_rel_report(rel_df)}")
    R = calculate_R(rel_df=rel_df, N=N)


    # Residualize the phenotypes using R
    phenotypes = R @ unresidualized_phenotypes

    # Create the SE matrix (omega multiplied on either side by R/R_t, used in SE calculations)
    se_matrix = R @ omega
    del omega
    se_matrix = se_matrix @ R.T


    return R, se_matrix, phenotypes


# -------------------------
# TODO(jonbjala) This might need to be more complicated, return a SNP breakdown plan (for LD scores)
def process_bim_file(bim_file: Union[str, pd.DataFrame], snp_list: Union[str, pd.DataFrame]
    ) -> Tuple[int, int, np.ndarray]:
    """Performs processing of the bim file

    Function to read in the bim file and possibly filter based on the optional
    user-provided snp list.

    Args:
        bim_file: String path to the bim file, or the Dataframe itself (already read in).
            (see https://www.cog-genomics.org/plink/1.9/formats#bim)
        snp_list: Optional file that contains the rsIDs (one per line) that should be included
            in the analysis

    Returns:
        A tuple (M_orig, M, snp_filter) where:

            M_orig is the initial number of SNPs in the bim file
            M is the final number of SNPs after filtering based on snp_list, if provided
                (should be equal to M_orig if no snp_list is provided)
            snp_filter is the list of numerical (values from 0 to M_orig-1) indices to use when
                filtering, say, the values in the bed file by SNP
    """
    # Read in bim file
    bim_df, _ = get_df(bim_file, "bim_file", {"usecols" : [BIM_COLS.index(BIM_RSID_COL)],
                                              "names" : [BIM_RSID_COL]})
    M_orig = len(bim_df)
    bim_name = bim_file if isinstance(bim_file, str) else "bim file"
    logging.info(f"\tThere are {M_orig} SNPs in {bim_name}")

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

        return M_orig, len(snp_df) - num_missing, np.sort(positions[~missing_mask])

    return M_orig, M_orig, None
        


# -------------------------
def get_residualized_genotype_data(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int,
                                   snp_filter: np.ndarray, sample_filter: np.ndarray,
                                   R: sp.csr_array, M_start: int,
                                   num_snps: int) -> Tuple[np.ndarray, int]:
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
        lower_index = np.searchsorted(snp_filter, M_start, side="left")
        upper_index = np.searchsorted(snp_filter, M_start + num_to_read, side="left")

        logging.debug(f"\t\t{lower_index=} {upper_index=}")
        if lower_index == len(snp_filter) or lower_index == upper_index:
            return None, 0

        mod_snp_filter = snp_filter[lower_index:upper_index] - M_start


    # Read the bed file (or do nothing if we already have the genotype array)
    G = read_bed_file(bed_filename=bed_file, M=M_orig, N=N_orig, M_start=M_start,
                      num_snps=num_to_read) if isinstance(bed_file, str) else \
        bed_file[M_start:M_start+num_to_read]


    # Filter the genotype array if filters exist
    G = G[:, sample_filter] if sample_filter is not None else G
    G = G[mod_snp_filter] if snp_filter is not None else G

    # Residualize the genotypes
    residualized_genotypes = G @ R.T

    # Replace NaN values with 0.0
    np.nan_to_num(residualized_genotypes, copy=False)

    return residualized_genotypes, residualized_genotypes.shape[0]


def calculate_betas(genotypes: np.ndarray, phenotypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    XtX = np.einsum('ij,ij->i', genotypes, genotypes)
    XtX[XtX == 0.0] = np.finfo(XtX.dtype).eps  # Replace 0.0 with something incredibly small

    betas = np.einsum('ij,j->i', genotypes, phenotypes) / XtX

    return betas, XtX


def calculate_ses(genotypes: np.ndarray, se_matrix: sp.csr_array, XtX: np.ndarray) -> np.ndarray:
    product1 = se_matrix @ genotypes.T
    product2 = np.einsum('ij,ij->j', product1, genotypes.T)

    ses = np.sqrt(product2) / XtX
    ses[ses == 0.0] = np.finfo(ses.dtype).eps  # Replace 0.0 with something incredibly small

    return ses


def process_genotypes(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int, M: int, N: int,
                      phenotypes: np.ndarray, snp_filter: np.ndarray, sample_filter: np.ndarray,
                      R: sp.csr_array, se_matrix: sp.csr_array, snps_per_block: int
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
        phenotypes: Phenotypes that have been residualized by relatedness groups
        snp_filter: Indices (valued 0 to M_orig-1) of SNPs to include in analysis.  If None,
            all SNPs are included
        sample_filter: Indices (valued 0 to N_orig-1) of samples to include in analysis.  If None,
            all samples are included
        R: Matrix used to residualize based on relatedness groups
        se_matrix: Matrix used to calculate standard errors (see GRMA paper for details,
            but this is essentially R @ Omega @ R.T)
        snps_per_block: The number of SNPs to process at a time (prior to filtering, so some blocks
            will potentially include less than this).  If more are specified here than are
            available, then all SNPs will be processed together in one block.

    Returns:
        A tuple (G, M) where:

            G is the residualized genotype array
            M is the number of SNPs in G
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
    betas, ses, XtX = np.zeros(M), np.zeros(M), np.zeros(M)

    # Process data in blocks of SNPs
    num_blocks = int(np.ceil(M_orig / snps_per_block))
    logging.info(f"\tGenotype data will be processed in {num_blocks} "
                  f"block{"s" if num_blocks > 1 else ""}\n")
    current_position = 0
    for block_num in range(num_blocks):
        logging.debug(f"\tProcessing block {block_num}")

        block_genotypes, snps_read = get_residualized_genotype_data(
            bed_file=bed_file,
            M_orig=M_orig,
            N_orig=N_orig,
            snp_filter=snp_filter,
            sample_filter=sample_filter,
            R=R,
            M_start=block_num*snps_per_block,
            num_snps=snps_per_block
        )
        if snps_read == 0:
            logging.debug(f"\tAll SNPs filtered out in this block!")
            continue

        # Calculate betas, squared sum of genotypes, and SEs for this block of SNPs
        block_betas, block_XtX = calculate_betas(genotypes=block_genotypes, phenotypes=phenotypes)
        block_ses = calculate_ses(genotypes=block_genotypes, se_matrix=se_matrix, XtX=block_XtX)

        # Record the values and increment the current result position
        betas[current_position:current_position+snps_read] = block_betas
        ses[current_position:current_position+snps_read] = block_ses
        XtX[current_position:current_position+snps_read] = block_XtX
        current_position += snps_read

    logging.info("\tDone processing bed file blocks.\n")
    logging.info(f"\tNumber of SNPs used: {M}")
    logging.info(f"\tNumber of samples used: {N}")
    return betas, ses, XtX

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

    # Construct strings for the P values
    # 1) Add one to the fractional part to ensure that the result mantissa is between 1 and 10
    # 2) Subtract one from the integer part to compensate and keep the overall value correct
    result = np.char.add(np.char.add(np.power(10.0, (frac_part + 1.0)).astype(str), 'e'),
                         (int_part - 1).astype(int).astype(str))

    return result


def calculate_chisqs(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    return np.square(betas / ses)


def create_output(bim_file: Union[str, np.ndarray], snp_filter: np.ndarray,
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
    snp_list: Union[str, pd.DataFrame] = None
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
    R, se_matrix, phenotypes = process_relatedness(
        rel_file=rel_file,
        fam_df=fam_df,
        rel_degree=rel_degree,
        unresidualized_phenotypes=unresidualized_phenotypes
    )
    del unresidualized_phenotypes
    logging.info(f"Processing relatedness info took {time.time() - rel_time} seconds\n")


    # Process bim file (generate SNP filter)
    bim_time = time.time()
    logging.info("Processing bim file")
    M_orig, M, snp_filter = process_bim_file(bim_file=bim_file, snp_list=snp_list)
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
        phenotypes=phenotypes,
        snp_filter=snp_filter,
        sample_filter=sample_filter,
        R=R,
        se_matrix=se_matrix,
        snps_per_block=snps_per_block
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

    