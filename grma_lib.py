#!/usr/bin/env python3

"""
Library / core code for GRMA method
"""

# TODO(jonbjala) Functions should have header comments to describe parameters, returns, and pre-/post-conditions

import logging
import time
from typing import Tuple, Union #Any, Callable, Dict, List, Tuple, Union

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

# Copy-on-Write will become the default behaviour in Pandas 3.0 and is turned on to increase clarity about whether objects are views or copies (https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#)
pd.options.mode.copy_on_write = True

# Calculate constants used in determination of P values for MAMA
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
    

    # Read in fam file and take note of starting number of samples (before any filtering)
    fam_df, fam_mi = get_df(fam_file, "fam_file", {"names":FAM_COLS}, FAM_KEY)
    fam_df = fam_df.reset_index(names=FAM_INDEX).set_index(FAM_KEY, drop=False)
    N_orig = len(fam_df)
    total_intersection = fam_mi.copy()
    logging.info(f"There are {N_orig} samples in {fam_file}")



    # Read in id_list file (if specified) and find intersection of sample IDs
    sample_df, sample_mi = get_df(sample_id_file, "sample_id_file",
                                  {"usecols" : (0,1), "names" : FAM_KEY}, FAM_KEY)
    if sample_id_file is not None:
        total_intersection = total_intersection.intersection(sample_mi)
        logging.info(f"There are {len(sample_mi)} samples in {sample_id_file}")


    # Read in phenotype file (if specified) and find intersection of sample IDs
    pheno_df, pheno_mi = get_df(pheno_file, "pheno_file",
                                {"names":[FAM_FID_COL, FAM_IID_COL, FAM_PHENO_COL]}, FAM_KEY)
    if pheno_file is not None:
        total_intersection = total_intersection.intersection(pheno_mi)
        logging.info(f"There are {len(pheno_mi)} samples in {pheno_file}")


    # Read in covariates file (if specified) and find intersection of sample IDs
    covar_df, _ = get_df(covar_file, "covar_file")
    if covar_file is not None:
        n_covars = covar_df.shape[1] - 2 # Since there should be two initial columns, FID and IID
        if n_covars < 1:
            raise ValueError(f"Covariates file should contain at least 3 columns")
        if covar_df.iloc[:, 2:].isna().values.any():
            num_orig_covar_rows = len(covar_df)
            covar_df = covar_df[covar_df.iloc[:, 2:].notna().all(axis=1)]
            num_nan_rows = num_orig_covar_rows - len(covar_df)
            logging.warning(f"Covariates file {covar_file} contains NaN values.  "
                            f"Dropped {num_nan_rows} samples.")
        covar_df.columns = FAM_KEY + [f"COVAR_{i+1}" for i in range(n_covars)]
        covar_mi = pd.MultiIndex.from_frame(covar_df[FAM_KEY])

        total_intersection = total_intersection.intersection(covar_mi)
        logging.info(f"There are {len(covar_mi)} (non-NaN-containing) samples in {covar_file}")



    # Restrict down to the intersection of available samples
    if any([sample_id_file is not None, pheno_file is not None, covar_file is not None]):
        logging.info(f"\nThere are {len(total_intersection)} samples in the intersection")
        if len(total_intersection) < N_orig:
            logging.info(f"Restricting to these samples")
            fam_df = fam_df.reindex(total_intersection)
            fam_mi = total_intersection

    # Reassign the phenotype values to the ones from the phenotype file (override fam values)
    if pheno_file is not None:
        pheno_s = pheno_df.set_index(FAM_KEY)[FAM_PHENO_COL]
        fam_df[FAM_PHENO_COL] = pheno_s.reindex(fam_df.index).to_numpy()


    # Drop and NaN / missing values
    fam_df = fam_df.dropna(subset=[FAM_PHENO_COL])
    num_nan = len(total_intersection) - len(fam_df)
    if num_nan > 0:
        logging.info(f"Dropped {num_nan} samples with missing/NaN phenotype values.")
    fam_mi = fam_df.index

    # Residualize phenotypes on covariates
    if covar_file is not None:
        covar_df = covar_df.set_index(FAM_KEY, drop=False).reindex(fam_mi)
        covar_df["Intercept"] = 1.0

        covars = covar_df.iloc[:, 2:].to_numpy(dtype=float)
        orig_phenotypes = fam_df[FAM_PHENO_COL].to_numpy(dtype=float)
        x, _, _ = np.linalg.lstsq(a = covars, b = orig_phenotypes, rcond = None)
        fam_df[FAM_PHENO_COL] = orig_phenotypes - covars @ x


    # Make sure fam_df is in sorted order based on original index
    fam_df = fam_df.sort_values(FAM_INDEX)

    # Grab the phenotype values and the final value for N
    N = len(fam_df)
    phenotypes = fam_df[FAM_PHENO_COL].copy().to_numpy()

    return (N_orig, N, fam_df[FAM_INDEX].to_numpy(), fam_df[FAM_KEY].reset_index(drop=True),
            phenotypes)


# -------------------------
def get_lambda_min(B: np.ndarray, tol=DEFAULT_EIGSH_TOL) -> float:

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

    pheno_variance = np.var(unresidualized_phenotypes)
    alpha = min((epsilon - pheno_variance) / lambda_min, 1.0)

    omega = pheno_variance * sp.eye(N) + alpha * off_diag

    return omega


def process_relatedness(
    rel_file: Union[str, pd.DataFrame],
    fam_df: pd.DataFrame,
    rel_degree: Union[str, int],
    unresidualized_phenotypes: np.ndarray
    ) -> Tuple[sp.csr_array, sp.csr_array, np.ndarray]:

    # Make sure the relatedness threshold is of the correct type
    if isinstance(rel_degree, str):
        rel_degree = int(rel_degree)
    elif not isinstance(rel_degree, int):
        raise TypeError(f"Expected str or int for parameter rel_degree, but "
                        f"received {type(rel_degree)}")

    # Read in relatedness file (should be in KING format)
    rel_df, _ = get_df(rel_file, "rel_file", {"header":0})
    rel_df = rel_df[NEEDED_KING_COLS]

    # Take note of the sample indices for each FID, IID pair to map to bed file and phenotypes
    N = len(fam_df)
    fam_mi = pd.MultiIndex.from_frame(fam_df[FAM_KEY])
    fam_index_vals = fam_df.index.to_numpy()

    pos1 = fam_mi.get_indexer(pd.MultiIndex.from_frame(rel_df[KING_KEY_1]))
    pos2 = fam_mi.get_indexer(pd.MultiIndex.from_frame(rel_df[KING_KEY_2]))
    mask = (pos1 != -1) & (pos2 != -1)

    rel_df = rel_df.loc[mask].drop(columns=(KING_KEY_1+KING_KEY_2))
    rel_df[FAM_INDEX_1] = fam_index_vals[pos1[mask]]
    rel_df[FAM_INDEX_2] = fam_index_vals[pos2[mask]]

    # Make sure the InfType column is a number rather than a string
    rel_df[KING_REL_COL] = rel_df[KING_REL_COL].map(INFTYPE_TO_DEG_MAP)

    # Filter out relatedness that's too far away
    # TODO(jonbjala) This will need to change if we bring back higher degree thresholds
    rel_df = rel_df[rel_df[KING_REL_COL] <= SE_RELATEDNESS]

    # Create omega matrix
    omega = calculate_omega(rel_df=rel_df, N=N, unresidualized_phenotypes=unresidualized_phenotypes)

    # Create R matrix
    rel_df = rel_df[rel_df[KING_REL_COL] <= rel_degree]
    i1 = rel_df[FAM_INDEX_1].to_numpy(np.int64)
    i2 = rel_df[FAM_INDEX_2].to_numpy(np.int64)
    rel_vals = rel_df[KING_REL_COL].to_numpy(np.int8)   # values 1..5

    min_rel_per_index = np.full(N, max(INFTYPE_TO_DEG_MAP.values()) + 1, dtype=np.int8)
    np.minimum.at(min_rel_per_index, i1, rel_vals)
    np.minimum.at(min_rel_per_index, i2, rel_vals)

    mask_i1 = (rel_vals == min_rel_per_index[rel_df[FAM_INDEX_1]])  # Where rel is min for INDEX 1
    mask_i2 = (rel_vals == min_rel_per_index[rel_df[FAM_INDEX_2]])  # Where rel is min for INDEX 2

    rows = np.concatenate([i1[mask_i1], i2[mask_i2], np.arange(N, dtype=np.int64)])
    cols = np.concatenate([i2[mask_i1], i1[mask_i2], np.arange(N, dtype=np.int64)])
    data = np.ones(rows.size, dtype=float)

    R = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    R *= -np.reciprocal(R.sum(axis=1)).reshape((N,1))
    R +=  sp.identity(N, dtype=float)


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

    # Read in bim file
    bim_df, _ = get_df(bim_file, "bim_file", {"usecols" : [BIM_COLS.index(BIM_RSID_COL)],
                                              "names" : [BIM_RSID_COL]})
    M_orig = len(bim_df)

    # Read in snp list (if it exists)
    if snp_list is not None:
        snp_df, _ = get_df(snp_list, "snp_list", {"names" : BIM_RSID_COL})

        bim_index = pd.Index(bim_df[BIM_RSID_COL])
        positions = bim_index.get_indexer(snp_df[BIM_RSID_COL])
        missing_mask = (positions == -1)
        num_missing = int(missing_mask.sum())

        return M_orig, len(snp_df) - num_missing, np.sort(positions[~missing_mask])

    return M_orig, M_orig, None
        


# -------------------------
def get_residualized_genotype_data(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int,
                                   snp_filter: np.ndarray, sample_filter: np.ndarray,
                                   R: sp.csr_array, M_start: int,
                                   num_snps: int) -> Tuple[np.ndarray, int]:

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
        upper_index = np.searchsorted(snp_filter, M_start + num_to_read, side="right")

        if lower_index == -1 or lower_index == len(snp_filter) or lower_index == upper_index:
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
    # TODO(jonbjala) Verify this approach is what we want
    np.nan_to_num(residualized_genotypes, copy=False)

    return residualized_genotypes, residualized_genotypes.shape[0]


def calculate_betas(genotypes: np.ndarray, phenotypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    XtX = np.einsum('ij,ij->i', genotypes, genotypes)
    XtX[XtX == 0.0] = np.finfo(XtX.dtype).eps  # Replace 0.0 with something incredibly small

    betas = np.einsum('ij,j->i', genotypes, phenotypes) / XtX

    return betas, XtX


def calculate_ses(genotypes: np.ndarray, se_matrix: np.ndarray, XtX: np.ndarray) -> np.ndarray:
    product1 = se_matrix @ genotypes.T
    product2 = np.einsum('ij,ij->j', product1, genotypes.T)

    ses = np.sqrt(product2) / XtX
    ses[ses == 0.0] = np.finfo(ses.dtype).eps  # Replace 0.0 with something incredibly small

    return ses


def process_genotypes(bed_file: Union[str, np.ndarray], M_orig: int, N_orig: int, M: int, N: int,
                      phenotypes: np.ndarray, snp_filter: np.ndarray, sample_filter: np.ndarray,
                      R: sp.csr_array, se_matrix: sp.csr_array, snps_per_block: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if not isinstance(bed_file, str) and not isinstance(bed_file, np.ndarray):
        raise TypeError(f"Expected str or Numpy array for parameter {bed_file}, but "
                        f"received {type(bed_file)}")

    # Create a place for results to be stored
    betas, ses, XtX = np.zeros(M), np.zeros(M), np.zeros(M)

    # Process data in blocks of SNPs
    num_blocks = int(np.ceil(M_orig / snps_per_block))
    logging.debug(f"Genotype data will be processed in {num_blocks} blocks")
    current_position = 0
    for block_num in range(num_blocks):
        logging.debug(f"Processing block {block_num}")

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
            logging.debug(f"All SNPs filtered out in this block")
            continue

        # Calculate betas, squared sum of genotypes, and SEs for this block of SNPs
        block_betas, block_XtX = calculate_betas(genotypes=block_genotypes, phenotypes=phenotypes)
        block_ses = calculate_ses(genotypes=block_genotypes, se_matrix=se_matrix, XtX=block_XtX)

        # Record the values and increment the current result position
        betas[current_position:current_position+snps_read] = block_betas
        ses[current_position:current_position+snps_read] = block_ses
        XtX[current_position:current_position+snps_read] = block_XtX
        current_position += snps_read

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


def create_output(bim_file: Union[str, np.ndarray], snp_filter: np.ndarray,
                  betas: np.ndarray, ses: np.ndarray, sum_sq_x: np.ndarray) -> pd.DataFrame:

    # Read in bim file
    bim_df, _ = get_df(bim_file, "bim_file", {"usecols" : list(range(len(BIM_COLS))), 
                                              "names" : BIM_COLS})

    # Filter dataframe if need be
    bim_df = bim_df.iloc[snp_filter] if snp_filter else bim_df

    # Calculate P values
    p_values = calculate_pvals(betas=betas, ses=ses)

    # Create new dataframe with output columns
    extra_cols_df = pd.DataFrame(data={
            OUTPUT_BETA_COL : -betas,  # For some reason we want to swap the alleles
            OUTPUT_SE_COL : ses,
            OUTPUT_P_COL: p_values,
            OUTPUT_SUMSQX_COL: sum_sq_x
        }
    )

    # Combine dataframes
    results_df = pd.concat([bim_df, extra_cols_df], axis=1)


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

    logging.debug(f"GRMA called with: {locals()}")

    grma_time = time.time()
    logging.info("Beginning grma() processing")

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
    logging.info(f"Processing fam file / phenotype data took {time.time() - fam_time} seconds")



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
    logging.info(f"Processing relatedness info took {time.time() - rel_time} seconds")


    # Process bim file (generate SNP filter)
    bim_time = time.time()
    logging.info("Processing bim file")
    M_orig, M, snp_filter = process_bim_file(bim_file=bim_file, snp_list=snp_list)
    if snp_filter is not None and len(snp_filter) == 0:
        raise ValueError("Resulting SNP filter is empty.")
    logging.info(f"Processing bim file took {time.time() - bim_time} seconds")


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
    logging.info(f"Processing bed file took {time.time() - bim_time} seconds")
 

    # Collate results and output them
    output_time = time.time()
    logging.info("Creating output (combining results with values from bim file)")
    results = create_output(bim_file=bim_file, snp_filter=snp_filter,
                            betas=betas, ses=ses, sum_sq_x=sum_sq_x)
    logging.info(f"Creating output took {time.time() - output_time} seconds")


    return results


#################################
if __name__ == "__main__":
    print("This script is not meant to be called directly.")

    