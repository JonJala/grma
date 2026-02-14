#!/usr/bin/env python3

"""
Library / core code for GRMA method
"""

# TODO(jonbjala) Functions should have header comments to describe parameters, returns, and pre-/post-conditions

import itertools as it
import logging
import time
from typing import Any, Callable, Dict, List, Tuple, Union

import bitarray as ba
import bitarray.util as baut
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.sparse as sp
from ast import literal_eval
import pickle

# Copy-on-Write will become the default behaviour in Pandas 3.0 and is turned on to increase clarity about whether objects are views or copies (https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#)
pd.options.mode.copy_on_write = True

from bedbimfam import (
    BED_SUFFIX,
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


# Object to hold relatedness.  Currently implemented as a list of lists (of ints), where
# the ith element of the outer list is the list of indices of all people to include in
# the residualization step (the list can simply be passed to numpy for indexing purposes)
THRESHOLDED_REL_TYPE = List[List[int]]


# Columns used in the steps to construct relatedness object
FID_COL = "FID"
IID_COL = "IID"
INDEX_COL = "INDEX"
INDEX1_COL = "INDEX1"
INDEX2_COL = "INDEX2"


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

# Values seen in KING file as InfTypes
INF_DUP_MZTWIN = "Dup/MZTwin"
INF_FULLSIB = "FS"
INF_PARENT_OFFSPRING = "PO"
INF_2ND = "2nd"
INF_3RD = "3rd"
INF_4TH = "4th"
INF_UNRELATED = "UN"

MAX_RELATEDNESS = 4  # Maximum degree of relatedness from king output
SE_RELATEDNESS = 3  # Relatedness to include in SE calculations

# Column name used to label phenotype when pulled in from separate phenotype file
PHENOFILE_PHENO_COL = "Phenofile_Phenotype"

# Kinship thresholds
MIN_KINSHIP_THRESH = 0.0
MAX_KINSHIP_THRESH = 0.0625

# Map of rel degree to numeric value to subset king output
REL_TO_DEG_MAP = {"FS": 0, "1": 1, "2": 2, "3": 3}

# List of inputs to accept as flags to specify degree of relation allowed
REL_DEG_INPUTS = list(REL_TO_DEG_MAP.keys())

# Map of possible InfTypes in a King output file. 
INF_TO_DEG_MAP = {
    INF_DUP_MZTWIN: 0,
    INF_FULLSIB: 0,
    INF_PARENT_OFFSPRING: 1,
    INF_2ND: 2,
    INF_3RD: 3,
    INF_4TH: 4,
    INF_UNRELATED: 5
    }

# Default number of SNPs to process at a time
DEFAULT_SNPS_PER_BLOCK = 100


# Output columns
OUTPUT_BETA_COL = 'BETA'
OUTPUT_SE_COL = 'SE'
OUTPUT_P_COL = 'P'
OUTPUT_SUMSQX_COL = 'SUM_SQ_X'

# -------------------------
def get_sample_indices_to_keep(id_list: str, fam_filename: str) -> pd.DataFrame:
    # Load id_list and get the indices in the fam_file of the individuals to keep
    ids_to_keep = pd.read_csv(id_list, sep=r"\s+", usecols=(0, 1), names=[FID_COL, IID_COL],
                              index_col=False, header=None)
    fam_df = pd.read_csv(fam_filename, sep=r"\s+", usecols=(0, 1), names=[FID_COL, IID_COL],
                         index_col=False)
    fam_df[INDEX_COL] = range(len(fam_df))
    
    merged_df = pd.merge(ids_to_keep, fam_df, on=[FID_COL, IID_COL], how="inner", copy=False)
    
    return merged_df[INDEX_COL].tolist()

# -------------------------
def get_snp_indices_to_keep(snp_list: str, bim_filename: str) -> pd.DataFrame:
    # Load snp_list and get the indices in the bim_file of the SNPs to keep
    snps_to_keep = pd.read_csv(snp_list, sep=r"\s+", usecols=[0], names=["rsid"],
                               index_col=False, header=None)
    bim_df = pd.read_csv(bim_filename, sep=r"\s+", usecols=[1], names=["rsid"], index_col=False)

    indices = bim_df.index[bim_df["rsid"].isin(snps_to_keep["rsid"])].tolist()
    
    return indices
    
# -------------------------
def get_phenotypes_from_file(pheno_filename: str, fam_filename: str, sample_indices_to_keep: str
    ) -> np.ndarray:
    # TODO(jonbjala)  Handle missing phenotype values?  Have support for case/control?  https://www.cog-genomics.org/plink/1.9/formats#fam

    fam_df = pd.read_csv(fam_filename, sep=r"\s+", names=FAM_COLS, index_col=False)
    pheno_col = FAM_PHENO_COL

    # If id_list is specified, then filter fam_df to include those individuals
    if sample_indices_to_keep:
        fam_df = fam_df.iloc[sample_indices_to_keep]

    if pheno_filename:
        phen = pd.read_csv(pheno_filename, sep=r"\s+",
                           names=[FAM_FID_COL, FAM_IID_COL, PHENOFILE_PHENO_COL], index_col=False)
        fam_df = fam_df.merge(phen, on=[FAM_FID_COL, FAM_IID_COL], how = "inner", copy=False)
        pheno_col = PHENOFILE_PHENO_COL

    return fam_df[pheno_col].to_numpy()


# -------------------------
# Creates a dataframe of FID, IID and Index number (from 0) from the .fam file or a .fam df
def _get_id_df_from_fam_file(fam_filename: Union[str, pd.DataFrame],
                             sample_indices_to_keep: list[int]) -> pd.DataFrame:
    
    # Make id_df using either a fam file or a fam dataframe
    if isinstance(fam_filename, str):
        id_df = pd.read_csv(fam_filename, sep=r"\s+", usecols=(0, 1), names=[FID_COL, IID_COL],
                            index_col=False)
    elif isinstance(fam_filename, pd.DataFrame):
        id_df = fam_filename[[FID_COL, IID_COL]]
    else:
        raise TypeError(f"Type of parameter fam_file ({type(fam_filename)}) is not supported.")  
    
    # If id_list is specified, then filter id_df to include those individuals
    if sample_indices_to_keep:
        id_df = id_df.iloc[sample_indices_to_keep] # TODO(jonbjala) Confirm this works as desired
          
    # Create an index col 
    id_df[INDEX_COL] = range(len(id_df))
    
    return id_df

# -------------------------
# Creates a dataframe of covariates that is filled to the size of the fam file 
def format_covar_file(covar_filename: Union[str, pd.DataFrame],
                      fam_filename: Union[str, pd.DataFrame],
                      sample_indices_to_keep: List[int]) -> np.ndarray:
    
    # Make id_df using either a fam file or a fam dataframe (NEEDS HEADER IN FILE)
   # TODO(dhruvaj) If multiple covar files, then merge them into 1 file
    if isinstance(covar_filename, str):
        covar_df = pd.read_csv(covar_filename, sep=r"\s+", index_col=False)
    elif isinstance(covar_filename, pd.DataFrame):
        covar_df = covar_filename
    else:
        raise TypeError(f"Type of parameter fam_file ({type(covar_filename)}) is not supported.")    
    
    # Get id_df to merge FID and IID and set any missing covariates to NA
    id_df = _get_id_df_from_fam_file(fam_filename, sample_indices_to_keep)
    
    covar_df = id_df.merge(covar_df, on=[FID_COL, IID_COL], copy = False, how = "left")
    covar_df = covar_df.drop(columns=[FID_COL, IID_COL, INDEX_COL])
    
    return covar_df.to_numpy()

# -------------------------
def convert_king_output_to_rel_info(
    king_output: Union[str, pd.DataFrame], fam_filename: str, rel_degree: Union[str, float],
    sample_indices_to_keep: List[int]=None, rel_info_file: str = ""
) -> tuple[THRESHOLDED_REL_TYPE, np.ndarray, sp.csr_array]:

    if rel_info_file:
        with open(rel_info_file, 'rb') as f:
            rel_info = pickle.load(f)
        # Store rel_set_sizes
        rel_set_sizes = np.array([len(rel_list) for rel_list in rel_info], dtype=float)
        return rel_info, rel_set_sizes
    
    king_time = time.time()
    logging.info(f'Relationship degree is {rel_degree}.')
    # Read in King output
    if isinstance(king_output, str):
        king_df = pd.read_csv(king_output, sep=r"\s+")[NEEDED_KING_COLS]
    elif isinstance(king_output, pd.DataFrame):
        king_df = king_output
    else:
        raise TypeError(f"Type of parameter king_output ({type(king_output)}) is not supported.")

    # Convert InfTypes using INF_TO_DEG_MAP
    king_df[KING_REL_COL] = king_df[KING_REL_COL].map(INF_TO_DEG_MAP)
    logging.info(f"Reading in king output took {time.time() - king_time} seconds")

    king_time = time.time()
    # Filter down to SE relatedness threshold first
    # TODO(jonbjala) This will need to change if we bring back higher degree thresholds
    king_df = king_df[king_df[KING_REL_COL] <= SE_RELATEDNESS]

    # Use the .fam file to get FID/IID mapping to person number
    id_df = _get_id_df_from_fam_file(fam_filename, sample_indices_to_keep)
    N = len(id_df)
    logging.info(f'Number of individuals to group is {N}')

    # Construct DataFrame that contains person number (INDEX) pairs that are related along with
    # their degree of relation (from king output)
    # By merging id_df into king_df twice, obtain king_df with index columns mapping to a unique ID.
    id_df.rename(
        inplace=True,
        columns={FID_COL: KING_FID1_COL, IID_COL: KING_IID1_COL, INDEX_COL: INDEX1_COL},
    )
    king_df = king_df.merge(id_df, on=[KING_FID1_COL, KING_IID1_COL], copy=False)
    id_df.rename(
        inplace=True,
        columns={
            KING_FID1_COL: KING_FID2_COL,
            KING_IID1_COL: KING_IID2_COL,
            INDEX1_COL: INDEX2_COL,
        },
    )
    king_df = king_df.merge(id_df, on=[KING_FID2_COL, KING_IID2_COL], copy=False)

    # Create the SE info object
    i1 = king_df[INDEX1_COL].to_numpy(dtype=np.int64, copy=False)
    i2 = king_df[INDEX2_COL].to_numpy(dtype=np.int64, copy=False)
    rows = np.concatenate([i1, i2])
    cols = np.concatenate([i2, i1])
    data = np.ones(rows.shape[0], dtype=np.int8)
    se_info = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    se_info.setdiag(1)
    se_info.eliminate_zeros()
    del i1, i2, rows, cols, data
    logging.info(f"Creating SE object took {time.time() - king_time} seconds")

    king_time = time.time()
    # Pare down the King dataframe to the relatedness threshold requested by the user
    king_df = king_df[king_df[KING_REL_COL] <= REL_TO_DEG_MAP[rel_degree]]


    # Find the minimum value of KING_REL_COL given groups of indices in index cols.
    # Then combine index, degree pairs into a single lowest_degree series.
    king_time = time.time()
    # Pandas Series are 1 dimensional so each element contains a group (index and min rel degree).
    # However, the index is based on by=INDEX_COL so you can access the correct group in the series
    # by using the original index number
    ind1_mins = king_df.groupby(by=INDEX1_COL)[KING_REL_COL].min()
    ind2_mins = king_df.groupby(by=INDEX2_COL)[KING_REL_COL].min()

    # Initialize a Pandas Series that contains every index number with a degree that is too high
    max_degree_series = pd.Series([MAX_RELATEDNESS + 1] * N)

    # Combine the two ind_mins Series into a Series that contains the min degree of individuals
    #  who have sufficiently close relatives.
    # Then, to have a complete Series containing all indices of individuals,
    #  combine that with max_degree_series
    # Since, it's a Pandas series, the order of the indices doesn't matter
    #  (though indices are usually sorted by default).
    # The lowest_degree[person_num] will give the lowest degree for that person regardless of
    #  the positional index value.
    lowest_degree = ind1_mins.combine(ind2_mins, min, MAX_RELATEDNESS + 1)
    lowest_degree = lowest_degree.combine(max_degree_series, min, MAX_RELATEDNESS + 1)
    logging.info(f"Finding the lowest value of relation for all people and combining into "
                 f"series takes {time.time() - king_time} seconds")

    index1_degree = king_df[INDEX1_COL].map(lowest_degree)
    index2_degree = king_df[INDEX2_COL].map(lowest_degree)


    df1 = king_df[index1_degree == king_df[KING_REL_COL]]
    df2 = king_df[index2_degree == king_df[KING_REL_COL]]


    rows = np.concatenate([df1[INDEX1_COL], df2[INDEX2_COL],
                           lowest_degree.index.to_numpy()])
    cols = np.concatenate([df1[INDEX2_COL], df2[INDEX1_COL],
                           lowest_degree.index.to_numpy()])

    data = np.ones(rows.shape[0], dtype=float)
    R_matrix = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    R_matrix.eliminate_zeros()

    R_matrix *= -np.reciprocal(R_matrix.sum(axis=1)).reshape((N,1))

    R_matrix +=  sp.identity(N, dtype=float)
    logging.info(f"Creating R matrix took {time.time() - king_time} seconds")
    # # Create the SE info object
    # i1 = king_df[INDEX1_COL].to_numpy(dtype=np.int64, copy=False)
    # i2 = king_df[INDEX2_COL].to_numpy(dtype=np.int64, copy=False)
    # rows = np.concatenate([i1, i2])
    # cols = np.concatenate([i2, i1])
    # data = np.ones(rows.shape[0], dtype=np.int8)
    # se_info = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    # se_info.setdiag(1)
    # se_info.eliminate_zeros()
    # del i1, i2, rows, cols, data


    # # Create the R matrix
    # #king_df['INDEX1_MIN_REL'] = 



    # i1 = king_df[INDEX1_COL].to_numpy(dtype=np.int64, copy=False)
    # i2 = king_df[INDEX2_COL].to_numpy(dtype=np.int64, copy=False)
    # rows = np.concatenate([i1, i2])
    # cols = np.concatenate([i2, i1])
    # data = np.ones(rows.shape[0], dtype=np.int8)
    # se_info = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    # se_info.setdiag(1)
    # se_info.eliminate_zeros()
    # del i1, i2, rows, cols, data


    # rows = np.array([i for i, sublist in enumerate(rel_info) for v in sublist])
    # cols = np.array([v for sublist in rel_info for v in sublist])
    # data = np.array([-1.0/len(sublist) for sublist in rel_info for v in sublist])
    # R_matrix = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    # R_matrix.setdiag(R_matrix.diagonal() + 1.0)
    # R_matrix.eliminate_zeros()



    # Store a list of lists that contains groups of individuals who are closely related
    #king_time = time.time()
    # rel_info = [
    #     list(
    #         sorted(
    #             set(
    #                 it.chain(
    #                     king_df[INDEX1_COL][
    #                         (king_df[INDEX2_COL] == person_num)
    #                         & (king_df[KING_REL_COL] == lowest_degree[person_num])
    #                     ],
    #                     king_df[INDEX2_COL][
    #                         (king_df[INDEX1_COL] == person_num)
    #                         & (king_df[KING_REL_COL] == lowest_degree[person_num])
    #                     ],
    #                     [person_num],
    #                 )
    #             )
    #         )
    #     )
    #     for person_num in range(N)
    # ]

    # logging.info(f"Making rel_lists takes {time.time() - king_time} seconds.")

    # rel_set_sizes = np.array([len(rel_list) for rel_list in rel_info], dtype=float)
    # logging.info(f"Max of rel set sizes is {max(rel_set_sizes)}")
    # logging.info(f"Min of rel set sizes is {min(rel_set_sizes)}")
    
    # counter = sum(len(inner_list) > 1 for inner_list in rel_info)
    # logging.info(f'Num focal individuals is {counter}')

    
    # print(f"GRMA {rel_info=}")
    # print(f"GRMA {R_matrix.toarray()=}")

    return R_matrix, se_info

# -------------------------

def calculate_R_matrix(rel_info: THRESHOLDED_REL_TYPE, rel_set_sizes: np.ndarray = None
    ) -> sp.csr_matrix:
    
    mat_time = time.time()
    logging.info(f"Creating R matrix")

    # Create the R matrix
    N = len(rel_info)
    rows = np.array([i for i, sublist in enumerate(rel_info) for v in sublist])
    cols = np.array([v for sublist in rel_info for v in sublist])
    data = np.array([-1.0/len(sublist) for sublist in rel_info for v in sublist])
    R_matrix = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    R_matrix.setdiag(R_matrix.diagonal() + 1.0)
    R_matrix.eliminate_zeros()

    logging.info(f"Time to create R matrix {time.time() - mat_time}")
    return R_matrix
# -------------------------
def demean_phenotypes(phenotypes: np.ndarray, R_matrix: sp.csr_array) -> np.ndarray:
    # TODO(jonbjala)  Handle missing phenotype values?
    # N = len(phenotypes)
    # rel_set_sizes = (np.array([len(rel_list) for rel_list in rel_info], dtype=float)
    #                  if rel_set_sizes is None else rel_set_sizes)  # TODO(jonbjala) Make this a function?

    # mean_phenos = (np.fromiter((np.sum(phenotypes[rel_list]) for rel_list in rel_info),
    #                    dtype=float, count=N,)/ rel_set_sizes)

    return R_matrix @ phenotypes

# -------------------------
def residualize_phenotypes_on_covars(phenotypes:np.ndarray, covars:np.ndarray) -> np.ndarray:
    # Regress phenotypes on covariates - assumes phenotypes has no NAs
    # Store rows where covar is NA
    valid_rows = ~np.isnan(covars).any(axis=1)

    # Add constant to X matrix
    valid_covars = covars[valid_rows]
    const = np.ones((valid_covars.shape[0], 1)) 
    valid_covars = np.hstack((valid_covars, const))

    # Run reg
    x, _, _, _ = np.linalg.lstsq(a = valid_covars, b = phenotypes[valid_rows], rcond = None)
    
    # Calc residuals
    pred_vals = np.dot(valid_covars, x)
    ols_resids = phenotypes[valid_rows] - pred_vals
    
    # Fill residuals with correct residuals and NAs for missing covars
    residuals = np.full(len(phenotypes), np.nan)    
    residuals[valid_rows] = ols_resids 
    
    return residuals
        

# -------------------------
def residualize_genotypes(
    genotypes: np.ndarray,
    R_matrix: sp.csr_array,
) -> np.ndarray:
    # geno_time = time.time()

    # # genotypes has dimension num_snps x N. The entry in the X matrix is the score (num alleles - 0, 1, or 2)
    # mean_genos = np.vstack([np.nanmean(genotypes[:, rel_list], axis=1) for rel_list in rel_info]).T

    # # Subtract row mean value from each snp for the row composed of the related group individuals. 
    # # np.nanmean throws a warning because there are some rows that are all NaNs. It's still correct.
    # logging.info(f"Residualizing genotypes takes {time.time() - geno_time} seconds")

    return (R_matrix @ genotypes.T).T


# -------------------------
def calculate_ses(R_matrix: sp.csr_array, se_info: sp.csr_array, residuals: np.ndarray,
                  residualized_genotypes: np.ndarray, residualized_phenotypes: np.ndarray) -> np.ndarray:
    ses_time = time.time()

    XtX = np.nansum(np.square(residualized_genotypes), axis=1) # This is X'X = M x 1
    
    M, N = residualized_genotypes.shape
    
    ses = np.zeros(M, dtype=float)
    for snp in range(M):
        snp_residuals = residuals[snp].reshape((N,1))

        center_matrix = R_matrix @ (snp_residuals.T * se_info * snp_residuals) @ R_matrix.T

        ses[snp] = np.sqrt(residualized_genotypes[snp] @ center_matrix @ residualized_genotypes[snp])


    ses /= XtX

    logging.info(f"Time to calculate ses is {time.time() - ses_time}")
    return ses

# -------------------------
def run_regressions(
    genotypes: np.ndarray, phenotypes: np.ndarray,
    residualized_genotypes: np.ndarray, residualized_phenotypes: np.ndarray,
    R_matrix: sp.csr_array, se_info: sp.csr_array) -> tuple[np.ndarray, np.ndarray]:

    reg_time = time.time()

    G_sq_sum_per_snp = np.nansum(np.square(residualized_genotypes), axis=1)

    betas = (
        np.nansum(residualized_genotypes * residualized_phenotypes, axis=1)
        / G_sq_sum_per_snp
    )

    M_sub, N = genotypes.shape
    residuals = -(betas * genotypes.T - phenotypes.reshape((N, 1))).T

    ses = calculate_ses(        
        R_matrix=R_matrix,
        se_info=se_info,
        residuals=residuals,
        residualized_genotypes=residualized_genotypes,
        residualized_phenotypes=residualized_phenotypes
    )

    logging.info(f"Running regressions takes {time.time() - reg_time}")
    return betas, ses, G_sq_sum_per_snp
# -------------------------
def get_var_y(residualized_phenotypes: np.ndarray) -> float:
    N = len(residualized_phenotypes)
    return np.dot(residualized_phenotypes, residualized_phenotypes) / N

# -------------------------
def calculate_pvals(betas: np.ndarray, ses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Creates z-statistics and p-values from a 2-tailed test from betas and ses
    zstats = betas / ses
    pvals = 2 * (norm.sf(np.abs(zstats)))
    
    return pvals

# -------------------------
def combine_results_with_bim_file(betas: np.ndarray, ses: np.ndarray,
                                  pvals: np.ndarray, sum_sq_x: np.ndarray,
                                  bim_filename: str, snp_indices_to_keep: List[int]
                                  ) -> pd.DataFrame:
    
    combined_df = pd.DataFrame({
        OUTPUT_BETA_COL: betas,
        OUTPUT_SE_COL: ses,
        OUTPUT_P_COL: pvals, 
        OUTPUT_SUMSQX_COL: sum_sq_x,
        })

    bim_df = pd.read_csv(bim_filename, sep='\t', header=None, names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    if snp_indices_to_keep:
        bim_df = bim_df.iloc[snp_indices_to_keep]

    # Append bim file
    results_df = pd.concat([bim_df, combined_df], axis=1)

    return results_df

# -------------------------
def subset_genotypes(
    genotypes: np.ndarray,
    sample_indices_to_keep: List[int],
    snp_indices_to_keep: List[int], 
    M_start: int, 
    snps_per_block: int
) -> np.ndarray:
    # Subset genotypes to only include individuals and SNPs of interest
    if sample_indices_to_keep:
        genotypes = genotypes[:, sample_indices_to_keep]
    # Get the correct indices of the SNPs to keep
    if snp_indices_to_keep:
        snps = [id % snps_per_block for id in snp_indices_to_keep if M_start <= id < M_start + snps_per_block]
        genotypes = genotypes[snps, :]
        
    return genotypes

# -------------------------
# TODO(jonbjala) May want to refactor a bit and reduce the file IO here and push that back out
#                to the grma CLI (or at least explore doing so)
def grma(
    *,
    rel_input: Union[str, pd.DataFrame, np.ndarray],
    bed_file: str,
    bim_file: str,
    fam_file: str,
    rel_info_file: str = "",
    pheno_file: str = "",
    covar_file: str = "",
    rel_degree: str = "1",
    snps_per_block: int = DEFAULT_SNPS_PER_BLOCK,
    id_list: str = "",
    snp_list: str = ""
) -> pd.DataFrame:

    # TODO(jonbjala) Add SE info flag

    logging.info(f"\nBeginning grma() for {bed_file}")
    logging.debug(
        f"\t{rel_input=}\n\t{bed_file=}\n\t{bim_file=}\n\t{fam_file=}\n\t{pheno_file=}"
        f"\n\t{covar_file=}\n\t{rel_degree=}\n\t{snps_per_block=}"
    )

    # Get basic information like number of SNPs
    M = get_num_snps_from_bim_file(bim_file)
    logging.debug(f"\t{M=}")
    
    # Get number of individuals
    N = get_sample_size_from_fam_file(fam_file)
    logging.debug(f"\t{N=}")
    
    # Get the indices of the SNPs to keep
    snp_indices_to_keep = get_snp_indices_to_keep(snp_list, bim_file) if snp_list else None
    num_snps = len(snp_indices_to_keep) if snp_indices_to_keep else M
    betas = np.zeros(num_snps)
    ses = np.zeros(num_snps)
    sum_sq_x = np.zeros(num_snps)
    
    # Get the indices of the individuals to keep
    sample_indices_to_keep = get_sample_indices_to_keep(id_list, fam_file) if id_list else None

    # Construct the relatedness object
    logging.debug("Converting King output to actionable relatedness info...")
    start_time = time.time()
    R_matrix, se_info = convert_king_output_to_rel_info(
        king_output=rel_input, fam_filename=fam_file, rel_degree=rel_degree,
        rel_info_file=rel_info_file, sample_indices_to_keep=sample_indices_to_keep
    )
    logging.info(f"Processed King output in {time.time() - start_time} seconds")

    # Calculate the relatedness matrix
    #R_matrix = calculate_R_matrix(rel_info=rel_info, rel_set_sizes=rel_set_sizes)

    # Retrieve raw phenotypes from the file
    p_not_demeaned = get_phenotypes_from_file(pheno_filename=pheno_file,
                                              fam_filename=fam_file,
                                              sample_indices_to_keep=sample_indices_to_keep)
               
    # Incorporate / residualize on covariates if they exist
    if covar_file:
        logging.info("Residualizing phenotypes on covariates...")
        start_time = time.time()
        p_not_demeaned = residualize_phenotypes_on_covars(
            phenotypes=p_not_demeaned,
            covars=format_covar_file(covar_file, fam_file, id_list)
        )
        logging.info(f"Residualized phenotypes on covariates in {time.time() - start_time} seconds")
  
    # Demean the phenotypes
    logging.debug("Demeaning phenotypes...")
    start_time = time.time()
    P = demean_phenotypes(
               phenotypes=p_not_demeaned,
               R_matrix=R_matrix,
        )
    logging.info(f"Demeaned the phenotypes in {time.time() - start_time} seconds")  

    # Residualize the genotypes and run the regressions for each block of SNPs
    logging.debug("Residualizing genotypes and running regressions...")
    start_time = time.time()

    num_blocks = int(np.ceil(M / snps_per_block))
    logging.debug(f"\t{num_blocks=}")

    for block_num in range(num_blocks):
        M_start = block_num * snps_per_block
        num_snps_in_block = min(M - M_start, snps_per_block)

        genotypes=subset_genotypes(
            genotypes=read_bed_file(
                bed_filename=bed_file,
                N=N,
                M=M,
                M_start=M_start,
                num_snps=num_snps_in_block
            ), 
            sample_indices_to_keep=sample_indices_to_keep, 
            snp_indices_to_keep=snp_indices_to_keep, 
            M_start=M_start, 
            snps_per_block=snps_per_block
        )

        block_betas, block_ses, block_sum_sq_x = run_regressions(
            genotypes=genotypes,
            phenotypes=p_not_demeaned,
            residualized_genotypes=residualize_genotypes(
                genotypes=genotypes,
                R_matrix=R_matrix
            ),
            residualized_phenotypes=P,
            R_matrix=R_matrix,
            se_info=se_info
        )
        betas[M_start: M_start + len(block_betas)] = block_betas
        ses[M_start: M_start + len(block_ses)] = block_ses
        sum_sq_x[M_start : M_start + len(block_sum_sq_x)] = block_sum_sq_x
    
        
    logging.info(f"Residualized genotypes and ran regressions in {time.time() - start_time} seconds")
    logging.info(f"Var_y for rel_degree {rel_degree} is {get_var_y(P)} ")
    
    pvals = calculate_pvals(betas=betas, ses=ses)
    results = combine_results_with_bim_file(betas=-betas, ses=ses, pvals=pvals,
                                            sum_sq_x=sum_sq_x, bim_filename=bim_file,
                                            snp_indices_to_keep=snp_indices_to_keep)
    
    return results


#################################
if __name__ == "__main__":
    print("This script is not meant to be called directly.")

    