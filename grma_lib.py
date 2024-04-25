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
KING_REL_COL = "InfType"
KING_FID1_COL = "FID1"
KING_IID1_COL = "ID1"
KING_FID2_COL = "FID2"
KING_IID2_COL = "ID2"
NEEDED_KING_COLS = [
    KING_FID1_COL,
    KING_IID1_COL,
    KING_FID2_COL,
    KING_IID2_COL,
    KING_REL_COL,
]

# Column name used to label phenotype when pulled in from separate phenotype file
PHENOFILE_PHENO_COL = "Phenofile_Phenotype"

# List of inputs to accept as flags to specify degree of relation allowed
REL_DEG_INPUTS = ["FS", "1", "2", "3", "4"]

# List of possible InfTypes in a King output file. 
INF_TO_DEG_MAP = {"Dup/MZTwin": 1, "FS": 1, "PO": 1, "2nd": 2, "3rd": 3, "4th": 4, "UN": 5,}

# Default number of SNPs to process at a time
DEFAULT_SNPS_PER_BLOCK = 100


# -------------------------
def get_phenotypes_from_file(pheno_filename: str, fam_filename: str):
    # TODO(jonbjala)  Handle missing phenotype values?  Have support for case/control?  https://www.cog-genomics.org/plink/1.9/formats#fam
    # Drop phenotype values that are missing

    fam_df = pd.read_csv(fam_filename, sep=r"\s+", names=FAM_COLS, index_col=False)
    pheno_col = FAM_PHENO_COL

    if pheno_filename:
        phen = pd.read_csv(pheno_filename, sep=r"\s+", names=[FAM_FID_COL, FAM_IID_COL, PHENOFILE_PHENO_COL], index_col=False,)
        fam_df = fam_df.merge(phen, on=[FAM_FID_COL, FAM_IID_COL], copy=False)
        pheno_col = PHENOFILE_PHENO_COL

    return fam_df[pheno_col].to_numpy()


# -------------------------


# Creates a dataframe of FID, IID and Index number (from 0) from the .fam file or a .fam df
def _get_id_df_from_fam_file(fam_filename: str) -> int:
    
    # Make id_df using either a fam file or a fam dataframe
    if isinstance(fam_filename, str):
        id_df = pd.read_csv(fam_filename, sep=r"\s+", usecols = (0, 1), names = [FID_COL, IID_COL], index_col=False)
        
    elif isinstance(fam_filename, pd.DataFrame):
        id_df = fam_filename[[FID_COL, IID_COL]]
    else:
        raise TypeError(f"Type of parameter fam_file ({type(fam_filename)}) is not supported.")    

    # Create an index col 
    id_df[INDEX_COL] = range(len(id_df))
    
    return id_df


# -------------------------
def convert_king_output_to_rel_info(
    king_output: Union[str, pd.DataFrame], fam_filename: str, rel_degree: str
) -> tuple[THRESHOLDED_REL_TYPE, np.ndarray]:
    MAX_RELATEDNESS = 4  # Maximum degree of relatedness from king output

    king_time = time.time()
    # Read in King output and filter out unneeded rows (where relatedness is too weak)
    if isinstance(king_output, str):
        king_df = pd.read_csv(king_output, sep=r"\s+")[NEEDED_KING_COLS]
    elif isinstance(king_output, pd.DataFrame):
        king_df = king_output
    else:
        raise TypeError(f"Type of parameter king_output ({type(king_output)}) is not supported.")
    logging.info(f"Reading in king output takes {time.time() - king_time} seconds")

    # Using INF_TO_DEG_MAP, if FS, then throw everything other than FS and Dup/MZTwin and convert to 1 (for later closest relative processing).
    # If not FS, then convert FS and PO to 1.
    # Keep Dup/MZTwin (and FS) in all cases (converted to 1).
    king_time = time.time()
    
    if rel_degree == "FS":
        king_df = king_df[king_df[KING_REL_COL].isin(["FS", "Dup/MZTwin"])]
        king_df[KING_REL_COL] = king_df[KING_REL_COL].map(INF_TO_DEG_MAP)
    else:
        # Replace values of PO, FS with 1 and others with their numeric value. Then, keep only those degrees that are closer.
        king_df[KING_REL_COL] = king_df[KING_REL_COL].map(INF_TO_DEG_MAP)
        king_df = king_df[king_df[KING_REL_COL] <= int(rel_degree)]
    logging.info(f"Converting degrees to numbers takes {time.time() - king_time} seconds")
    # Use the .fam file to get FID/IID mapping to person number
    king_time = time.time()
    id_df = _get_id_df_from_fam_file(fam_filename)
    N = len(id_df)
    logging.info(f"Getting id_df takes {time.time() - king_time} seconds")

    # Construct DataFrame that contains person number (INDEX) pairs that are related along with their degree of relation (from king output)
    king_time = time.time()
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
    print(f"Len of king_df is {len(king_df)}")
    print(f"Length of ID df is {N}")
    logging.info(f"Merging king_df and id_df takes {time.time() - king_time} seconds")
    # Find the minimum value of KING_REL_COL given groups of indices in index cols. Then combine index, degree pairs into a single lowest_degree series.
    king_time = time.time()
    ind1_mins = king_df.groupby(by=INDEX1_COL)[KING_REL_COL].min()
    ind2_mins = king_df.groupby(by=INDEX2_COL)[KING_REL_COL].min()

    # Initialize a Pandas Series that contains every index number with a degree that is too high
    max_degree_series = pd.Series([MAX_RELATEDNESS + 1] * N)

    # Combine the two ind_mins Series into a Series that contains the min degree of individuals who have sufficiently close relatives
    # Then, to have a complete Series containing all indices of individuals, combine that with max_degree_series
    lowest_degree = ind1_mins.combine(ind2_mins, min, MAX_RELATEDNESS + 1)
    lowest_degree = lowest_degree.combine(max_degree_series, min, MAX_RELATEDNESS + 1)
    logging.info(f"Finding the lowest value of relation for all people and combining into series takes {time.time() - king_time} seconds")
    # Store a list of lists that contains groups of individuals who are closely related
    king_time = time.time()
    rel_info = [
        list(
            sorted(
                set(
                it.chain(
                    king_df[INDEX1_COL][
                        (king_df[INDEX2_COL] == person_num)
                        & (king_df[KING_REL_COL] == lowest_degree[person_num])
                    ],
                    king_df[INDEX2_COL][
                        (king_df[INDEX1_COL] == person_num)
                        & (king_df[KING_REL_COL] == lowest_degree[person_num])
                    ],
                    [person_num],
                )
                )
            )
        )
        for person_num in range(N)
    ]
    logging.info(f"Making rel_lists takes {time.time() - king_time} seconds.")
    rel_set_sizes = np.array([len(rel_list) for rel_list in rel_info], dtype=float)
    print(f"Max of rel set sizes is {max(rel_set_sizes)}")
    print(f"Min of rel set sizes is {min(rel_set_sizes)}")
    
    """file_path = '/disk/genetics3/data_dirs/ukb/private/v3/processed/user/dhruvaj/grma_ukb_testing/rel_info_degFS.txt'
    rel_info_str = str(rel_info)
    with open(file_path, "w") as f:
        f.write(rel_info_str)
    """        
    return rel_info, rel_set_sizes

# -------------------------

def calculate_neff(rel_info: THRESHOLDED_REL_TYPE, rel_set_sizes: np.ndarray) -> int:
    
    # Implementing rank algorithm where duplicates are dropped from the matrix before calculating rank
    # Finding the total number of pairs, triplets, and other duplicate groups uptil the largest group size and storing the number of linearly independent cols as num_dups
    # while storing the indices that compose these duplicates
    dup_indices = [] 
    num_dups = 0
    for rel_length in range(1, int(max(rel_set_sizes) + 1)):
        bool_vals = [True] * len(rel_info)
        for index, inner_list in enumerate(rel_info):
            if len(inner_list) == rel_length:
                if all(bool_vals[i] for i in inner_list) == True:
                    indices_to_check = [i for i in inner_list if i != index]
                    sublists_to_check = [rel_info[i] for i in indices_to_check]
                    # all() returns true if iterable (sublists_to_check) is empty so works to find singletons as well.
                    if len(sublists_to_check) == rel_length - 1 and all(sorted(sublist) == sorted(inner_list) for sublist in sublists_to_check):
                        num_dups += 1 * (rel_length - 1)
                        dup_indices.append(tuple(inner_list))
                        for i in inner_list:
                            bool_vals[i] = False

    # Taking the set of indices that compose these duplicates
    old_indices_of_dups = set(it.chain(*set(dup_indices)))

    # Finding indices that aren't duplicates to create N_matrix
    non_dup_indices = [pindex for pindex, pval in enumerate(rel_info) if pindex not in old_indices_of_dups]
    
    # Calculate indices of non-duplicate people / samples
    num_samples_non_dup = len(non_dup_indices)
    logging.debug(f"Number of non-singleton individuals = {num_samples_non_dup}")

    # Construct a reverse lookup to compensate for dropped duplicates when referring to rel_info
    # the indices of non-duplicates are set to values that np.arange produces (0, 1, 2, 3...). everything else (the duplicates' indices) is 0.
    # reverse indices has, at the old indices of non-duplicates, the new indices of the non-duplicates. therefore, reverse_indices[old_pindex] = new_pindex
    reverse_indices = np.zeros(len(rel_info), dtype=int)
    reverse_indices[non_dup_indices] = np.arange(num_samples_non_dup)

    # Construct matrix whose rank needs to be evaluated
    
    neff_time = time.time()
    N_matrix = np.identity(num_samples_non_dup, dtype=float)
    for new_pindex, old_pindex in enumerate(non_dup_indices):
        # new p_vals is the result of converting the inner list in rel_info to a list of the new_pindices to know which columns to index into
        # the length of new_pvals then gives the number we should find the reciprocal of for the matrix R
        new_pvals = [reverse_indices[non_adj_person_num]
                     for non_adj_person_num in rel_info[old_pindex]]
        # new_pindex is the row, new_pvals are the columns
        N_matrix[new_pindex, new_pvals] -= np.reciprocal(len(new_pvals), dtype=float)
    logging.info(f"Time to create matrix {time.time() - neff_time}")
    
    # Calculating the rank of the smaller submatrix
    start_time = time.time()
    logging.info("Calculating rank of subset matrix")
    subset_rank = np.linalg.matrix_rank(N_matrix)
    logging.info(f"Num duplicates of any group size = {num_dups}")
    logging.info(f"subset rank is {subset_rank}")
    n_eff = num_dups + subset_rank
    logging.info(f"N_eff is {n_eff}")
    logging.info(f"total time taken to calculate smaller subset is {time.time() - start_time}")
    

    return n_eff

# -------------------------
def residualize_phenotypes(
    phenotypes: np.ndarray,
    rel_info: THRESHOLDED_REL_TYPE,
    rel_set_sizes: np.ndarray = None,
) -> np.ndarray:
    # TODO(jonbjala)  Handle missing phenotype values? DROP THEM
    N = len(phenotypes)
    rel_set_sizes = (np.array([len(rel_list) for rel_list in rel_info], dtype=float) if rel_set_sizes is None else rel_set_sizes)  # TODO(jonbjala) Make this a function?

    mean_phenos = (np.fromiter((np.sum(phenotypes[rel_list]) for rel_list in rel_info), dtype=float, count=N,)/ rel_set_sizes)

    return phenotypes - mean_phenos


# -------------------------
def residualize_genotypes(
    genotypes: np.ndarray,
    rel_info: THRESHOLDED_REL_TYPE,
    *,
    rel_set_sizes: np.ndarray = None,
) -> np.ndarray:
    # TODO(jonbjala) Might want to experiment with different numpy API calls and approaches to see if there are good speed / memory tradeoffs
    # mean_genos = np.vstack([np.sum(G[:, rel_list], axis=1) for rel_list in rel_info]).T / rel_set_sizes
    # genotypes has dimension num_snps x N
    geno_time = time.time()
    mean_genos = np.vstack([np.nanmean(genotypes[:, rel_list], axis=1) for rel_list in rel_info]).T
    empty_slices = sum(1 for rel_list in rel_info if np.isnan(genotypes[:, rel_list]).all())
    print(f"Number of empty slices: {empty_slices}")
    logging.info(f"Residualizing genotypes takes {time.time() - geno_time} seconds")
    return genotypes - mean_genos


# -------------------------
def run_regressions(
    residualized_genotypes: np.ndarray, residualized_phenotypes: np.ndarray, N_eff: int
) -> tuple[np.ndarray, np.ndarray]:
    # TODO(jonbjala) The math could (will) change when adding covariates, and the calculations for SEs have
    #                not been fully vetted by Patrick
    reg_time = time.time()
    G_sq_sum_per_snp = np.nansum(np.square(residualized_genotypes), axis=1)

    betas = (
        np.nansum(residualized_genotypes * residualized_phenotypes, axis=1)
        / G_sq_sum_per_snp
    )
    ses = np.sqrt(np.dot(residualized_phenotypes, residualized_phenotypes) / (G_sq_sum_per_snp * N_eff))
    logging.info(f"Running regressions takes {time.time() - reg_time}")
    return betas, ses

# -------------------------
def calculate_Zstats_and_pvals(betas: np.ndarray, ses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Creates z-statistics and p-values from a 2-tailed test from betas and ses
    zstats = betas / ses
    pvals = 2 * (1 - norm.cdf(np.abs(zstats)))
    
    return zstats, pvals

# -------------------------
def combine_results_with_bim_file(betas: np.ndarray, ses: np.ndarray, zstats: np.ndarray, pvals: np.ndarray, bim_filename: str) -> pd.DataFrame:
    
    combined_df = pd.DataFrame({
        'Beta': betas,
        'SE': ses,
        'Z': zstats,
        'Pval': pvals
        })
    
    bim_df = pd.read_csv(bim_filename, sep='\t', header=None, names=['chr', 'id', 'pos', 'bpcoord', 'A1', 'A2'])
    # Append bim file
    results_df = pd.concat([combined_df, bim_df], axis=1)

    return results_df

# -------------------------
def grma(
    *,
    rel_input: Union[str, pd.DataFrame, np.ndarray],
    bed_file: str,
    bim_file: str,
    fam_file: str,
    pheno_file: str = "",
    covar_file: str = "",
    rel_degree: float = 0.125,
    snps_per_block: int = DEFAULT_SNPS_PER_BLOCK,
) -> pd.DataFrame:
    # TODO(jonbjala) Need to add the extra covariates at all levels of the software (covar_file)

    logging.info("Beginning grma()...")
    logging.debug(
        f"\t{rel_input=}\n\t{bed_file=}\n\t{bim_file=}\n\t{fam_file=}\n\t{pheno_file=}"
        f"\n\t{covar_file=}\n\t{rel_degree=}\n\t{snps_per_block=}"
    )

    # Get basic information like number of SNPs
    M = get_num_snps_from_bim_file(bim_file)
    logging.debug(f"\t{M=}")

    # Construct the vector to hold the results
    betas = np.zeros(M)
    ses = np.zeros(M)

    # Construct the relatedness object
    logging.debug("Converting King output to actionable relatedness info...")
    start_time = time.time()
    rel_info, rel_set_sizes = convert_king_output_to_rel_info(
        rel_input, fam_file, rel_degree
    )
    logging.info(f"Processed King output in {time.time() - start_time} seconds")

    #Calculate the effective N
    logging.debug("Calculating effective N...")
    start_time = time.time()
    N_eff = calculate_neff(rel_info=rel_info, rel_set_sizes=rel_set_sizes)
    logging.info(f"Calculated effective N in {time.time() - start_time} seconds")


    # Residualize the phenotypes
    logging.debug("Residualizing phenotypes...")
    start_time = time.time()
    P = residualize_phenotypes(
        phenotypes=get_phenotypes_from_file(
            pheno_filename=pheno_file, fam_filename=fam_file
        ),
        rel_info=rel_info,
        rel_set_sizes=rel_set_sizes,
    )
    logging.info(f"Residualized the phenotypes in {time.time() - start_time} seconds")

    # Residualize the genotypes and run the regressions for each block of SNPs - (FWL?)
    logging.debug("Residualizing genotypes and running regressions...")
    start_time = time.time()

    num_blocks = int(np.ceil(M / snps_per_block))
    logging.debug(f"\t{num_blocks=}")

    for block_num in range(num_blocks):
        M_start = block_num * snps_per_block
        num_snps_in_block = min(M - M_start, snps_per_block)

        block_betas, block_ses = run_regressions(
            residualized_genotypes=residualize_genotypes(
                genotypes=read_bed_file(
                    bed_filename=bed_file,
                    N=len(P),
                    M=M,
                    M_start=M_start,
                    num_snps=num_snps_in_block,
                ),
                rel_info=rel_info,
                rel_set_sizes=rel_set_sizes,
            ),
            residualized_phenotypes=P,
            N_eff=N_eff,
        )

        betas[M_start : M_start + len(block_betas)] = block_betas
        ses[M_start : M_start + len(block_ses)] = block_ses

    logging.info(f"Residualized genotypes and ran regressions in {time.time() - start_time} seconds")
    
    zstats, pvals = calculate_Zstats_and_pvals(betas=betas, ses=ses)
    results = combine_results_with_bim_file(betas=betas, ses=ses, zstats=zstats, pvals=pvals, bim_filename=bim_file)

    return results


#################################
if __name__ == "__main__":
    print("This script is not meant to be called directly.")
