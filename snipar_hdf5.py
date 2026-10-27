"""
Reading and writing of imputed parental genotype files, as written by snipar's impute.py.

This module is deliberately snipar-specific, while the user-facing flag (--imputed-parents) and
the library parameter (imputed_parental_file) are deliberately generic.  That mirrors how
bedbimfam.py is named for the formats it parses while the flag supplying them is --bfile: generic
at the boundary, format-specific where the parsing happens.  A generic module name here would
promise format-agnosticism that the dataset keys below immediately contradict.

The published snipar documentation of this format is out of date in at least two respects:
num_observed_parental_alleles does not appear in it at all, and standard_f is described as a
boolean when the code writes a per-SNP allele frequency vector.  The structure below was verified
against a file generated from snipar's own example data rather than taken from the docs.
"""

import logging

import h5py
import numpy as np


# Constants =========================

# Suffix snipar's impute.py gives its output
SNIPAR_FILE_SUFFIX = ".hdf5"


# Dataset keys -------

# [families x SNPs] float16.  For a family with one parent observed this holds the imputed
# MISSING parent; where neither is observed it holds the imputed AVERAGE of the two.  Families
# with both parents genotyped have no row here at all.
SNIPAR_IMPUTED_GTS_KEY = "imputed_par_gts"

# [families x SNPs] float16.  Count of parental alleles observed, valued 2, 3 or 4, and NaN where
# imputation fell back to its backup path.  Written unconditionally -- not gated on --robust,
# which is a flag on snipar's gwas.py and cannot affect this file.
SNIPAR_NUM_OBS_PAR_ALLELES_KEY = "num_observed_parental_alleles"

# [SNPs] float64.  Per-SNP allele frequency of allele1, folded into the imputed genotypes and
# removed again by GRMA.  Despite its name and its docstring, this is NOT a boolean.
SNIPAR_STANDARD_F_KEY = "standard_f"

# [families x 3] bool -- genuinely boolean, unlike pedigree's equivalent columns
SNIPAR_PARENTAL_STATUS_KEY = "parental_status"

# [individuals + 1 x 6] fixed-length bytes.  ROW 0 IS A HEADER, not a record.
SNIPAR_PEDIGREE_KEY = "pedigree"

# [families] fixed-length bytes, sorted lexicographically -- so position does NOT correspond to
# pedigree's ordering, and lookups must be by value
SNIPAR_FAMILIES_KEY = "families"

# [SNPs x k] fixed-length bytes.  Every value is stringified, including numerics: a coordinate
# arrives as b'1000', not an integer.
SNIPAR_BIM_VALUES_KEY = "bim_values"

# [k] fixed-length bytes, naming the columns of SNIPAR_BIM_VALUES_KEY
SNIPAR_BIM_COLUMNS_KEY = "bim_columns"


# parental_status columns -------

SNIPAR_PAR_STATUS_HAS_FATHER_COL = 0
SNIPAR_PAR_STATUS_HAS_MOTHER_COL = 1
SNIPAR_PAR_STATUS_SINGLE_PARENT_COL = 2


# pedigree columns, after the header row is dropped -------

SNIPAR_PED_FID_COL = 0
SNIPAR_PED_IID_COL = 1
SNIPAR_PED_FATHER_COL = 2
SNIPAR_PED_MOTHER_COL = 3
SNIPAR_PED_HAS_FATHER_COL = 4
SNIPAR_PED_HAS_MOTHER_COL = 5

SNIPAR_PED_COLS = ("FID", "IID", "FATHER_ID", "MOTHER_ID", "has_father", "has_mother")


# Column names sought within bim_columns -------

# These are SNIPAR's names for its own copy of a bim table, not PLINK's: a real file gives
# 'Chr'/'id'/'coordinate' where bedbimfam.py's BIM_* constants are 'CHR'/'RSID'/'BP'.  They are
# prefixed to keep that distinction unambiguous, since both modules otherwise have BIM_ constants.
#
# They are search keys rather than positional constants.  snipar stores bim_columns alongside
# bim_values precisely because the order is not fixed, and the spellings differ between its code
# paths -- a file written from real data gives 'Chr'/'id'/'coordinate' where the simulation path
# gives 'chrom'/'rsid'/'position'.  Each tuple lists the spellings seen, most recent first, and
# get_snipar_bim_column_index() resolves whichever is present.
SNIPAR_BIM_CHR_NAMES = ("Chr", "chrom")
SNIPAR_BIM_ID_NAMES = ("id", "rsid")
SNIPAR_BIM_BP_NAMES = ("coordinate", "position")
SNIPAR_BIM_A1_NAMES = ("allele1",)
SNIPAR_BIM_A2_NAMES = ("allele2",)


# Functions ======================================

# -------------------------
def decode_snipar_bytes(arr: np.ndarray) -> np.ndarray:
    """
    Converts an array of fixed-length byte strings into one of Python strings.  Every string
    valued dataset in this format is stored as bytes, and has to be decoded before it can be
    compared against values read from a fam or bim file, which pandas gives as str.
    """

    return np.char.decode(arr.astype("S"), "utf-8")


# -------------------------
def get_snipar_bim_column_index(bim_columns: np.ndarray, candidate_names: tuple) -> int:
    """
    Finds the index within bim_columns of the first of candidate_names that is present.

    :param bim_columns: Decoded contents of the bim_columns dataset
    :param candidate_names: Spellings to look for, in order of preference

    :raises ValueError: If none of the candidate names is present

    :return int: Index of the matching column within bim_values
    """

    for name in candidate_names:
        matches = np.flatnonzero(bim_columns == name)
        if matches.size > 0:
            return int(matches[0])

    raise ValueError(f"None of the expected column names {list(candidate_names)} appear in the "
                     f"imputed parental genotype file's bim_columns, which contains "
                     f"{list(bim_columns)}.  This may indicate a version of snipar whose output "
                     f"format is not yet supported.")


# -------------------------
def get_snipar_chromosome(imputed_filename: str) -> str:
    """
    Reads the chromosome an imputed parental genotype file describes.  Kept separate from
    read_snipar_metadata() because validation needs only this, for every file, before any processing
    begins: it reads a single element rather than whole datasets.

    :param imputed_filename: Full path of the imputed parental genotype file

    :return str: The chromosome recorded in the file
    """

    with h5py.File(imputed_filename, "r") as hdf5_file:
        bim_columns = decode_snipar_bytes(hdf5_file[SNIPAR_BIM_COLUMNS_KEY][:])
        chr_index = get_snipar_bim_column_index(bim_columns, SNIPAR_BIM_CHR_NAMES)

        return str(decode_snipar_bytes(hdf5_file[SNIPAR_BIM_VALUES_KEY][0, chr_index]))


# -------------------------
def read_snipar_metadata(imputed_filename: str) -> dict:
    """
    Reads the datasets needed once, at setup, to align an imputed parental genotype file against
    the samples and SNPs of a fileset.  The two large per-SNP arrays are not read here -- those
    are streamed a chunk at a time during the block loop, by read_snipar_snp_range().

    Byte strings are decoded and pedigree's header row is dropped, so that callers work with
    ordinary strings and with records only.

    :param imputed_filename: Full path of the imputed parental genotype file

    :raises ValueError: If pedigree's header row is not where it is expected

    :return dict: Keyed by the dataset name constants above
    """

    logging.debug(f"\tReading metadata from imputed parental genotype file {imputed_filename}")

    with h5py.File(imputed_filename, "r") as hdf5_file:
        pedigree = decode_snipar_bytes(hdf5_file[SNIPAR_PEDIGREE_KEY][:])

        # Row 0 of pedigree holds column names rather than a family, so it must be dropped before
        # the rest is treated as records.  Checked rather than assumed, since silently losing a
        # real family (or keeping a fake one) would be hard to notice later.
        if tuple(pedigree[0]) != SNIPAR_PED_COLS:
            raise ValueError(f"Expected the first row of the imputed parental genotype file's "
                             f"pedigree to be the header {list(SNIPAR_PED_COLS)}, but found "
                             f"{list(pedigree[0])}.")

        metadata = {
            SNIPAR_PEDIGREE_KEY : pedigree[1:],
            SNIPAR_FAMILIES_KEY : decode_snipar_bytes(hdf5_file[SNIPAR_FAMILIES_KEY][:]),
            SNIPAR_PARENTAL_STATUS_KEY : hdf5_file[SNIPAR_PARENTAL_STATUS_KEY][:],
            SNIPAR_BIM_VALUES_KEY : decode_snipar_bytes(hdf5_file[SNIPAR_BIM_VALUES_KEY][:]),
            SNIPAR_BIM_COLUMNS_KEY : decode_snipar_bytes(hdf5_file[SNIPAR_BIM_COLUMNS_KEY][:]),
            SNIPAR_STANDARD_F_KEY : hdf5_file[SNIPAR_STANDARD_F_KEY][:]
        }

    logging.debug(f"\t\t{len(metadata[SNIPAR_FAMILIES_KEY])} imputed families, "
                  f"{len(metadata[SNIPAR_PEDIGREE_KEY])} pedigree records, "
                  f"{len(metadata[SNIPAR_BIM_VALUES_KEY])} SNPs")

    return metadata


# -------------------------
def get_snipar_chunk_width(imputed_hdf5: h5py.File) -> int:
    """
    Returns the SNP-dimension chunk width of the per-SNP arrays.  This must be read rather than
    assumed: snipar writes with chunks=True, so h5py picks the shape from the dataset dimensions,
    and it varies widely between files.

    :param imputed_hdf5: Open handle to an imputed parental genotype file

    :return int: Number of SNPs spanned by one chunk
    """

    return imputed_hdf5[SNIPAR_IMPUTED_GTS_KEY].chunks[1]


# -------------------------
def read_snipar_snp_range(imputed_hdf5: h5py.File, snp_start: int, snp_stop: int):
    """
    Reads a contiguous SNP range of the two per-SNP arrays.

    Takes an open handle rather than a path because the block loop holds one open across all of
    its iterations, re-reading only when a block crosses out of the cached range.  HDF5 cannot
    serve a partial chunk, so requests much narrower than the chunk width are read and discarded
    repeatedly -- the caller is expected to request chunk-aligned ranges.

    :param imputed_hdf5: Open handle to an imputed parental genotype file
    :param snp_start: First SNP of the range, as an index into the file's own SNP ordering
    :param snp_stop: One past the last SNP of the range

    :return: A tuple (imputed_gts, num_obs_par_alleles), each [families x (snp_stop - snp_start)]
    """

    return (imputed_hdf5[SNIPAR_IMPUTED_GTS_KEY][:, snp_start:snp_stop],
            imputed_hdf5[SNIPAR_NUM_OBS_PAR_ALLELES_KEY][:, snp_start:snp_stop])


# -------------------------
def write_snipar_file(imputed_filename: str, imputed_gts: np.ndarray,
                                num_obs_par_alleles: np.ndarray, standard_f: np.ndarray,
                                parental_status: np.ndarray, pedigree: np.ndarray,
                                families: np.ndarray, bim_values: np.ndarray,
                                bim_columns: tuple = None, chunk_snps: int = None):
    """
    Writes a file in this format.  Exists for building test fixtures, mirroring the write_*
    functions in bedbimfam.py, and reproduces the structure observed in a real file: byte string
    datasets, a header row prepended to pedigree, float16 per-SNP arrays, and families sorted
    lexicographically.

    chunk_snps lets a test force a chunk width narrower than its block size, so that the block
    loop's chunk-crossing path is exercised.  Left unset, h5py chooses, which at fixture sizes
    will not produce a crossing.

    :param imputed_filename: Full path to write to
    :param imputed_gts: [families x SNPs] imputed parental genotypes
    :param num_obs_par_alleles: [families x SNPs] counts, NaN where unavailable
    :param standard_f: [SNPs] per-SNP allele frequencies
    :param parental_status: [families x 3] booleans
    :param pedigree: [individuals x 6] records, WITHOUT a header row (one is added here)
    :param families: [families] family IDs
    :param bim_values: [SNPs x k] variant information
    :param bim_columns: Names of the bim_values columns, defaulting to those of a real file
    :param chunk_snps: SNP-dimension chunk width, or None to let h5py choose
    """

    if bim_columns is None:
        bim_columns = ("Chr", "id", "morgans", "coordinate", "allele1", "allele2")

    chunks = None if chunk_snps is None else (imputed_gts.shape[0], chunk_snps)

    with h5py.File(imputed_filename, "w") as hdf5_file:
        hdf5_file.create_dataset(SNIPAR_IMPUTED_GTS_KEY, data=imputed_gts.astype("float16"),
                                 chunks=chunks)
        hdf5_file.create_dataset(SNIPAR_NUM_OBS_PAR_ALLELES_KEY,
                                 data=num_obs_par_alleles.astype("float16"), chunks=chunks)
        hdf5_file.create_dataset(SNIPAR_STANDARD_F_KEY, data=np.asarray(standard_f, dtype="float64"))
        hdf5_file.create_dataset(SNIPAR_PARENTAL_STATUS_KEY, data=np.asarray(parental_status, dtype=bool))
        hdf5_file.create_dataset(SNIPAR_PEDIGREE_KEY,
                                 data=np.vstack([np.array(SNIPAR_PED_COLS), np.asarray(pedigree)]
                                                ).astype("S"))
        hdf5_file.create_dataset(SNIPAR_FAMILIES_KEY, data=np.sort(np.asarray(families)).astype("S"))
        hdf5_file.create_dataset(SNIPAR_BIM_VALUES_KEY, data=np.asarray(bim_values).astype("S"))
        hdf5_file.create_dataset(SNIPAR_BIM_COLUMNS_KEY, data=np.array(bim_columns).astype("S"))



# -------------------------
def open_snipar_file(imputed_filename: str) -> h5py.File:
    """
    Opens an imputed parental genotype file for reading, as a context manager.

    Exists so that callers needing a handle across many reads -- the block loop holds one open for
    its whole duration -- do not have to import h5py themselves, which would put knowledge of the
    storage format outside this module.

    :param imputed_filename: Full path of the imputed parental genotype file

    :return h5py.File: An open handle, to be used in a with-statement
    """

    return h5py.File(imputed_filename, "r")


# -------------------------
def get_snipar_snp_count(imputed_hdf5: h5py.File) -> int:
    """
    Returns the number of SNPs in the per-SNP arrays.  Used to clamp a read range to the end of
    the file.
    """

    return imputed_hdf5[SNIPAR_IMPUTED_GTS_KEY].shape[1]

#################################
if __name__ == "__main__":

    print("This script is not meant to be called directly.")
