import collections
import itertools as it
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

# TODO(jonbjala) Comment code / include function descriptions

# General constants ==============================

_BITS_PER_BYTE = 8

# ================================================


# Constants =========================

# Bed-specific -------

# The first two bytes of a .bed file should be this
_BED_FILE_PREFIX_MAGIC_HEX = '6c1b'
_BED_FILE_PREFIX_MAGIC_BYTEARRAY = bytearray.fromhex(_BED_FILE_PREFIX_MAGIC_HEX)

# The third byte of a .bed file indicating SNP- or individual-major (should be SNP)
_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX = '01'
_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY = bytearray.fromhex(_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX)

# The number of bits used for each sample in a .bed file
_BED_BITS_PER_SAMPLE = 2

# BED file suffix
BED_SUFFIX = ".bed"


# Bim-specific -------

# BIM file suffix
BIM_SUFFIX = ".bim"

BIM_CHR_COL = "CHR"
BIM_RSID_COL = "RSID"
BIM_CM_COL = "CM"
BIM_BP_COL = "BP"
BIM_A1_COL = "A1"
BIM_A2_COL = "A2"

BIM_COLS = (BIM_CHR_COL, BIM_RSID_COL, BIM_CM_COL, BIM_BP_COL, BIM_A1_COL, BIM_A2_COL)


# Fam-specific -------

FAM_FID_COL = "FID"
FAM_IID_COL = "IID"
FAM_IIDF_COL = "IIDF"
FAM_IIDM_COL = "IIDM"
FAM_SEX_COL = "SEX"
FAM_PHENO_COL = "PHENO"
FAM_COLS = (FAM_FID_COL, FAM_IID_COL, FAM_IIDF_COL, FAM_IIDM_COL, FAM_SEX_COL, FAM_PHENO_COL)

# FAM file suffix
FAM_SUFFIX = ".fam"


# ================================================


# Derived constants ==============================

# The number of .bed samples contained in one byte of data
_BED_SAMPLES_PER_BYTE = _BITS_PER_BYTE // _BED_BITS_PER_SAMPLE

# ================================================



# Functions ======================================

# -------------------------
def get_sample_size_from_fam_file(fam_filename: str):
    with open(fam_filename) as f:
        N = sum(1 for line in f)

    return N


# -------------------------
def get_num_snps_from_bim_file(bim_filename: str):
    with open(bim_filename) as f:
        M = sum(1 for line in f)

    return M


# -------------------------
def get_phenotypes_from_fam_file(fam_filename: str):
    fam_df = pd.read_csv(fam_filename, sep=r"\s+", usecols=(5,),
                         names=[FAM_PHENO_COL], index_col=False)

    return fam_df[FAM_PHENO_COL].to_numpy()



# -------------------------
# TODO(jonbjala) Allow for float64?
def read_bed_file(bed_filename: str, N: int, M: int, M_start: int = 0, num_snps: int = -1):

    num_snps = num_snps if num_snps >= 0 else M - M_start
    M_end = M_start + num_snps
    if M_end > M:
        raise ValueError(f"Incompatible M_start ({M_start}) and num_snps ({num_snps}).  "
                         f"This causes M_end ({M_end}) > M ({M}).")


    # Amount of bytes to read in for each SNP
    bed_block_size_in_bytes = math.ceil(N / _BED_SAMPLES_PER_BYTE)

    G = np.zeros((num_snps, N), dtype=np.float32)
    
    with open(bed_filename, 'rb') as bed_file:

        # Read in the first 3 bytes and check against expected .bed file prefix
        initial_bytes = bed_file.read(3)        
        if not(initial_bytes[0:2] == _BED_FILE_PREFIX_MAGIC_BYTEARRAY):
            raise RuntimeError("Error: Initial bytes of bed file [0x%s] are not expected [%s].",
                initial_bytes[0:2].hex(), _BED_FILE_PREFIX_MAGIC_HEX)

        if not(initial_bytes[2] == _BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY[0]):
            raise RuntimeError("Error: BED file not in SNP major order, third byte = %s" % 
                hex(initial_bytes[2]))

        # Read in the actual genetic data and check that it's the expected length
        expected_data_size_in_bytes = num_snps * bed_block_size_in_bytes
        bed_file.seek(M_start * bed_block_size_in_bytes, 1)  # 1 seeks from current file position
        raw_bed_file_contents = bed_file.read(expected_data_size_in_bytes)

        actual_data_size_in_bytes = len(raw_bed_file_contents)
        if actual_data_size_in_bytes != expected_data_size_in_bytes:
            logging.error(f"Section of bed file {bed_filename}, which should have {num_snps} SNPs "
                          f"for {N} individuals, should contain {expected_data_size_in_bytes} "
                          f"bytes of information, but contains {actual_data_size_in_bytes}.")
            raise RuntimeError("Error reading bed file.")


        # Convert the raw data into a matrix of floats (potentially with NaNs)
        for i in range(0, num_snps):
            start_byte_pos = i * bed_block_size_in_bytes
            G[i] = read_bed_file._BED_MAP_ARRAY[list(raw_bed_file_contents[
                start_byte_pos : start_byte_pos + bed_block_size_in_bytes])].ravel()[0:N]

    return G


# See https://www.cog-genomics.org/plink/1.9/formats#bed
read_bed_file._BED_BINARY_TO_VALUE_MAP = {
    '00' : 0.0,
    '01' : np.nan,
    '10' : 1.0,
    '11' : 2.0
}

# A bit complicated, but this is to map a byte's worth of data from a .bed file to a Numpy array
# that contains the genotype information for the entries that comprise the byte
read_bed_file._BED_BYTE_TO_VALARR_MAP = {
    int("%s%s%s%s" % tup, base=2) :
        np.array([read_bed_file._BED_BINARY_TO_VALUE_MAP[element] for element in reversed(tup)],
                 dtype=np.float32) for tup in it.product(
                     read_bed_file._BED_BINARY_TO_VALUE_MAP.keys(),
                     repeat=_BED_SAMPLES_PER_BYTE)
}

read_bed_file._BED_MAP_ARRAY = np.array([read_bed_file._BED_BYTE_TO_VALARR_MAP[x]
                                        for x in sorted(
                                            read_bed_file._BED_BYTE_TO_VALARR_MAP.keys())])



# -------------------------
# TODO(jonbjala) This needs to be written better
def write_bed_file(bed_filename: str, G: np.ndarray):
    # G is shape (M, N)
    M, N = G.shape

    # Amount of bytes to associate each SNP
    bed_block_size_in_bytes = math.ceil(N / _BED_SAMPLES_PER_BYTE)
    pad_amount = bed_block_size_in_bytes * _BED_SAMPLES_PER_BYTE - N

    padded_G = np.pad(np.nan_to_num(G, nan=write_bed_file._NAN_REPLACEMENT),
        pad_width=((0,0), (0, pad_amount)))

    temp_bytes = bytearray(bed_block_size_in_bytes)

    with open(bed_filename, 'wb') as bed_file:
        bed_file.write(_BED_FILE_PREFIX_MAGIC_BYTEARRAY)
        bed_file.write(_BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY)

        for snp in range(0, M):
            count = 0
            for cluster_start in range(0, N, _BED_SAMPLES_PER_BYTE):
                map_key = tuple(padded_G[snp, cluster_start:cluster_start+_BED_SAMPLES_PER_BYTE])
                temp_bytes[count] = write_bed_file._BED_VALARR_TO_BYTE_MAP[map_key]
                count += 1

            bed_file.write(temp_bytes)

# See https://www.cog-genomics.org/plink/1.9/formats#bed
write_bed_file._NAN_REPLACEMENT = -1.0
write_bed_file._BED_VALUE_TO_BINARY_MAP = {
    0.0 : '00',
    write_bed_file._NAN_REPLACEMENT : '01',
    1.0 : '10',
    2.0 : '11'
}
write_bed_file._BED_VALARR_TO_BYTE_MAP = {
    tup : int("%s%s%s%s" % tuple(write_bed_file._BED_VALUE_TO_BINARY_MAP[element] 
        for element in reversed(tup)), base=2)
            for tup in it.product(write_bed_file._BED_VALUE_TO_BINARY_MAP.keys(),
                                  repeat=_BED_SAMPLES_PER_BYTE)}


# -------------------------
def write_fam_file(fam_filename: str, fid: np.ndarray, iid: np.ndarray, iidf: np.ndarray = None,
                   iidm: np.ndarray = None, sex: np.ndarray = None, pheno: np.ndarray = None):
    # TODO(jonbjala) Validate anything?
    N = len(fid)
    fam_dict = {
        FAM_FID_COL : fid,
        FAM_IID_COL : iid,
        FAM_IIDF_COL : iidf if iidf is not None else np.zeros(N),
        FAM_IIDM_COL : iidm if iidm is not None else np.zeros(N),
        FAM_SEX_COL : sex if sex is not None else np.zeros(N),
        FAM_PHENO_COL : pheno if pheno is not None else np.zeros(N)
    }
    fam_df=pd.DataFrame(data=fam_dict, columns=FAM_COLS)

    fam_df.to_csv(fam_filename, sep="\t", header=False, index=False)


# -------------------------
def write_bim_file(bim_filename: str, chrs: np.ndarray, rsid: np.ndarray, bp: np.ndarray,
                   a1: np.ndarray, a2: np.ndarray, cm: np.ndarray = None):
    # TODO(jonbjala) Validate anything?
    M = len(rsid)
    bim_dict = {
        BIM_CHR_COL : chrs,
        BIM_RSID_COL : rsid,
        BIM_CM_COL : cm if cm is not None else np.zeros(M),
        BIM_BP_COL : bp,
        BIM_A1_COL : a1,
        BIM_A2_COL : a2
    }
    bim_df=pd.DataFrame(data=bim_dict, columns=BIM_COLS)

    bim_df.to_csv(bim_filename, sep="\t", header=False, index=False)


#################################
if __name__ == "__main__":

    print("This script is not meant to be called directly.")
