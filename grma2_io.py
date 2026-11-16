"""
Reading GRMA's inputs, including genotypes and snipar's imputed parental genotypes a block of SNPs
at a time; and building and writing its output.

One reader per input file.  Each accepts a path or an already-read DataFrame (the tests pass
DataFrames), and returns a DataFrame in one normalized form:

    - every ID column is TEXT.  Read as numbers, an ID like 007 would become 7 and stop matching
      the same person in another file, so IDs are never converted.
    - every value column has the numeric type it should, and a value that is present but cannot
      be read as a number is an error naming the rows affected, rather than silently missing.

Each file's layout is stated once, as a tuple of column names, and positions and counts are derived
from it.  The fam and bim layouts come from bedbimfam; the KING layout is defined here.  This module
only reads: filtering samples and checking one input against another are planning steps, done in
grma2_lib.
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

import bedbimfam as bbf
import grma2_util
import snipar_hdf5


# KING's file format ==============================================================================

KING_FID1_COL = "FID1"
KING_IID1_COL = "ID1"
KING_FID2_COL = "FID2"
KING_IID2_COL = "ID2"
KING_REL_COL = "InfType"

KING_ID_COLS = (KING_FID1_COL, KING_IID1_COL, KING_FID2_COL, KING_IID2_COL)

# The columns GRMA reads.  The kinship coefficient is not used: relationships come from InfType.
KING_COLS = KING_ID_COLS + (KING_REL_COL,)

# KING's inferred relationship types, as they appear in the InfType column
INF_DUP_MZTWIN = "Dup/MZTwin"
INF_PARENT_OFFSPRING = "PO"
INF_FULLSIB = "FS"
INF_2ND = "2nd"
INF_3RD = "3rd"
INF_4TH = "4th"
INF_UNRELATED = "UN"


# Other input formats =============================================================================

# Whitespace-separated, with no header row
SNP_LIST_COLS = (bbf.BIM_RSID_COL,)  # Any further columns are ignored
PHENO_FILE_ID_COLS = (bbf.FAM_FID_COL, bbf.FAM_IID_COL)
PHENO_FILE_COLS = PHENO_FILE_ID_COLS + (bbf.FAM_PHENO_COL,)
ID_LIST_COLS = (bbf.FAM_FID_COL, bbf.FAM_IID_COL)  # Any further columns are ignored

# A covariate file is its ID columns followed by one or more covariates, named here in order
COVAR_FILE_ID_COLS = (bbf.FAM_FID_COL, bbf.FAM_IID_COL)
COVAR_PREFIX = "COVAR_"

# How a missing value is written in the output: what R, pandas and most tools read as missing
OUTPUT_MISSING = "NA"

# The output's columns for each reported estimate, followed by the estimate's label: the
# regressions it combines, joined by underscores (PO_FS_2ND, for instance)
OUTPUT_BETA_PREFIX = "BETA_"
OUTPUT_SE_PREFIX = "SE_"
OUTPUT_P_PREFIX = "P_"

LN_2 = np.log(2.0)
RECIP_LN_10 = 1.0 / np.log(10.0)

# Text that means "no value" when it appears in a numeric column
MISSING_TEXT = ("", "NA", "NaN", "nan")


# Shared helpers ==================================================================================

def _read_table(source, name: str, read_params: dict) -> pd.DataFrame:
    """A path read with pandas, or a copy of a DataFrame passed in directly

    Args:
        source: Path to the file, or the DataFrame itself
        name: What the input is, for error messages
        read_params: Keyword arguments for pandas.read_csv, used only when source is a path.
            Whitespace-separated with no header unless they say otherwise.

    Returns:
        A DataFrame the caller may modify freely
    """

    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, str) and source:
        params = {"sep" : r"\s+", "header" : None, "index_col" : False}
        params.update(read_params)
        return pd.read_csv(source, **params)
    raise TypeError(f"Expected a path or a pandas DataFrame for the {name}, "
                    f"but received {type(source).__name__}")


def _ids_as_text(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Converts ID columns to text in place, returning df

    A DataFrame passed in directly may hold numeric IDs; a file is read with these columns as text
    already, so that leading zeros survive.  Either way they end up as strings.
    """

    for column in columns:
        df[column] = df[column].astype(str)
    return df


def _numeric_column(df: pd.DataFrame, column: str, name: str, row_ids: pd.Series) -> pd.Series:
    """A column as floats, blanks and NA as NaN, raising if any value is not a number

    Args:
        df: The table holding the column
        column: The column to convert
        name: What the input is, for the error message
        row_ids: How each row is identified in an error message, (rows,)

    Raises:
        ValueError: Naming the rows whose value is present but not a number

    Returns:
        The converted column, (rows,)
    """

    raw = df[column]
    # A file's blanks and NA are already NaN when pandas reads them, but a DataFrame passed in may
    # still hold them as text, so recognize them either way
    blank = raw.isna() | raw.astype(str).str.strip().isin(MISSING_TEXT)
    values = pd.to_numeric(raw.mask(blank), errors="coerce")
    unreadable = values.isna() & ~blank
    if unreadable.any():
        raise ValueError(f"The {name} has values in column {column} that are not numbers, for "
                         f"{grma2_util.describe_offenders(row_ids[unreadable].tolist())}")
    return values.astype(float)


def _phenotype_column(df: pd.DataFrame, name: str) -> pd.Series:
    """The phenotype column as floats, with the fam format's missing phenotype code as NaN"""

    row_ids = df[bbf.FAM_FID_COL] + " " + df[bbf.FAM_IID_COL]
    values = _numeric_column(df, bbf.FAM_PHENO_COL, name, row_ids)
    return values.mask(values == bbf.FAM_MISSING_PHENOTYPE)


def _first_columns(columns) -> dict:
    """read_csv parameters reading just the leading columns of a file, named as given"""

    return {"usecols" : list(range(len(columns))), "names" : list(columns)}


# Readers =========================================================================================

def read_fam(source) -> pd.DataFrame:
    """A fam file

    Returns:
        bedbimfam.FAM_COLS, in file order: FID, IID, father, mother and sex as text; the phenotype
        as a float, NaN where blank, NA or the missing code
    """

    text_cols = (bbf.FAM_FID_COL, bbf.FAM_IID_COL, bbf.FAM_IIDF_COL, bbf.FAM_IIDM_COL,
                 bbf.FAM_SEX_COL)
    df = _read_table(source, "fam file",
                     {"names" : list(bbf.FAM_COLS), "dtype" : {c : str for c in text_cols}})
    df = _ids_as_text(df[list(bbf.FAM_COLS)].copy(), text_cols)
    df[bbf.FAM_PHENO_COL] = _phenotype_column(df, "fam file")
    return df.reset_index(drop=True)


def read_pheno(source) -> pd.DataFrame:
    """A phenotype file: PHENO_FILE_COLS, with no header

    Returns:
        FID and IID as text; the phenotype as a float, NaN where blank, NA or the missing code
    """

    df = _read_table(source, "phenotype file",
                     {**_first_columns(PHENO_FILE_COLS),
                      "dtype" : {c : str for c in PHENO_FILE_ID_COLS}})
    df = df.iloc[:, :len(PHENO_FILE_COLS)].set_axis(list(PHENO_FILE_COLS), axis=1)
    df = _ids_as_text(df, PHENO_FILE_ID_COLS)
    df[bbf.FAM_PHENO_COL] = _phenotype_column(df, "phenotype file")
    return df.reset_index(drop=True)


def read_covar(source) -> pd.DataFrame:
    """A covariate file: COVAR_FILE_ID_COLS then at least one covariate, with no header

    A missing covariate is kept as NaN here; the sample table decides what that means.

    Raises:
        ValueError: If there is no covariate column, or a covariate is not a number

    Returns:
        FID and IID as text, then COVAR_1, COVAR_2, ... as floats
    """

    num_id_cols = len(COVAR_FILE_ID_COLS)
    df = _read_table(source, "covariate file", {"dtype" : {n : str for n in range(num_id_cols)}})
    num_covars = df.shape[1] - num_id_cols
    if num_covars < 1:
        raise ValueError(f"The covariate file must have at least one covariate column after FID "
                         f"and IID, but has {df.shape[1]} column(s) in all")

    covar_cols = [f"{COVAR_PREFIX}{n + 1}" for n in range(num_covars)]
    df = df.set_axis(list(COVAR_FILE_ID_COLS) + covar_cols, axis=1)
    df = _ids_as_text(df, COVAR_FILE_ID_COLS)
    row_ids = df[bbf.FAM_FID_COL] + " " + df[bbf.FAM_IID_COL]
    for column in covar_cols:
        df[column] = _numeric_column(df, column, "covariate file", row_ids)
    return df.reset_index(drop=True)


def read_id_list(source) -> pd.DataFrame:
    """An ID list: ID_LIST_COLS, anything after them ignored

    Returns:
        FID and IID as text
    """

    df = _read_table(source, "ID list", {**_first_columns(ID_LIST_COLS), "dtype" : str})
    df = df.iloc[:, :len(ID_LIST_COLS)].set_axis(list(ID_LIST_COLS), axis=1)
    return _ids_as_text(df, ID_LIST_COLS).reset_index(drop=True)


def read_king(source) -> pd.DataFrame:
    """A KING relatedness file, with its header row

    Raises:
        ValueError: If any column GRMA needs is absent

    Returns:
        KING_COLS: the four ID columns as text, and InfType as text
    """

    df = _read_table(source, "KING file",
                     {"header" : 0, "dtype" : {c : str for c in KING_ID_COLS}})
    missing = [c for c in KING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"The KING file has no column(s) {missing}; it must have "
                         f"{list(KING_COLS)}")
    df = _ids_as_text(df[list(KING_COLS)].copy(), KING_COLS)
    return df.reset_index(drop=True)


def read_bim(source) -> pd.DataFrame:
    """A bim file

    Returns:
        bedbimfam.BIM_COLS, in file order: chromosome, rsid and both alleles as text; the genetic
        distance as a float; the base-pair position as an integer
    """

    text_cols = (bbf.BIM_CHR_COL, bbf.BIM_RSID_COL, bbf.BIM_A1_COL, bbf.BIM_A2_COL)
    df = _read_table(source, "bim file",
                     {"names" : list(bbf.BIM_COLS), "dtype" : {c : str for c in text_cols}})
    df = _ids_as_text(df[list(bbf.BIM_COLS)].copy(), text_cols)
    df[bbf.BIM_CM_COL] = df[bbf.BIM_CM_COL].astype(float)
    df[bbf.BIM_BP_COL] = df[bbf.BIM_BP_COL].astype("int64")
    return df.reset_index(drop=True)



def read_snp_list(source) -> pd.DataFrame:
    """A SNP list: one rsid per line, anything after it ignored

    Returns:
        SNP_LIST_COLS, the rsid as text
    """

    df = _read_table(source, "SNP list", {**_first_columns(SNP_LIST_COLS), "dtype" : str})
    df = df.iloc[:, :len(SNP_LIST_COLS)].set_axis(list(SNP_LIST_COLS), axis=1)
    return _ids_as_text(df, SNP_LIST_COLS).reset_index(drop=True)


# snipar's imputed parental genotypes =============================================================

class SniparReader:
    """Reads snipar's imputed parental genotypes a block of SNPs at a time

    The file is chunked by SNP, and blocks arrive in increasing SNP order, so the reader keeps one
    chunk in memory and moves forward through the file: each chunk is read once, however many
    blocks it serves.  That relies on the SNPs' positions in the file never decreasing from one to
    the next, which planning guarantees (it rejects a file whose SNPs are in a different order
    from the bim file's).

    Use as a context manager, so that the file is closed:

        with SniparReader(path, standard_f) as reader:
            imputed, counts, f = reader.read(positions, flip)
    """

    def __init__(self, path: str, standard_f: np.ndarray):
        """
        Args:
            path: The imputed parental genotype file
            standard_f: Its allele frequency for every SNP in it, (file snps,), as
                snipar_hdf5.read_snipar_metadata() returns it
        """

        self._file = snipar_hdf5.open_snipar_file(path)
        self._chunk_width = snipar_hdf5.get_snipar_chunk_width(self._file)
        self._snp_count = snipar_hdf5.get_snipar_snp_count(self._file)
        self._standard_f = np.asarray(standard_f, dtype=float)
        self._cached_start = self._cached_stop = -1
        self._cached_imputed = self._cached_counts = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()

    def read(self, positions: np.ndarray, flip: np.ndarray):
        """One block of SNPs' imputed data, for every family in the file

        Args:
            positions: Each SNP's position in the file, (snps,), -1 for a SNP the file does not
                have.  The others must never decrease from one to the next.
            flip: Whether each SNP's alleles are swapped relative to the bim file's, (snps,).  A
                flipped SNP's imputed values become 2 - value and its frequency 1 - f, so that
                everything is counted in the bim file's allele.

        Returns:
            A tuple (imputed, counts, f): the imputed genotypes and the numbers of observed
            parental alleles, each (snps, families); and the allele frequencies, (snps,).  NaN
            for every SNP the file does not have.
        """

        positions = np.asarray(positions)
        present = np.flatnonzero(positions != -1)
        num_families = None
        columns = {}
        filled = 0
        while filled < len(present):
            position = int(positions[present[filled]])
            if self._cached_start == -1 or position >= self._cached_stop:
                start = (position // self._chunk_width) * self._chunk_width
                stop = min(start + self._chunk_width, self._snp_count)
                self._cached_imputed, self._cached_counts = snipar_hdf5.read_snipar_snp_range(
                    self._file, start, stop)
                self._cached_start, self._cached_stop = start, stop
                logging.debug(f"\t\tRead imputed parental SNPs [{start}, {stop})")
            num_families = self._cached_imputed.shape[0]
            covered = filled + int(np.searchsorted(positions[present[filled:]], self._cached_stop))
            for k in present[filled:covered]:
                offset = positions[k] - self._cached_start
                columns[k] = (self._cached_imputed[:, offset], self._cached_counts[:, offset])
            filled = covered

        if num_families is None:
            num_families = snipar_hdf5.read_snipar_snp_range(self._file, 0, 0)[0].shape[0]
        imputed = np.full((len(positions), num_families), np.nan)
        counts = np.full((len(positions), num_families), np.nan)
        for k, (values, number) in columns.items():
            imputed[k] = values
            counts[k] = number
        f = np.full(len(positions), np.nan)
        f[present] = self._standard_f[positions[present]]

        flip = np.asarray(flip, dtype=bool) & (positions != -1)
        imputed[flip] = 2.0 - imputed[flip]
        f[flip] = 1.0 - f[flip]
        return imputed, counts, f



# Genotypes =======================================================================================

def read_genotype_block(bed_source, num_file_snps: int, num_file_samples: int,
                        bim_rows: np.ndarray, bed_cols: np.ndarray) -> np.ndarray:
    """One block of SNPs' genotypes, for the samples the plan reads

    Reads the range of the bed file the block spans, then keeps only the block's SNPs and the
    plan's samples, so that memory holds only what the regressions use.

    Args:
        bed_source: Path to the bed file, or its genotypes as an array, (file snps, file samples)
        num_file_snps, num_file_samples: The bed file's dimensions (its bim and fam files' lengths)
        bim_rows: The block's SNPs' rows in the bim file, (snps,), increasing
        bed_cols: The samples to keep, as bed file columns, (loaded samples,)

    Returns:
        Genotypes, (snps, loaded samples), NaN where missing
    """

    if isinstance(bed_source, np.ndarray):
        return bed_source[np.asarray(bim_rows)][:, bed_cols].astype(float)
    first, last = int(bim_rows[0]), int(bim_rows[-1])
    genotypes = bbf.read_bed_file(bed_source, M=num_file_snps, N=num_file_samples, M_start=first,
                                  num_snps=last - first + 1)
    return genotypes[np.asarray(bim_rows) - first][:, bed_cols]


# Output ==========================================================================================

def calculate_pvals(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """Two-sided p-values for each estimate, as text, accurate however small

    P = 2 Phi(-|z|) is computed in log space and written as a mantissa and an exponent, so a
    p-value far below the smallest representable float (about 1e-308) is still written correctly
    rather than as 0.  Copied from the original implementation, except that a missing estimate
    (NaN) now has no p-value, None, so that it is written as missing rather than as the text
    "nan".

    Args:
        betas: The estimates, (n,)
        ses: Their standard errors, (n,)

    Returns:
        The p-values, (n,), as text in the form "1.23e-45", None where there is none
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = betas / ses
    # Since P = 2 * normal_cdf(-|Z|), P = e ^ (log_normal_cdf(-|Z|) + ln 2)
    # This can be changed to base 10 as P = 10 ^ ((log_normal_cdf(-|Z|) + ln 2) / ln 10)
    log_10_p = RECIP_LN_10 * (stats.norm.logcdf(-np.abs(z_scores)) + LN_2)
    # Break up the log based 10 of P values into the integer and fractional part
    # To handle the case of Z = 0 (and not result in "10e-1"), set initial values to (-1.0, 1.0)
    frac_part, int_part = np.full_like(z_scores, -1.0), np.full_like(z_scores, 1.0)
    np.modf(log_10_p, out=(frac_part, int_part), where=(z_scores != 0.0))
    usable = np.isfinite(z_scores)
    frac_part = np.where(usable, frac_part, 0.0)
    int_part = np.where(usable, int_part, 1.0)
    # Construct strings for the P values
    # 1) Add one to the fractional part to ensure that the result mantissa is between 1 and 10
    # 2) Subtract one from the integer part to compensate and keep the overall value correct
    result = np.char.add(np.char.add(np.power(10.0, (frac_part + 1.0)).astype(str), 'e'),
                         (int_part - 1).astype(int).astype(str))
    return np.where(usable, result.astype(object), None)


def estimate_label(regressions: tuple) -> str:
    """An estimate's label: the regressions it combines, joined by underscores"""

    return "_".join(regressions)


def build_output(bim: pd.DataFrame, estimates: list, betas: np.ndarray,
                 ses: np.ndarray) -> pd.DataFrame:
    """GRMA's results: each analysed SNP's bim columns, then each estimate's beta, SE and p-value

    Args:
        bim: The analysed SNPs' bim rows, as read_bim() returns them
        estimates: The reported estimates, each the regressions it combines
        betas, ses: The estimates and their standard errors, each (snps, estimates); each beta is
            for the A2 allele, as in the original implementation

    Returns:
        bedbimfam.BIM_COLS, then BETA_<label>, SE_<label>, P_<label> for each estimate, with
        missing values missing (pandas holds them as NaN, whatever their column)
    """

    output = bim[list(bbf.BIM_COLS)].reset_index(drop=True)
    columns = {}
    for e, combined in enumerate(estimates):
        label = estimate_label(combined)
        columns[OUTPUT_BETA_PREFIX + label] = betas[:, e]
        columns[OUTPUT_SE_PREFIX + label] = ses[:, e]
        columns[OUTPUT_P_PREFIX + label] = calculate_pvals(betas[:, e], ses[:, e])
    return pd.concat([output, pd.DataFrame(columns)], axis=1)


def write_results(results: pd.DataFrame, path: str):
    """Writes GRMA's results: tab-separated, with a header, missing values as OUTPUT_MISSING"""

    results.to_csv(path, sep="\t", index=False, header=True, na_rep=OUTPUT_MISSING)


#################################
if __name__ == "__main__":

    print("This script is not meant to be called directly.")
