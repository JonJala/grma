#!/usr/bin/env python3

"""
Python tool for genetic association analyses
"""

import argparse as argp
import contextlib
import glob
import hashlib
from io import StringIO
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd

import bedbimfam as bbf
import grma_io
import grma_lib as lib
import snipar_hdf5 as snipar


# Copy-on-Write will become the default behaviour in Pandas 3.0 and is turned on to
# increase clarity about whether objects are views or copies
# (https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#)
pd.options.mode.copy_on_write = True


# Software version
__version__ = '0.1.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "pturley@usc.edu"

# Logging banner to use at the top of the log file
HEADER = f"""
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<>
<> GRMA: Genetic-Relatedness Matched Association
<> Version: {__version__}
<> (C) 2023 Social Science Genetic Association Consortium (SSGAC)
<> MIT License
<>
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<> Software-related correspondence: {SOFTWARE_CORRESPONDENCE_EMAIL}
<> All other correspondence: {OTHER_CORRESPONDENCE_EMAIL}
<><>
"""

# Map of rel degree to numeric value to subset king output
REL_TO_DEG_MAP = {"FS": 0, "1": 1, "2": 2, "3": 3}

# List of inputs to accept as flags to specify degree of relation allowed
REL_DEG_INPUTS = list(REL_TO_DEG_MAP.keys())


# Relatedness degree constants
MIN_RELATEDNESS = 1
MAX_RELATEDNESS = 4 # This is the max degree that King outputs 
DEFAULT_REL_DEG = 1

# The default short file prefix to use for output and logs
DEFAULT_SHORT_PREFIX = "grma"

# Default prefix to use for output when not specified
DEFAULT_FULL_OUT_PREFIX = os.path.join(os.getcwd(), DEFAULT_SHORT_PREFIX)

# Suffix of the per-fileset results file
RES_SUFFIX = ".res"

# Suffix expected on imputed parental genotype files (snipar `impute.py` output)
IMPUTED_PARENTAL_SUFFIX = ".hdf5"

# Dictionary keys for internal usage
OUT_DIR = "Output Directory"
OUT_PREFIX = "Output Prefix"
BED_FILES = "Bed files"
BIM_FILES = "Bim files"
FAM_FILES = "Fam files"
REL_FILE = "Relatedness File"
PHENO_FILE = "Phenotype File"
COVAR_FILE = "Covariate File"
REL_DEG = "Relatedness Degree"
SNPS_PER_BLOCK = "SNPs Per Block"
ID_LIST = "ID List"
SNP_LIST = "SNP List"
IMPUTED_PARENTAL_FILES = "Imputed Parental Files"

# Type declaration
ParserFunc = Callable[[str], argp.ArgumentParser]


# Functions ##################################

#################################
def numpy_err_handler(err: str, flag: int):
    """
    Function that numpy should call when an error occurs.  This is used to ensure that any errors
    are also logged, as opposed to just going to stderr and not being collected in the log

    :param err: String describing the error
    :param flag: An int describing the error (see numpy.seterrcall() docs)
    """
    logging.error(f"Received Numpy error: {err} ({flag})")


#################################
def input_file(s_input: str) -> str:
    """
    Used for parsing some inputs to this program, namely input files.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return str: The input file name
    """

    stripped_filename = s_input.strip()

    if not os.path.exists(stripped_filename):
        raise ValueError(f"Input \"{stripped_filename}\" does not exist.")
    if not os.path.isfile(stripped_filename):
        raise ValueError(f"Input \"{stripped_filename}\" does not appear to be a file.")

    return stripped_filename


#################################
def output_prefix(s_input: str) -> str:
    """
    Used for parsing some inputs to this program, namely full file prefixes used for output.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return str: The prefix
    """

    stripped_p = s_input.strip()

    # Validate existence of output directory (and that no conflicts exist)
    if os.path.exists(stripped_p):
        raise ValueError(f"The designated output prefix \"{stripped_p}\" conflicts with "
                           f"an existing file or directory")

    s_dir = os.path.dirname(stripped_p)
    if s_dir != "" and not os.path.exists(s_dir):
        raise ValueError(f"The designated output directory [{s_dir}] does not exist.")

    return stripped_p

#################################
def rel_thresh_type(s_input: str) -> int:
    """
    Used for parsing some inputs to this program, namely relatedness thresholds.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return int: The relatedness degree threshold
    """

    stripped_thresh = s_input.strip().upper()

    if stripped_thresh in REL_DEG_INPUTS:
        return REL_TO_DEG_MAP[stripped_thresh]

    raise ValueError(f"Invalid input ({stripped_thresh}) for relatedness threshold.")


def to_arg(flag_str: str) -> str:
    """
    Utility method to convert from an argparse flag name to the name of the corresponding attribute
    in the argparse Namespace (which often is adopted elsewhere in this code, as well)

    :param flag_str: Name of the flag (sans "--")

    :return: The name of the argparse attribute/var
    """

    return flag_str.replace("-", "_")

#################################
def bfile_glob_path(s_input: str) -> Set[str]:    
    """ 
    Used for parsing some inputs to this program, namely glob paths (see Python glob module docs).

    :param s_input: String passed in by argparse

    :return: Set of file paths
    """
    files_to_find = f"{s_input}{bbf.BED_SUFFIX}"
    file_path_list = glob.glob(files_to_find)  
    
    if not file_path_list:
        raise ValueError(f"Invalid glob pattern, no files found for: {s_input}")
    
    file_path_prefixes = {os.path.splitext(f)[0] for f in file_path_list}
    return file_path_prefixes

def file_glob_path(s_input: str) -> Set[str]:
    """
    Used for parsing inputs that name whole files by a glob pattern, suffix included (unlike
    bfile_glob_path, which names filesets by their prefix).

    :param s_input: String passed in by argparse

    :return: Set of file paths
    """
    file_path_list = glob.glob(s_input)

    if not file_path_list:
        raise ValueError(f"Invalid glob pattern, no files found for: {s_input}")

    return set(file_path_list)

#################################
def get_grma_parser(progname: str) -> argp.ArgumentParser:
    """
    Return a parser configured for this command line utility

    :param prog: Value to pass to ArgumentParser for prog (should generally be sys.argv[0])

    :return: argparse ArgumentParser
    """

    # Create the initially blank parser
    parser = argp.ArgumentParser(prog=progname)


    # Now, add argument groups and options:

    # Main Input Options
    infile_opt = parser.add_argument_group(title="Input Specifications")
    infile_opt.add_argument("--bfile", type=bfile_glob_path, required=True,
                            metavar="GLOB_PATH", nargs="+",
                            help="Paths to PLINK 1 binary files. See python glob module for "
                                 "documentation on the string(s) to be provided here "
                                 "(full path with support for \"*\", \"?\", and \"[]\").  "
                                 "These strings should be encased in quotes.  Files must "
                                 "be split by chromosome (one and only one chromosome per file)")

    infile_opt.add_argument("--fam", metavar="FILE", type=input_file, required=False,
                            help="Optional input to specify full path and filename of "
                                 "input fam file.  Overrides --bfile flag if specified")

    infile_opt.add_argument("--pheno", metavar="FILE", type=input_file, required=False,
                            help="Optional input to specify a (Plink-style) phenotype file: "
                                 "https://www.cog-genomics.org/plink/1.9/input#pheno")

    infile_opt.add_argument("--covar", metavar="FILE", type=input_file,
                            help="Optional input to specify a (Plink-style) covariates file: "
                                 "https://www.cog-genomics.org/plink/2.0/input#covar")

    infile_opt.add_argument("--rel-pedigree", metavar="FILE", type=input_file, required=True,
                            help=f"Path to KING-formatted relatedness file (ASCII). "
                                 f"Needs the following columns: {list(grma_io.KING_COLS)}")
    infile_opt.add_argument("--imputed-parents", type=file_glob_path, required=False,
                            metavar="GLOB_PATH", nargs="+",
                            help="Optional input to specify imputed parental genotype files "
                                 "(HDF5, as written by snipar's impute.py).  One file per "
                                 "chromosome, matched to the --bfile filesets by the chromosome "
                                 "recorded inside each file.  See python glob module for "
                                 "documentation on the string(s) to be provided here (full path, "
                                 "including the file's suffix, with support for \"*\", \"?\", "
                                 "and \"[]\").  These strings should be encased in quotes.  "
                                 "Parents, and the parent-offspring regression, come only from "
                                 "these files: without them there is no parent-offspring "
                                 "regression, and the fam file's parental ID columns are never "
                                 "consulted.  Parents whose genotypes are OBSERVED (i.e. are part "
                                 "of the sample) have genotypes read from the bed file(s).")

    a_opt = parser.add_argument_group(title="Analysis Options")
    a_opt.add_argument("--rel-thresh", metavar="THRESHOLD",
                       default=DEFAULT_REL_DEG, type=rel_thresh_type,
                       help=f"Relatedness threshold: which regressions to run and report.  "
                            f"FS or 1: parent-offspring (with --imputed-parents) and full "
                            f"siblings; 2: adds 2nd-degree relatives; 3: adds 3rd-degree "
                            f"relatives.  Can be one of {REL_DEG_INPUTS}.  "
                            f"Default is {DEFAULT_REL_DEG}")

    
    infilt_opt = parser.add_argument_group(title="Input Filtering Options")
    infilt_opt.add_argument("--id-list", metavar="FILE", type=input_file,
                            help="Optional input to specify a whitespace-delimited sample ID file "
                                 "of sample FIDs and IIDs to include.  No header allowed.  "
                                 "See Plink flag --keep.")
    infilt_opt.add_argument("--snp-list", metavar="FILE", type=input_file,
                            help="Optional input to specify a whitespace-delimited variant ID file "
                                 "of variant IDs to include.  No header allowed.  "
                                 "See Plink flag --extract.")
    
    # Output Options
    out_opt = parser.add_argument_group(title="Output Specifications")
    out_opt.add_argument("--out", metavar="FILE_PREFIX", type=output_prefix,
                         default=DEFAULT_FULL_OUT_PREFIX,
                         help="Full prefix of output files (logs, sumstats results, etc.).  "
                              f"If not set, [current working directory]/{DEFAULT_SHORT_PREFIX} = "
                              f"\"{DEFAULT_FULL_OUT_PREFIX}\" will be used.  "
                              "Note: The containing directory specified must already exist.")
    

    # General Options
    gen_opt = parser.add_argument_group(title="General Options")
    gen_opt.add_argument("--snps-per-block", metavar="SNPS_PER_BLOCK", type=int,
                         default=lib.DEFAULT_SNPS_PER_BLOCK,
                         help=f"Number of SNPs to process at a time.  Default is "
                              f"{lib.DEFAULT_SNPS_PER_BLOCK}")


    #   Logging options (subgroup)
    log_opt = gen_opt.add_mutually_exclusive_group()
    log_opt.add_argument("--quiet", action="store_true",
                         help="This option will cause the program to limit logging and terminal "
                              "output to warnings and errors and reduce output compared to "
                              "the default/standard logging mode.  It is mutually "
                              "exclusive with the --verbose option.")
    log_opt.add_argument("--verbose", action="store_true",
                         help="This option will greatly increase the logging and terminal output "
                              "of the program compared to the default/standard logging mode.  "
                              "This is useful for debugging and greater visibility into the "
                              "processing that is occurring.  It is mutually exclusive with the "
                              "--quiet option.")

    return parser


#################################
def format_terminal_call(cmd: List[str]) -> str:
    """
    Format commands to/from the terminal for readability

    :param cmd: List of strings much like sys.argv

    :return: Formatted string used for display purposes
    """

    return ' '.join(cmd).replace("--", " \\ \n\t--")


#################################
def get_user_inputs(argv: List[str], parsed_args: argp.Namespace) -> Dict[str, Any]:
    """
    Create dictionary of user-specified options/flags and their values.  Leverages the argparse
    parsing output to glean the actual value, but checks for actual user-set flags in the input

    :param argv: Tokenized list of inputs (meant to be sys.argv in most cases)
    :param parsed_args: Result of argparse parsing the user input

    :return: Dictionary containing user-set args keyed to their values
    """

    # Search for everything beginning with "--" (flag names), strip off the --, take everything
    # before any "=", and convert - to _
    user_set_args = {to_arg(token[2:].split("=")[0]) for token in argv if token.startswith("--")}

    # Since any flag actually specified by the user shouldn't have been replaced by a default
    # value, one can grab the actual value from argparse without having to parse again
    return {user_arg : getattr(parsed_args, user_arg) for user_arg in user_set_args}


#################################
def set_up_logger(log_file: str, log_level: int):
    """
    Set up the logger for this utility.

    :param log_file: Full path to the file used to store logs
    :param log_level: Level used for logging
    """

    log_handlers = []

    # Create the stderr handler
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    log_handlers.append(stderr_handler)

    # Create the stdout handler (if, based on log level, it could possibly have messages to log)
    if log_level <= logging.INFO:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_formatter = logging.Formatter('%(message)s')
        stdout_handler.setFormatter(stdout_formatter)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        log_handlers.append(stdout_handler)

    # Create the file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(file_formatter)
    log_handlers.append(file_handler)

    # Set logging handlers and level for root logger
    logging.basicConfig(handlers=log_handlers, level=log_level, datefmt='%I:%M:%S %p', force=True)


#################################
def setup_func(argv: List[str], get_parser: ParserFunc,
               header: str = HEADER) -> Tuple[argp.Namespace, Dict[str, Any]]:
    """
    Function to handle argument parsing, logging setup, and header printing

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    :param get_parser: Function to call to get argument parser, given a program name

    :return: Tuple of:
               1) Argparse Namespace of parsed arguments
               2) Dictionary of user-specified arguments
    """

    # Parse the input flags using argparse
    parser = get_parser(argv[0])
    parsed_args = parser.parse_args(argv[1:])

    # Break down inputs to keep track of arguments and values specified directly by the user
    user_args = get_user_inputs(argv, parsed_args)

    # Set up the logger
    log_file = parsed_args.out + ".log"
    if parsed_args.quiet:
        log_level = logging.WARNING
    elif parsed_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    set_up_logger(log_file, log_level)

    # Log header and other information
    logging.info(header)
    logging.info(f"See full log at: {os.path.abspath(log_file)}\n\n")
    logging.info(f"Program executed via:\n{format_terminal_call(argv)}\n")

    return parsed_args, user_args



#################################
def results_filename(out_prefix: str, bed_file: str) -> str:
    """
    Derives the results filename for a given bed file.  Single source of truth for the output
    naming rule: validate_inputs() relies on this to detect collisions before any processing
    begins, rather than discovering them after the last fileset has been written.

    :param out_prefix: Full output prefix, as given by --out
    :param bed_file: Full path of the bed file for this fileset

    :return str: Full path of the results file for this fileset
    """

    bfile_basename = os.path.splitext(os.path.basename(bed_file))[0]

    return f"{out_prefix}_{bfile_basename}{RES_SUFFIX}"


#################################
def hash_file(filename: str) -> str:
    """
    Returns a hash of a file's contents.

    :param filename: Full path of the file to hash

    :return str: Hex digest of the file's contents
    """

    with open(filename, "rb") as f:
        return hashlib.blake2b(f.read()).hexdigest()


#################################
def validate_fam_file_agreement(fam_files: List[str]):
    """
    Confirms that every fileset's fam file is identical.  PLINK does not require per-chromosome
    fam files to match, but GRMA's statistics assume a single sample set: if they differ, the
    sample set, R, omega and the standard errors all differ per chromosome and results are not
    comparable between them.  Four of the six fam columns matter -- FID and IID for sample
    keying, IIDF and IIDM for trio determination, and PHENO when no --pheno file is given.

    Comparison is by hash rather than by field, because detection is cheap where reconciliation
    is not: a hash catches membership, parental IDs, phenotypes and ordering at once, without
    having to decide which file's value would win for a conflicting row.  Users who intend a
    single sample set can say so explicitly with --fam.

    :param fam_files: Full paths of the fam files, one per fileset

    :raises ValueError: If the fam files are not all identical
    """

    if len(fam_files) < 2:
        return

    logging.debug("Confirming that the fam file of every fileset is identical, including ordering.\n")

    digests = {}
    for fam_file in fam_files:
        digests.setdefault(hash_file(fam_file), []).append(fam_file)

    if len(digests) > 1:
        groups = "; ".join(f"[{', '.join(sorted(g))}]" for g in digests.values())
        raise ValueError(f"The fam files of the specified filesets are not identical, so the "
                         f"sample set would differ between chromosomes and the results would not "
                         f"be comparable.  Distinct groups found: {groups}.  Note that line "
                         f"ordering is included in this comparison.  Use --fam to specify a "
                         f"single fam file for every fileset.")


#################################
def pair_imputed_parental_files(bim_files: List[str],
                                imputed_parental_files: List[str]) -> List[str]:
    """
    Reorders imputed parental genotype files to correspond to the filesets, pairing them by the
    chromosome each describes.  Position is not usable for this: the --bfile and --imputed-parents
    globs are expanded independently, so nothing guarantees they produce corresponding orders, and
    a mismatch would silently subtract one chromosome's parental genotypes from another's.

    Only the chromosome is read from each file -- the first line of each bim, and a single element
    of each HDF5 -- so this is cheap enough to do before any processing begins.

    :param bim_files: Full paths of the bim files, one per fileset
    :param imputed_parental_files: Full paths of the imputed parental genotype files

    :raises ValueError: If either set of files covers a chromosome more than once, or if the two
                        sets do not describe the same chromosomes

    :return List[str]: The imputed parental genotype files, ordered to match bim_files
    """

    logging.debug("Pairing imputed parental genotype files to filesets by chromosome.\n")

    bim_chromosomes = [bbf.get_chromosome_from_bim_file(bim_file) for bim_file in bim_files]
    imputed_chromosomes = [snipar.get_snipar_chromosome(imputed_file)
                           for imputed_file in imputed_parental_files]

    for description, chromosomes in (("--bfile filesets", bim_chromosomes),
                                     ("imputed parental genotype files", imputed_chromosomes)):
        duplicates = sorted({chrom for chrom in chromosomes if chromosomes.count(chrom) > 1})
        if duplicates:
            raise ValueError(f"More than one of the {description} covers the same chromosome: "
                             f"{duplicates}.  GRMA requires one chromosome per file and one file "
                             f"per chromosome.")

    if set(bim_chromosomes) != set(imputed_chromosomes):
        missing = sorted(set(bim_chromosomes) - set(imputed_chromosomes))
        extra = sorted(set(imputed_chromosomes) - set(bim_chromosomes))
        raise ValueError(f"The chromosomes covered by the imputed parental genotype files do not "
                         f"match those of the --bfile filesets.  Missing imputed parental "
                         f"genotypes for: {missing}.  Imputed parental genotypes supplied for "
                         f"chromosomes with no corresponding fileset: {extra}.")

    chrom_to_imputed_file = dict(zip(imputed_chromosomes, imputed_parental_files))

    return [chrom_to_imputed_file[chrom] for chrom in bim_chromosomes]


#################################
def validate_inputs(pargs: argp.Namespace, user_args: Dict[str, Any]):
    """
    Responsible for coordinating whatever initial validation of inputs can be done
    after argparse has done its part

    :param pargs: Result of argparse parsing user command / flags
    :param user_args: Flags explicitly set by the user along with their values

    :return: Dictionary that contains flags and parameters needed by this program.  It contains
             user-input flags along with defaults set through argparse, and any additional flags
             added as calculations proceed
    """

    # Log user-specified arguments
    logging.debug(f"Program was called with the following arguments: {user_args}\n")

    # Make sure bed/bim/fam files are specified.  Sorted because set.union() returns a set,
    # whose iteration order varies between runs: without this the filesets are processed in an
    # arbitrary order, error messages come out differently each time, and anything that pairs
    # another input to the filesets positionally would be unreliable.
    logging.debug("Checking whether bed/bim/fam files were specified.\n")
    bfiles = sorted(set.union(*pargs.bfile))
    bed_files = [f"{bfile}{bbf.BED_SUFFIX}" for bfile in bfiles]
    bim_files = [f"{bfile}{bbf.BIM_SUFFIX}" for bfile in bfiles]
    fam_files = [pargs.fam] if pargs.fam else [f"{bfile}{bbf.FAM_SUFFIX}" for bfile in bfiles]
    file_list = bed_files + bim_files + fam_files
    
    # Check for missing files
    missing_files = [file for file in file_list if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"Missing the following files: {missing_files}")

    # Check for duplicate bfile basenames (causes collisions in results reporting)
    basenames = [os.path.splitext(os.path.basename(bed_file))[0] for bed_file in bed_files]
    duplicates = sorted({basename for basename in basenames if basenames.count(basename) > 1})
    if duplicates:
        raise ValueError(f"Multiple filesets specified by --bfile share a basename, which would "
                         f"cause their output files to overwrite each other: {duplicates}.  "
                         f"Rename the inputs or run them separately with distinct --out prefixes.")

    # Every fileset's fam file must describe the same sample set.  Only meaningful when the fam
    # files are per-fileset: --fam supplies one for all of them by construction.
    if not pargs.fam:
        validate_fam_file_agreement(fam_files)

    if len(fam_files) == 1:
        fam_files = fam_files * len(bed_files)

    # Imputed parental genotype files, if supplied, must correspond one-to-one with the filesets.
    # Pairing is by the chromosome each file describes rather than by position, because the two
    # globs are independent and nothing guarantees they expand in a corresponding order.
    imputed_parental_files = None
    if pargs.imputed_parents:
        logging.debug("Checking imputed parental genotype files.\n")
        imputed_parental_files = sorted(set.union(*pargs.imputed_parents))

        missing_imputed = [file for file in imputed_parental_files if not os.path.exists(file)]
        if missing_imputed:
            raise FileNotFoundError(f"Missing the following imputed parental genotype files: "
                                    f"{missing_imputed}")

        if len(imputed_parental_files) != len(bed_files):
            raise ValueError(f"The number of imputed parental genotype files "
                             f"({len(imputed_parental_files)}) does not match the number of "
                             f"filesets specified by --bfile ({len(bed_files)}).  Each fileset "
                             f"must have exactly one corresponding file.")

        imputed_parental_files = pair_imputed_parental_files(bim_files, imputed_parental_files)


    # Prepare dictionary that will hold internal values for this program
    logging.debug("Constructing dictionary of values from flags passed in.\n")
    internal_values = {
        OUT_PREFIX : pargs.out,
        OUT_DIR : os.path.dirname(pargs.out),
        BED_FILES : bed_files,
        BIM_FILES : bim_files,
        FAM_FILES : fam_files,
        REL_FILE : pargs.rel_pedigree,
        PHENO_FILE : pargs.pheno,
        COVAR_FILE : pargs.covar,
        REL_DEG : pargs.rel_thresh,
        SNPS_PER_BLOCK : pargs.snps_per_block,
        ID_LIST : pargs.id_list,
        SNP_LIST : pargs.snp_list,
        IMPUTED_PARENTAL_FILES : imputed_parental_files
    }


    return internal_values


#################################
def write_results_to_file(filename: str, results: pd.DataFrame):

    res_start_time = time.time()

    grma_io.write_results(results, filename)
    logging.info(f"Time taken to write results is {time.time() - res_start_time}")

    


#################################
def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args = setup_func(argv, get_grma_parser)

    # Set Numpy error handling to shunt error messages to a logging function
    np.seterr(all='call')
    np.seterrcall(numpy_err_handler)

    # Attempt to print package version info (pandas has a nice version info summary)
    if logging.root.level <= logging.DEBUG:
        logging.debug("Printing Pandas' version summary:")
        with contextlib.redirect_stdout(StringIO()) as f:
            pd.show_versions()
        logging.debug(f"{f.getvalue()}\n")

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        logging.info("Performing additional validation of inputs.\n")
        iargs = validate_inputs(parsed_args, user_args)

        # One imputed parental genotype file per fileset, or None throughout when the flag was
        # not given, so that a single zip() covers both modes
        imputed_parental_files = iargs[IMPUTED_PARENTAL_FILES] or [None] * len(iargs[BED_FILES])

        # Run the GRMA pipeline
        logging.info("Calling main GRMA function\n")
        for bed_file, bim_file, fam_file, imputed_parental_file in zip(
                iargs[BED_FILES], iargs[BIM_FILES], iargs[FAM_FILES], imputed_parental_files):
            results = lib.grma(
                rel_file=iargs[REL_FILE],
                bed_file=bed_file,
                bim_file=bim_file,
                fam_file=fam_file,
                rel_degree=iargs[REL_DEG],
                pheno_file=iargs[PHENO_FILE],
                covar_file=iargs[COVAR_FILE],
                snps_per_block=iargs[SNPS_PER_BLOCK],
                id_list=iargs[ID_LIST],
                snp_list=iargs[SNP_LIST],
                imputed_parental_file=imputed_parental_file
            )

            # Write out the results to disk per chromosome
            logging.info("Writing results to disk.")
            bfile_basename = os.path.splitext(os.path.basename(bed_file))[0]
            filename = f"{iargs[OUT_PREFIX]}_{bfile_basename}.res"
            logging.debug(f"\t{filename}")
            write_results_to_file(filename, results)

        # Log any remaining information
        logging.info("\nExecution complete.\n")

    # Disable pylint error since we do actually want to capture all exceptions here
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(exc)
        sys.exit(1)

#################################
if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)

