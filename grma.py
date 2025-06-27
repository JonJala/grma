#!/usr/bin/env python3

"""
Python tool for TODO
"""

import argparse as argp
import contextlib
import glob
from io import StringIO
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import glob

import numpy as np
import pandas as pd

from bedbimfam import (BED_SUFFIX, BIM_SUFFIX, FAM_SUFFIX)
import grma_lib as lib


# Copy-on-Write will become the default behaviour in Pandas 3.0 and is turned on to
# increase clarity about whether objects are views or copies
# (https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#)
pd.options.mode.copy_on_write = True


# Software version
__version__ = '0.1.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org" # TODO(jonbjala) Change this?

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

# Relatedness degree constants
MIN_RELATEDNESS = 1
MAX_RELATEDNESS = 4 # This is the max degree that King outputs 
DEFAULT_REL_DEG = 1

# The default short file prefix to use for output and logs
DEFAULT_SHORT_PREFIX = "grma"

# Default prefix to use for output when not specified
DEFAULT_FULL_OUT_PREFIX = os.path.join(os.getcwd(), DEFAULT_SHORT_PREFIX)

# Dictionary keys for internal usage
OUT_DIR = "Output Directory"
OUT_PREFIX = "Output Prefix"
BED_FILES = "Bed files"
BIM_FILES = "Bim files"
FAM_FILES = "Fam files"
REL_FILE = "Relatedness File"
REL_INFO_FILE = "Rel info File"
PHENO_FILE = "Phenotype File"
COVAR_FILE = "Covariate File"
REL_DEG = "Relatedness Degree"
SNPS_PER_BLOCK = "SNPs Per Block"
ID_LIST = "ID List"
SNP_LIST = "SNP List"


# Type declaration
ParserFunc = Callable[[str], argp.ArgumentParser]


# Functions ##################################

#################################
def numpy_err_handler(err: str, flag: bytes):
    """
    Function that numpy should call when an error occurs.  This is used to ensure that any errors
    are also logged, as opposed to just going to stderr and not being collected in the log

    :param err: String describing the error
    :param flag: A byte describing the error (see numpy.seterrcall() docs)
    """
    logging.error("Received Numpy error: %s (%s)", err, flag)


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
    if not os.path.exists(s_dir):
        raise ValueError(f"The designated output directory [{s_dir}] does not exist.")

    return stripped_p

#################################
def rel_thresh_type(s_input: str) -> Union[str, float]:
    """
    Used for parsing some inputs to this program, namely relatedness thresholds.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return float: The relatedness threshold
    """

    stripped_thresh = s_input.strip().upper()

    if stripped_thresh in lib.REL_DEG_INPUTS:
        return stripped_thresh


    # Kinship value currently not supported.  TODO(jonbjala) Implement that in GRMA lib and then
    # remove this exception
    raise ValueError("Specifying kinship value for relatedness threshold not supported yet.")

    float_thresh = float(stripped_thresh)
    if not (lib.MIN_KINSHIP_THRESH <= float_thresh <= lib.MAX_KINSHIP_THRESH):
        raise ValueError(f"Expected threshold value of {lib.REL_DEG_INPUTS} or "
                         f"float between {lib.MIN_KINSHIP_THRESH} and {lib.MAX_KINSHIP_THRESH}")

    return float_thresh


#################################
def to_flag(arg_str: str) -> str:
    """
    Utility method to convert from the name of an argparse Namespace attribute / variable
    (which often is adopted elsewhere in this code, as well) to the corresponding flag

    :param arg_str: Name of the arg

    :return: The name of the flag (sans "--")
    """

    return arg_str.replace("_", "-")


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
    files_to_find = f"{s_input}{BED_SUFFIX}"
    file_path_list = glob.glob(files_to_find)  
    
    if not file_path_list:
        raise ValueError(f"Invalid glob pattern, no files found for: {s_input}")
    
    file_path_prefixes = {os.path.splitext(f)[0] for f in file_path_list}
    return file_path_prefixes

#################################
def get_grma_parser(progname: str) -> argp.ArgumentParser:
    """
    Return a parser configured for this command line utility

    :param prog: Value to pass to ArgumentParser for prog (should generally be sys.argv[0])

    :return: argparse ArgumentParser
    """

    # Create the initally blank parser
    parser = argp.ArgumentParser(prog=progname)


    # Now, add argument groups and options:

    # Main Input Options
    infile_opt = parser.add_argument_group(title="Input Specifications")
    infile_opt.add_argument("--bfile", type=bfile_glob_path, required=True,
                            metavar="GLOB_PATH", nargs="+",
                            help="Paths to PLINK 1 binary files. See python glob module for "
                                 "documentation on the string(s) to be provided here "
                                 "(full path with support for \"*\", \"?\", and \"[]\").  "
                                 "These strings should be encased in quotes.  ")

    infile_opt.add_argument("--fam", metavar="FILE", type=input_file, required=False,
                            help="Optional input to specify full path and filename of "
                                 "input fam file.  Overrides --bfile flag if specified")

    infile_opt.add_argument("--pheno", metavar="FILE", type=input_file, required=False,
                            help="Optional input to specify a (Plink-style) phenotype file: "
                                 "https://www.cog-genomics.org/plink/1.9/input#pheno")

    infile_opt.add_argument("--covar", metavar="FILE", type=input_file,
                            help="Optional input to specify a (Plink-style) covariates file: "
                                 "https://www.cog-genomics.org/plink/2.0/input#covar")

    rel_opt = infile_opt.add_mutually_exclusive_group(required=True)
    rel_opt.add_argument("--rel-grma", metavar="FILE", type=input_file,
                         help="Mutually exclusive with --rel-pedigree.  Required.  "
                              "Path to serialized relatedness information that has been "
                              "pre-processed by GRMA.")
    
    rel_opt.add_argument("--rel-pedigree", metavar="FILE", type=input_file,
                         help=f"Mutually exclusive with --rel-grma.  Required.  "
                              f"Path to KING-formatted relatedness file (ASCII). "
                              f"Needs the following columns: {lib.NEEDED_KING_COLS}")


    a_opt = parser.add_argument_group(title="Analysis Options")
    a_opt.add_argument("--rel-thresh", metavar="THRESHOLD",
                       default=DEFAULT_REL_DEG, type=rel_thresh_type,
                       help=f"Relatedness threshold (required with --rel-pedigree). Ignored if "
                            f"--rel-grma is specified. Can be one of {lib.REL_DEG_INPUTS} or a "
                            f"number in [{lib.MIN_KINSHIP_THRESH}, {lib.MAX_KINSHIP_THRESH}].")

    
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
def get_user_inputs(argv: List[str], parsed_args: argp.Namespace) -> str:
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
        log_level = logging.WARN
    elif parsed_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    set_up_logger(log_file, log_level)

    # Log header and other information
    logging.info(header)
    logging.info("See full log at: %s\n", os.path.abspath(log_file))
    logging.info("\nProgram executed via:\n%s\n", format_terminal_call(argv))

    return parsed_args, user_args


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
    logging.debug("\nProgram was called with the following arguments:\n%s", user_args)

    # Make sure bed/bim/fam files are specified
    logging.debug("Checking whether bed/bim/fam files were specified.")
    bfiles = set.union(*pargs.bfile)
    bed_files = [f"{bfile}{BED_SUFFIX}" for bfile in bfiles]
    bim_files = [f"{bfile}{BIM_SUFFIX}" for bfile in bfiles]
    fam_files = [pargs.fam] if pargs.fam else [f"{bfile}{FAM_SUFFIX}" for bfile in bfiles]
    
    missing_files = [file for file in (bed_files + bim_files + fam_files)
                     if not os.path.exists(file)]
    
    if missing_files:
        raise FileNotFoundError(f"Missing the following files: {missing_files}")

    # Prepare dictionary that will hold internal values for this program
    logging.debug("Constructing dictionary of values from flags passed in.")
    internal_values = {
        OUT_PREFIX : pargs.out,
        OUT_DIR : os.path.dirname(pargs.out),
        BED_FILES : bed_files,
        BIM_FILES : bim_files,
        FAM_FILES : [pargs.fam]*len(bfiles) if pargs.fam else [f"{bfile}{FAM_SUFFIX}"
                                                               for bfile in bfiles],
        REL_FILE : pargs.rel_pedigree,
        REL_INFO_FILE : pargs.rel_grma,
        PHENO_FILE : pargs.pheno,
        COVAR_FILE : pargs.covar,
        REL_DEG : pargs.rel_thresh,
        SNPS_PER_BLOCK : pargs.snps_per_block,
        ID_LIST : pargs.id_list,
        SNP_LIST : pargs.snp_list
    }


    return internal_values


#################################
def write_results_to_file(filename: str, results: pd.DataFrame):

    res_start_time = time.time()

    results.to_csv(filename, index = False, header=True, sep='\t')
    logging.info(f"Time taken to write results is {time.time() - res_start_time}")
    return results
    


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
        logging.debug("%s\n", f.getvalue())

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        logging.info("Performing additional validation of inputs.")
        iargs = validate_inputs(parsed_args, user_args)

        # Run the GRMA pipeline
        logging.info("Calling main GRMA function")
        for bed_file, bim_file, fam_file in zip(iargs[BED_FILES], iargs[BIM_FILES], iargs[FAM_FILES]):
            results = lib.grma(
                rel_input=iargs[REL_FILE], rel_info_file=iargs[REL_INFO_FILE],
                bed_file=bed_file, bim_file=bim_file,
                fam_file=fam_file, pheno_file=iargs[PHENO_FILE], covar_file=iargs[COVAR_FILE],
                rel_degree=iargs[REL_DEG], snps_per_block=iargs[SNPS_PER_BLOCK],
                id_list=iargs[ID_LIST], snp_list=iargs[SNP_LIST]
            )

            # Write out the results to disk per chromosome
            logging.info("Writing results to disk.")
            bfile_basename = os.path.splitext(os.path.basename(bed_file))[0]
            filename = f"{iargs[OUT_PREFIX]}_{bfile_basename}.res"
            logging.debug(f"\t{filename}")
            write_results_to_file(filename, results)

        # Log any remaining information TODO(jonbjala) Timing info?
        logging.info("\nExecution complete.\n")

    # Disable pylint error since we do actually want to capture all exceptions here
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(exc)
        sys.exit(1)

#################################
if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)

