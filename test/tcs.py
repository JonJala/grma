"""
The test cases for grma_lib.grma(), and how to run them.

Each case is a dict of inputs:

    FAM_DF, REL_DF, BIM_DF, G        required: the fam, KING and bim files, and the genotypes,
                                     (bim snps, fam samples), NaN where missing
    PHENO_DF, COVAR_DF, ID_LIST,     optional: None where not given
    SNP_LIST, SNPS_PER_BLOCK
    SNIPAR                           optional: snipar's data, as keyword arguments for
                                     snipar_hdf5.write_snipar_file(); present means a PO regression
    EQUIVALENT_TO                    optional: the key of a case this one must give exactly the same
                                     results as
    EQUIVALENT_EXCEPT                optional: rsids where the two need not agree, with the reason
                                     given where the case is defined

Each case's inputs are defined in its own module (tc_*.py), described in that module's docstring.
Variants that differ only in an option are defined here.

To add a case, write a module defining the inputs above and add a line to REGISTRY.  It is then run
against grma_mock's independent implementation and through every invariance test.
"""

import os

import numpy as np

import pandas as pd

import snipar_hdf5
import tc_parental_snipar
import tc_patrick_v1
import tc_simple
import tc_three_generations
import tc_two_family
import tc_varied
import tc_varied_absent_parents
import tc_varied_options


INPUT_NAMES = ("FAM_DF", "REL_DF", "BIM_DF", "G", "PHENO_DF", "COVAR_DF", "ID_LIST", "SNP_LIST",
               "SNPS_PER_BLOCK", "SNIPAR")
META_NAMES = ("EQUIVALENT_TO", "EQUIVALENT_EXCEPT")

REL_DEGREES = (0, 1, 2, 3)


def _module_case(module, **overrides):
    case = {name : getattr(module, name, None) for name in INPUT_NAMES + META_NAMES}
    case.update(overrides)
    return case


def _hidden_mother_snipar():
    """PARENTAL_SNIPAR's snipar data, with family A's mother hidden from the pedigree and given to
    snipar as an imputed parent whose every allele is known: her true genotypes, k = 4"""
    module = tc_parental_snipar
    snipar = {k : np.array(v, copy=True) for k, v in module.SNIPAR.items()}
    pedigree = snipar["pedigree"]
    in_a = pedigree[:, 0] == "A"
    pedigree[in_a, 3] = "hidden_mum_a"
    pedigree[in_a, 5] = "False"
    mum_a = module.G[:, module.IIDS.index("mum_a")]
    snipar_snps = [s for s in range(module.NUM_SNPS) if s != module.MISSING_FROM_SNIPAR]
    row = mum_a[snipar_snps].copy()
    swapped = snipar_snps.index(module.SWAPPED_IN_SNIPAR)
    row[swapped] = 2.0 - row[swapped]                   # Counted in the file's other allele
    snipar["imputed_gts"] = np.vstack([row, snipar["imputed_gts"]])
    snipar["num_obs_par_alleles"] = np.vstack([np.full(len(snipar_snps), 4.0),
                                               snipar["num_obs_par_alleles"]])
    snipar["parental_status"] = np.zeros((4, 3), dtype=bool)
    snipar["families"] = np.array(["A", "B", "C", "D"])
    return snipar


def _without(module, iid):
    """An ID list of everyone in the module's fam file but one sample"""
    fam = module.FAM_DF
    keep = fam["IID"].astype(str) != iid
    return pd.DataFrame({"FID" : fam["FID"][keep], "IID" : fam["IID"][keep]}).reset_index(
        drop=True)


REGISTRY = {
    "SIMPLE" : _module_case(tc_simple),
    "TWO_FAMILY" : _module_case(tc_two_family),
    "VARIED" : _module_case(tc_varied),
    "VARIED_OPTIONS" : _module_case(tc_varied_options),

    # GRMA takes parents only from snipar's pedigree, so parent columns in the fam file -- here
    # naming people who are not in it -- must change nothing
    "VARIED_ABSENT_PARENTS" : _module_case(tc_varied_absent_parents, EQUIVALENT_TO="VARIED"),

    "PATRICK_V1" : _module_case(tc_patrick_v1),
    "THREE_GENERATIONS" : _module_case(tc_three_generations),

    # The unrelated sample listed first, then a parent, left out of the analysis by an ID list
    "THREE_GENERATIONS_DROP_X" : _module_case(tc_three_generations,
                                              ID_LIST=_without(tc_three_generations, "X")),
    "THREE_GENERATIONS_DROP_PARENT" : _module_case(tc_three_generations,
                                                   ID_LIST=_without(tc_three_generations, "mo1")),

    "PARENTAL_SNIPAR" : _module_case(tc_parental_snipar),

    # Patrick's check: a genotyped mother hidden from the pedigree, and given to snipar as an
    # imputed parent with every allele known, must give exactly the results of her being genotyped
    # -- wherever snipar has the SNP.  Where it does not, her children have one parent and nothing
    # from snipar, so their residuals are missing; with her genotyped they need nothing from it.
    "PARENTAL_SNIPAR_HIDDEN_MOTHER" : _module_case(
        tc_parental_snipar, SNIPAR=_hidden_mother_snipar(), EQUIVALENT_TO="PARENTAL_SNIPAR",
        EQUIVALENT_EXCEPT=[f"rs{tc_parental_snipar.MISSING_FROM_SNIPAR}"]),
}
KEYS = list(REGISTRY)


def get(key):
    """A case's inputs"""
    return REGISTRY[key]


def grma_kwargs(key, rel_degree, directory, case=None):
    """Keyword arguments for grma_lib.grma() that run the case (or a modified copy of it)"""
    case = case if case is not None else get(key)
    kwargs = dict(rel_file=case["REL_DF"], bed_file=case["G"], bim_file=case["BIM_DF"],
                  fam_file=case["FAM_DF"], rel_degree=rel_degree, pheno_file=case["PHENO_DF"],
                  covar_file=case["COVAR_DF"], id_list=case["ID_LIST"],
                  snp_list=case["SNP_LIST"], imputed_parental_file=None)
    if case["SNIPAR"] is not None:
        path = os.path.join(str(directory), f"{key}_{id(case)}" + snipar_hdf5.SNIPAR_FILE_SUFFIX)
        if not os.path.exists(path):
            snipar_hdf5.write_snipar_file(path, **case["SNIPAR"])
        kwargs["imputed_parental_file"] = path
    if case["SNPS_PER_BLOCK"] is not None:
        kwargs["snps_per_block"] = case["SNPS_PER_BLOCK"]
    return kwargs


def mock_kwargs(key, rel_degree, case=None):
    """Keyword arguments for grma_mock.mock_grma() that run the case (or a modified copy of it)"""
    case = case if case is not None else get(key)
    return dict(fam=case["FAM_DF"], king=case["REL_DF"], bim=case["BIM_DF"], genotypes=case["G"],
                rel_degree=rel_degree, pheno=case["PHENO_DF"], covar=case["COVAR_DF"],
                id_list=case["ID_LIST"], snp_list=case["SNP_LIST"], snipar=case["SNIPAR"])


def params(keys=None):
    """(key, rel_degree) for every case at every degree"""
    return [(key, degree) for key in (keys or KEYS) for degree in REL_DEGREES]


def equivalent_keys():
    return [key for key in KEYS if get(key)["EQUIVALENT_TO"] is not None]
