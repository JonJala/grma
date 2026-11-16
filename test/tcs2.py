"""
The test cases for grma2_lib.grma2(), and how to run them.

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

Cases come from two places.  Those of tcs.py's cases that need no imputed parental data are used
as they are; GRMA ignores fam-file parent columns, so the fam-trio cases among them run without a
PO regression.  tcs.py's imputed cases are not used: their imputed values were made up, not derived
from genotypes, so they are not the whole numbers of alleles GRMA's formulas need.  In their place
are cases built for GRMA, whose snipar data follows from simulated haplotypes.

To add a case, write a module defining the inputs above and add a line to REGISTRY.
"""

import os

import numpy as np

import snipar_hdf5
import tcs
import tc2_parental_snipar


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
    module = tc2_parental_snipar
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


SHARED_KEYS = ("SIMPLE", "TWO_FAMILY", "TWO_FAMILY_TRIOS", "VARIED", "PT", "VARIED_OPTIONS",
               "VARIED_ABSENT_PARENTS", "VARIED_ONE_SNP_PER_BLOCK", "PARENTAL_FAM",
               "PARENTAL_FAM_DROP_X", "PARENTAL_FAM_DROP_PARENT")

REGISTRY = {key : None for key in SHARED_KEYS}
REGISTRY.update({
    "PARENTAL_SNIPAR" : _module_case(tc2_parental_snipar),

    # Patrick's check: a genotyped mother hidden from the pedigree, and given to snipar as an
    # imputed parent with every allele known, must give exactly the results of her being genotyped
    # -- wherever snipar has the SNP.  Where it does not, her children have one parent and nothing
    # from snipar, so their residuals are missing; with her genotyped they need nothing from it.
    "PARENTAL_SNIPAR_HIDDEN_MOTHER" : _module_case(
        tc2_parental_snipar, SNIPAR=_hidden_mother_snipar(), EQUIVALENT_TO="PARENTAL_SNIPAR",
        EQUIVALENT_EXCEPT=[f"rs{tc2_parental_snipar.MISSING_FROM_SNIPAR}"]),
})
KEYS = list(REGISTRY)


def get(key):
    """A case's inputs"""
    if REGISTRY[key] is None:
        case = tcs.REGISTRY[key]
        return {**{name : case[name] for name in INPUT_NAMES if name != "SNIPAR"},
                "SNIPAR" : None, "EQUIVALENT_TO" : None, "EQUIVALENT_EXCEPT" : None}
    return REGISTRY[key]


def grma2_kwargs(key, rel_degree, directory, case=None):
    """Keyword arguments for grma2_lib.grma2() that run the case (or a modified copy of it)"""
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
    """Keyword arguments for mock2.mock_grma2() that run the case (or a modified copy of it)"""
    case = case if case is not None else get(key)
    return dict(fam=case["FAM_DF"], king=case["REL_DF"], bim=case["BIM_DF"], genotypes=case["G"],
                rel_degree=rel_degree, pheno=case["PHENO_DF"], covar=case["COVAR_DF"],
                id_list=case["ID_LIST"], snp_list=case["SNP_LIST"], snipar=case["SNIPAR"])


def params(keys=None):
    """(key, rel_degree) for every case at every degree"""
    return [(key, degree) for key in (keys or KEYS) for degree in REL_DEGREES]


def equivalent_keys():
    return [key for key in KEYS if get(key)["EQUIVALENT_TO"] is not None]
