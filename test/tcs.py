import os
import sys

import numpy as np
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma_lib as lib

import tc_simple
import tc_2fam
import tc_varied
import tc_pt



NAME = 'TC_NAME'
REL_INFO = 'TC_REL_INFO'
KING_DF = 'KING_DF'
REL_THRESH = 'REL_THRESHOLD'
G = 'G'
P = 'PHENOTYPE'
OUTPUT = 'EXPECTED_OUTPUT'



KEY_SIMPLE = "SIMPLE"
KEY_2FAM = "TWO_FAMILY"
KEY_VARIED = "VARIED"
KEY_PT = "PT"

KEY_MOD_DICT = {
    KEY_SIMPLE : tc_simple,
    KEY_2FAM : tc_2fam,
    KEY_VARIED : tc_varied, 
    KEY_PT : tc_pt
}

KEYS = list(KEY_MOD_DICT.keys())


GROUPNAME_SUBNAME_DICT = {
    "BIM_DFS" : "BIM_DF",
    "SNP_LISTS" : "SNP_LIST",
    "SNP_FILTERS" : "SNP_FILTER",
    "SAMPLE_LISTS" : "SAMPLE_LIST",
    "SAMPLE_FILTERS" : "SAMPLE_FILTER",
    "FAM_DFS" : "FAM_DF",
    "REL_DFS" : "REL_DF",
    "R_MATRICES" : "R_MATRICES",
    "G" : "G",
    "DEMEANED_P" : "DEMEANED_P",
    "DEMEANED_G" : "DEMEANED_G",
    "BETAS" : "BETAS",
    "SES" : "SES"
}

for groupname, subname in GROUPNAME_SUBNAME_DICT.items():
    globals()[groupname] = {key : getattr(mod_name, subname, None)
                            for key, mod_name in KEY_MOD_DICT.items()}
