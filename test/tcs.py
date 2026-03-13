import os
import sys

import numpy as np
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma_lib_new as lib

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
    "FAM_DFS" : "FAM_DF",
    "REL_DFS" : "REL_DF",
    "R_MATRICES" : "R_MATRICES",
    "BASE_SE_MATRICES" : "BASE_SE_MATRIX",
    "OMEGA" : "OMEGA",
    "SE_MATRICES" : "SE_MATRICES",
    "DEMEANED_P" : "DEMEANED_P",
    "DEMEANED_G" : "DEMEANED_G",
    "BETAS" : "BETAS",
    "SES" : "SES"
}



BIM_DFS = {key : getattr(mod_name, "BIM_DF", None) for key, mod_name in KEY_MOD_DICT.items()}


for groupname, subname in GROUPNAME_SUBNAME_DICT.items():
    globals()[groupname] = {key : getattr(mod_name, subname, None)
                            for key, mod_name in KEY_MOD_DICT.items()}


# BIM_DFS = {
#     KEY_SIMPLE : tc_simple.BIM_DF,
#     KEY_2FAM : tc_2fam.BIM_DF,
#     KEY_VARIED : tc_varied.BIM_DF
#     KEY_PT : tc_pt.BIM_DF
# }

# SNP_LISTS = {
#     KEY_SIMPLE : tc_simple.SNP_LIST,
#     KEY_2FAM : tc_2fam.SNP_LIST,
#     KEY_VARIED : tc_varied.SNP_LIST,
#     KEY_PT : tc_pt.BIM_DF
# }

# SNP_FILTERS = {
#     KEY_SIMPLE : tc_simple.SNP_FILTER,
#     KEY_2FAM : tc_2fam.SNP_FILTER,
#     KEY_VARIED : tc_varied.SNP_FILTER,
#     KEY_PT : tc_pt.BIM_DF
# }

# FAM_DFS = {
#     KEY_SIMPLE : tc_simple.FAM_DF,
#     KEY_2FAM : tc_2fam.FAM_DF,
#     KEY_VARIED : tc_varied.FAM_DF,
#     KEY_PT : tc_pt.BIM_DF
# }


# REL_DFS = {
#     KEY_SIMPLE : tc_simple.REL_DF,
#     KEY_2FAM : tc_2fam.REL_DF,
#     KEY_VARIED : tc_varied.REL_DF,
#     KEY_PT : tc_pt.BIM_DF
# }
    

# R_MATRICES = {
#     KEY_SIMPLE : tc_simple.R_MATRICES,
#     KEY_2FAM : tc_2fam.R_MATRICES,
#     KEY_VARIED : tc_varied.R_MATRICES,
#     KEY_PT : tc_pt.BIM_DF
# }



# BASE_SE_MATRICES = {
#     KEY_SIMPLE : tc_simple.BASE_SE_MATRIX,
#     KEY_2FAM : tc_2fam.BASE_SE_MATRIX,
#     KEY_VARIED : tc_varied.BASE_SE_MATRIX
# }

# SE_MATRICES = {
#     KEY_SIMPLE : tc_simple.SE_MATRICES,
#     KEY_2FAM : tc_2fam.SE_MATRICES,
#     KEY_VARIED : tc_varied.SE_MATRICES
# }


# DEMEANED_P = {
#     KEY_SIMPLE : tc_simple.DEMEANED_P,
#     KEY_2FAM : tc_2fam.DEMEANED_P,
#     KEY_VARIED : tc_varied.DEMEANED_P
# }

# DEMEANED_G = {
#     KEY_SIMPLE : tc_simple.DEMEANED_G,
#     KEY_2FAM : tc_2fam.DEMEANED_G,
#     KEY_VARIED : tc_varied.DEMEANED_G
# }




TC_DATA = [
    {
        NAME : "TC1",
        KING_DF : pd.DataFrame(
            {
                lib.KING_FID1_COL : [0, 0, 1, 3, 5],
                lib.KING_IID1_COL : [0, 0, 1, 3, 5],
                lib.KING_FID2_COL : [1, 2, 2, 4, 6],
                lib.KING_IID2_COL : [1, 2, 2, 4, 6],
                lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.0],
                lib.KING_REL_COL : ['FS', 'FS', 'FS', 'FS', 'UN']
            }
        ),
        REL_INFO : [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [5], [6]],
        G : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0]]),
        P : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0]]),
        OUTPUT : {"FS" : None}
    },
    {
        NAME : "TC2",
        KING_DF : pd.DataFrame(
            {
                lib.KING_FID1_COL : [0, 0, 1, 3, 3, 5],
                lib.KING_IID1_COL : [0, 0, 1, 3, 3, 5],
                lib.KING_FID2_COL : [1, 2, 2, 4, 5, 6],
                lib.KING_IID2_COL : [1, 2, 2, 4, 5, 6],
                lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                lib.KING_REL_COL : ['FS', 'FS', 'FS', 'FS', '2nd', '2nd']
            }
        ),
        REL_INFO : [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [3, 5, 6], [5, 6]],
        G : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0]]),
        P : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0]]),
        OUTPUT : {"2" : None}
    },
    {
        NAME : "PT_TC1",
        KING_DF : pd.DataFrame(
            {
                lib.KING_FID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
                lib.KING_IID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
                lib.KING_FID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
                lib.KING_IID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
                lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5,
                                             0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.125,
                                             0.125, 0.125, 0.125],
                lib.KING_REL_COL : ['2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd',
                                         'FS', '3rd', 'FS', '3rd', 'FS', '3rd', 'FS', '3rd', '3rd',
                                         '3rd', '3rd', '3rd']
            }
        ),
        #REL_INFO : TODO(jonbjala) Ideally we'd check this too [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [3, 5, 6], [5, 6]],
        G : np.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0],
                      [0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                      [1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0],
                      [2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                      [2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        P : np.array([[-1.42344863693359, -0.389217449209409, 0.96544596473892, -0.187216762581015,
                       -0.95279482865147, -0.492898529597248, 1.08251333278776, 0.380757278361716,
                       -1.79484158790788, -0.450136643248664, -0.533908194857535, -1.04204709826705,
                       1.30584821614275, 0.0451393286800111, -1.51153606825486, -1.94726566134081]]),
        OUTPUT : {
            "2" : (-np.array([-0.1406, 0.3927, 0.6023, 0.5681, -0.3527]),
                    np.array([0.1579, 0.2930, 0.4219, 0.5366, 0.3946]))
        }

    },
    {
        NAME : "PT_TC2",
        KING_DF : pd.DataFrame(
            {
                lib.KING_FID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
                lib.KING_IID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
                lib.KING_FID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
                lib.KING_IID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
                lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5,
                                             0.125, 0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.125,
                                             0.125, 0.125, 0.125],
                lib.KING_REL_COL : ['2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd',
                                         'FS', '3rd', 'FS', '3rd', 'FS', '3rd', 'FS', '3rd', '3rd',
                                         '3rd', '3rd', '3rd']
            }
        ),
        G : np.array([[np.nan, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0],
                      [2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                      [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0]]),
        P : np.array([[-0.341905000073181, -0.521146710987144, -0.457787817929638,
                       -0.649079974170937, 1.31901689906512, 0.0469918382036728, 0.683195789569862,
                       0.854897375543712, -0.251033113382805, -0.213594117514055, -1.37435039334038,
                       1.01505338064033, -0.741416243216773, -0.13940270416759, -0.251812536912985,
                       1.01079848109058]]),
        OUTPUT : {
            "FS" : (-np.array([-0.2606, -0.1602, -0.2606, 0.4899, -1.1223]),
                     np.array([0.2606, 0.1602, 0.2606, 0.5332, 0.8671])),
            "2" : (-np.array([-0.3724, 0.0515, -0.1684, 0.7521, -0.4618]),
                    np.array([0.3724, 0.3421, 0.2207, 0.5843, 0.6962])),
            "3" : (-np.array([-0.4644, -0.1047, 0.1630, 0.4977, -0.4589]),
                    np.array([0.3806, 0.2347, 0.2779, 0.3636, 0.6009]))
        }

    }
]