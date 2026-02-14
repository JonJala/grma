import os
import sys

import numpy as np
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
import grma_lib


NAME = 'TC_NAME'
REL_INFO = 'TC_REL_INFO'
SE_INFO = 'TS_SE_INFO'
KING_DF = 'KING_DF'
REL_THRESH = 'REL_THRESHOLD'
G = 'G'
P = 'PHENOTYPE'
OUTPUT = 'EXPECTED_OUTPUT'


TC_DATA = [
    {
        NAME : "TC1",
        REL_THRESH : "FS",
        KING_DF : pd.DataFrame(
            {
                grma_lib.KING_FID1_COL : 0,
                grma_lib.KING_IID1_COL : [0, 0, 1, 3, 5],
                grma_lib.KING_FID2_COL : 0,
                grma_lib.KING_IID2_COL : [1, 2, 2, 4, 6],
                grma_lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.0],
                grma_lib.KING_REL_COL : ['FS', 'FS', 'FS', 'FS', 'UN']
            }
        ),
        REL_INFO : [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [5], [6]],
        G : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0]]),
        P : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0]]),

    },
    {
        NAME : "TC2",
        REL_THRESH : "2",
        KING_DF : pd.DataFrame(
            {
                grma_lib.KING_FID1_COL : 0,
                grma_lib.KING_IID1_COL : [0, 0, 1, 3, 3, 5],
                grma_lib.KING_FID2_COL : 0,
                grma_lib.KING_IID2_COL : [1, 2, 2, 4, 5, 6],
                grma_lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                grma_lib.KING_REL_COL : ['FS', 'FS', 'FS', 'FS', '2nd', '2nd']
            }
        ),
        REL_INFO : [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [3, 5, 6], [5, 6]],
        G : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0]]),
        P : np.array([[0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0]]),

    },
    {
        NAME : "PT_TC1",
        REL_THRESH : "2",
        KING_DF : pd.DataFrame(
            {
                grma_lib.KING_FID1_COL : 0,
                grma_lib.KING_IID1_COL : [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11],
                grma_lib.KING_FID2_COL : 0,
                grma_lib.KING_IID2_COL : [4, 8, 5, 9, 6, 10, 7, 11, 8, 12, 9, 13, 10, 14, 11, 15, 12, 13, 14, 15],
                grma_lib.KING_KINSHIP_COL : [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.125,
                                             0.5, 0.125, 0.5, 0.125, 0.5, 0.125, 0.125, 0.125, 0.125, 0.125],
                grma_lib.KING_REL_COL : ['2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', 'FS', 
                                         '3rd', 'FS', '3rd', 'FS', '3rd', 'FS', '3rd', '3rd', '3rd', '3rd', '3rd']
            }
        ),
        #REL_INFO : TODO(jonbjala) Ideally we'd check this too [[0, 1, 2], [0, 1, 2], [0, 1, 2], [3, 4], [3, 4], [3, 5, 6], [5, 6]],
        G : np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                      [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0],
                      [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]]),
        P : np.array([[0.603337104800799, -2.15109346613129, -1.84098136095182, 0.621513289522601,
                       -1.51923743106228, 0.504673163660100, 0.844569723965057, -1.28168116937994,
                       0.825550615142059, -0.727381594163808, 0.937130869463832, -0.925313249270390,
                       1.25661134338207, -1.39027626529803, -0.490438048864656, -0.263305423950850]]),
        OUTPUT : (np.array([-0.5278, -0.6866, 0.3934, -0.2798, -0.1751]),
                  np.array([0.9715, 0.0765, 0.6191, 0.6232, 0.1864]))

    }
]