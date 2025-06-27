import os
import sys

import numpy as np
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
import grma_lib


NAME = 'TC_NAME'
REL_INFO = 'TC_REL_INFO'
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
]