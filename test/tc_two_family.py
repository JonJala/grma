"""
TWO_FAMILY: two families, each two parents and three full sibs, with every parent-offspring and
sibling pair listed in KING.  The fam file's parent columns are empty, and there is no snipar file,
so the parents are focal in no regression.  IIDs run from 1, so that none is the missing-parent
code 0.
"""

import numpy as np
import pandas as pd


FAM_DF = pd.DataFrame({'FID' : [0, 0, 1, 1, 0, 0, 0, 1, 1, 1], 'IID' : [1, 2, 3, 4, 5, 6, 7, 8, 9,
    10], 'IIDF' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'IIDM' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'SEX' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'PHENO' : [3.5, 0.5, -5.0, 2.0, 0.0, 1.0, 3.0,
    -2.5, -1.5, -3.5]})

REL_DF = pd.DataFrame({'FID1' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'ID1' : [1,
                       1,
    1, 2, 2, 2, 5, 5, 6, 3, 3, 3, 4, 4, 4, 8, 8, 9], 'FID2' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1], 'ID2' : [5, 6, 7, 5, 6, 7, 6, 7, 7, 8, 9, 10, 8, 9, 10, 9, 10,
    10], 'Kinship' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0], 'InfType' : ['PO', 'PO', 'PO', 'PO', 'PO', 'PO', 'FS', 'FS', 'FS',
    'PO', 'PO', 'PO', 'PO', 'PO', 'PO', 'FS', 'FS', 'FS']})

BIM_DF = pd.DataFrame({'CHR' : [1, 1, 1], 'RSID' : ['rs1', 'rs2', 'rs3'], 'CM' : [2.0, 4.0, 6.0],
    'BP' : [50, 100, 150], 'A1' : ['A', 'G', 'A'], 'A2' : ['C', 'T', 'T']})

G = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
              [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0],
              [2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]])
