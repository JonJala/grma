"""
SIMPLE: sample 0 is related to each of six others at a different degree -- full sib, parent-
offspring, 2nd, 3rd, 4th, and one listed as unrelated.  A SNP list keeps three of the five SNPs.
"""

import numpy as np
import pandas as pd


FAM_DF = pd.DataFrame({'FID' : [0, 1, 2, 3, 4, 5, 6], 'IID' : [0, 1, 2, 3, 4, 5, 6], 'IIDF' : [0, 0,
    0, 0, 0, 0, 0], 'IIDM' : [0, 0, 0, 0, 0, 0, 0], 'SEX' : [0, 0, 0, 0, 0, 0, 0],
    'PHENO' : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

REL_DF = pd.DataFrame({'FID1' : [0, 0, 0, 0, 0, 0], 'ID1' : [0, 0, 0, 0, 0, 0], 'FID2' : [1, 2, 3,
                       4,
    5, 6], 'ID2' : [1, 2, 3, 4, 5, 6], 'Kinship' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'InfType' : ['FS', 'PO', '2nd', '3rd', '4th', 'UN']})

BIM_DF = pd.DataFrame({'CHR' : [1, 1, 1, 1, 1], 'RSID' : ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
    'CM' : [1.0, 2.0, 3.0, 4.0, 5.0], 'BP' : [10, 20, 30, 40, 50], 'A1' : ['A', 'G', 'A',
    'A', 'C'], 'A2' : ['C', 'T', 'G', 'T', 'T']})

SNP_LIST = pd.DataFrame({'RSID' : ['rs1', 'rs3', 'rs4']})

G = np.array([[1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0],
              [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
              [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
              [1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0],
              [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0]])
