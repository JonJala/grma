"""
THREE_GENERATIONS: three generations and several families -- grandparents, their children, the
grandchildren, a 2nd-degree uncle and a 3rd-degree cousin -- with one unrelated sample, X, listed
first.  The fam file's parent columns are filled in, but GRMA takes parents only from snipar's
pedigree, and there is no snipar file, so they are ignored.  The registry also runs it with X, and
with the parent mo1, left out of the analysis by an ID list.
"""

import numpy as np
import pandas as pd


FAM_DF = pd.DataFrame({'FID' : ['FX', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C',
                       'D',
    'D', 'U', 'C', 'D'], 'IID' : ['X', 'gf', 'gm', 'fa1', 'mo1', 'k1a', 'k1b', 'k1c', 'fa2',
    'k2a', 'k2b', 'k3a', 'k3b', 'k4a', 'k4b', 'u1', 'un3', 'cz4'], 'IIDF' : ['0', '0', '0',
    'gf', '0', 'fa1', 'fa1', 'fa1', '0', 'fa2', 'fa2', 'pa3', 'pa3', 'pa4', 'pa4', '0', '0',
    '0'], 'IIDM' : ['0', '0', '0', 'gm', '0', 'mo1', 'mo1', 'mo1', '0', 'mo2', 'mo2', 'ma3',
    'ma3', 'ma4', 'ma4', '0', '0', '0'], 'SEX' : ['0', '1', '2', '1', '2', '0', '0', '0',
    '1', '0', '0', '0', '0', '0', '0', '0', '1', '2'], 'PHENO' : [-1.5, -0.6, 1.9, -1.1, 0.3,
    1.0, -1.2, 0.8, 1.4, 1.3, 3.0, -0.9, -0.4, -0.1, -3.1, 0.1, -1.8, -3.0]})

REL_DF = pd.DataFrame({'FID1' : ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A',
                       'B',
    'C', 'D', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'D', 'D'], 'ID1' : ['gf', 'gm',
    'fa1', 'fa1', 'fa1', 'mo1', 'mo1', 'mo1', 'fa2', 'fa2', 'k1a', 'k1a', 'k1b', 'k2a',
    'k3a', 'k4a', 'gf', 'gf', 'gf', 'gm', 'gm', 'gm', 'un3', 'un3', 'k3a', 'cz4', 'cz4'],
    'FID2' : ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'B', 'C', 'D',
    'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'D', 'D', 'D'], 'ID2' : ['fa1', 'fa1', 'k1a',
    'k1b', 'k1c', 'k1a', 'k1b', 'k1c', 'k2a', 'k2b', 'k1b', 'k1c', 'k1c', 'k2b', 'k3b',
    'k4b', 'k1a', 'k1b', 'k1c', 'k1a', 'k1b', 'k1c', 'k3a', 'k3b', 'k4a', 'k4a', 'k4b'],
    'Kinship' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'InfType' : ['PO', 'PO',
    'PO', 'PO', 'PO', 'PO', 'PO', 'PO', 'PO', 'PO', 'FS', 'FS', 'FS', 'FS', 'FS', 'FS',
    '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '2nd', '3rd', '3rd', '3rd']})

BIM_DF = pd.DataFrame({'CHR' : [1, 1, 1, 1, 1, 1, 1], 'RSID' : ['rs1', 'rs2', 'rs3', 'rs4', 'rs5',
    'rs6', 'rs7'], 'CM' : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'BP' : [100, 200, 300, 400,
    500, 600, 700], 'A1' : ['A', 'C', 'G', 'A', 'C', 'A', 'G'], 'A2' : ['C', 'G', 'T', 'G',
    'T', 'T', 'C']})

G = np.array([[1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 0.0,
               0.0],
              [1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 1.0,
               0.0],
              [0.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0,
               1.0],
              [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0,
               0.0],
              [0.0, 1.0, np.nan, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0,
               0.0, 1.0],
              [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0,
               0.0],
              [0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0,
               2.0]])

SNPS_PER_BLOCK = 3
