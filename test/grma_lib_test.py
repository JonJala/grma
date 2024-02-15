"""
Testing of grma_lib.py
"""

import os
import sys

__file__ = "grma_lib_test.py"
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import numpy as np
import pytest
import pandas as pd
import itertools as it

import grma_lib as sut


rng = np.random.default_rng()
test_directory = os.path.abspath(os.path.join(main_directory, "test"))
data_directory = os.path.abspath(os.path.join(test_directory, "data"))
testcase_name = "toy_example_1"
testcase_dir = os.path.join(data_directory, testcase_name)
        # fam_file = '/disk/genetics/ws/dhruvaj/grma/test/data/toy_example_1/toy_example_1.fam'
#fam_file = os.path.join(testcase_dir, f"{testcase_name}.fam")

# TODO(jonbjala) Many more tests will need to be written

@pytest.fixture
def base_king_output():
    king_output = {
        "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
        "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
        "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
    }
    df = pd.DataFrame(king_output)
    return df


@pytest.fixture
def KingOutput_DisconnectedRels():
    KING_DF_LENGTH = 6

    ID1 = [i for i in range(1, 2*KING_DF_LENGTH, 2)]
    ID2 = [i for i in range(2, 2*KING_DF_LENGTH + 1, 2)]
    FID1 = [1] * KING_DF_LENGTH
    FID2 = [1] * KING_DF_LENGTH
    Kinship = [0.50] * KING_DF_LENGTH
    InfType = ["Dup/MZTwin", "FS", "PO", "2nd", "3rd", "4th"]

    king_output = {
        "ID1" : ID1,
        "ID2" : ID2,
        "FID1" : FID1,
        "FID2" : FID2,
        "Kinship" : Kinship,
        "InfType" : InfType,
    }
    df = pd.DataFrame(king_output)
    return df
 
def generate_KingOutput_SameDegree():
    
    NUM_RELS = 7
    NUM_ROWS = NUM_RELS * (NUM_RELS)
    
    ID1 = [i for i in range(1, NUM_RELS + 1) for _ in range(NUM_RELS)]
    ID2 = [i for _ in range(NUM_RELS) for i in range(1, NUM_RELS + 1)]

    FID1 = [1] * NUM_ROWS
    FID2 = [1] * NUM_ROWS
    Kinship = [0.50] * NUM_ROWS
    InfType = ["Dup/MZTwin"] * NUM_ROWS

    king_output = {
        "ID1" : ID1,
        "ID2" : ID2,
        "FID1" : FID1,
        "FID2" : FID2,
        "Kinship" : Kinship,
        "InfType" : InfType,
    }
    df = pd.DataFrame(king_output)
    df = df.loc[df["ID1"] != df["ID2"]]
    return df    
    
@pytest.fixture
def KingOutput_SameDegree():
    return generate_KingOutput_SameDegree()

def fam_file_format(num_rows):
    NUM_ROWS = num_rows

    FID = [1] * NUM_ROWS
    IID = [i for i in range(1, NUM_ROWS + 1)]
    IIDF = [0] * NUM_ROWS
    IIDM = [0] * NUM_ROWS 
    SEX = [1] * NUM_ROWS
    PHENO = [4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 6.0, 4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 6.0, 4.0, 4.0, 3.0, 4.0, 3.0]
    PHENO = PHENO[0:NUM_ROWS]
    
    fam_file = {
        "FID": FID,
        "IID": IID,
        "IIDF" : IIDF,
        "IIDM": IIDM,
        "SEX": SEX,
        "PHENO": PHENO
    }
    fam_df = pd.DataFrame(fam_file)
    return fam_df

@pytest.fixture
def base_fam_file():
    return fam_file_format(num_rows=8)

@pytest.fixture
def long_fam_file():
    return fam_file_format(num_rows=20)

class TestKingOutputtoRelInfo:
    def test_DropEnds(self, base_king_output, base_fam_file):
        # Start and end IDs are dropped given rel degree 3 but everyone else stays
        rel_degree = "3"
        df = base_king_output
        df["InfType"] = ["4th", "4th", "4th", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "4th",]
        
        expected_rel_info = [[0], [1, 2], [1, 2, 5], [3], [1, 4], [2, 5], [5, 6], [7]]
        expected_rel_size = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=base_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size)
        
    def test_DropBetween(self, base_king_output, base_fam_file):    
        # Start is UN so dropped and last ID is kept given rel degree 2. Some pairs are dropped in between. 
        rel_degree = "2"
        df = base_king_output
        df["InfType"] = ["UN", "UN", "UN", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "PO",]
        
        expected_rel_info = [[0], [1, 2], [1, 2, 5], [3], [1, 4], [2, 5, 7], [5, 6], [5,7]]
        expected_rel_size = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.0])
        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=base_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size)
     
    def test_FS(self, base_king_output, base_fam_file):    
        # Checking rel degree = FS protocol works as expected. 
        rel_degree = "FS"
        df = base_king_output
        df["InfType"] = ["UN", "UN", "UN", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "PO",]
        
        expected_rel_info = [[0], [1, 2], [1, 2, 5], [3], [4], [2, 5], [6], [7]]
        expected_rel_size = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0])
        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=base_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size) 
        
    def test_NoObs(self, base_king_output, base_fam_file):
        rel_degree = "1"
        df = base_king_output
        df["InfType"] = ["2nd", "3rd", "3rd", "4th", "2nd", "3rd", "4th", "4th", "3rd", "2nd", "3rd",]
        
        expected_rel_info = [[0], [1], [2], [3], [4], [5], [6], [7]]
        expected_rel_size = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=base_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size)          

    # test__iid_value_doesnt_matter makes sure that no matter where the excluded ID is (due to having relatives that are not close enough), the rel info object is calculated correctly.
    # It does this by iterating through 1 until the Max_ID, and setting every row that contains a degree equal to the counter to UN.
    # Since this DF holds individuals who are all fully connected with the rest of the pop and all with the same degree of relation, this ensures that the iid observed has length of the rel_list for iid = 1 (due to UN relatives), while the rest are still connected to everyone else (Max_ID - 1)       
    DF2 = generate_KingOutput_SameDegree()
    DF2_MAX_ID = max(DF2['ID1'].max(), DF2['ID2'].max())
    # Since iid is a range, the test will be run DF2_MAX_ID times, each time with a single iid value. 
    @pytest.mark.parametrize("iid", range(1, DF2_MAX_ID + 1))
    def test__iid_value_doesnt_matter(self, KingOutput_SameDegree, long_fam_file, iid):

        rel_degree = "4"
        MAX_ID = max(KingOutput_SameDegree['ID1'].max(), KingOutput_SameDegree['ID2'].max())
        fam_df = long_fam_file
        king_df= KingOutput_SameDegree
        contains_iid = (king_df['ID1']==iid) | (king_df['ID2']==iid)
        king_df.loc[contains_iid, 'InfType'] = 'UN'

        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=king_df, fam_filename=fam_df, rel_degree=rel_degree)

        assert len(actual_rel_info[iid-1]) == 1
        assert all(len(actual_rel_info[id]) == MAX_ID-1 for id in range(MAX_ID) if id != iid-1)  
    
        
    
    # test__varying_threshold__expected_results repeatedly raises the relatedness degree input on a df that has one pair of individuals per degree of relation
    # This allows us to test that rel_info is correct for all levels of relatedness inputs on a single df. 
    @pytest.mark.parametrize("rel_index", range(len(sut.REL_DEG_INPUTS)))
    def test__varying_threshold__expected_results(self, KingOutput_DisconnectedRels, long_fam_file, rel_index):
        
        fam_df = long_fam_file
        king_df= KingOutput_DisconnectedRels
        
        expected_sizes = [[1]] * (len(fam_df))
        expected_sizes[0:(4 + 2*rel_index)] = [["a", "b"]] * (4+ 2*rel_index)

        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=king_df, fam_filename=fam_df, rel_degree=sut.REL_DEG_INPUTS[rel_index])
        
        assert len(list(it.chain(*actual_rel_info))) == len(list(it.chain(*expected_sizes)))
        assert all(len(actual_rel_info[i]) == len(expected_sizes[i]) for i in range(len(fam_df)))

            
        