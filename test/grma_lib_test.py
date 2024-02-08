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
def base_fam_file():
    fam_file = {
    'FID': [1, 1, 1, 1, 1, 1, 1, 1],
    'IID': [1, 2, 3, 4, 5, 6, 7, 8],
    'IIDF': [0, 0, 0, 0, 0, 0, 0, 0],
    'IIDM': [0, 0, 0, 0, 0, 0, 0, 0],
    'SEX': [1, 1, 1, 1, 1, 1, 1, 1],
    'PHENO': [4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 6.0, 4.0]
}
    
    fam_df = pd.DataFrame(fam_file)
    return fam_df


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
        
        