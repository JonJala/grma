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
fam_file = os.path.join(testcase_dir, f"{testcase_name}.fam")

# TODO(jonbjala) Many more tests will need to be written


class TestKingOutputtoRelInfo:
    # @pytest.mark.parametrize("KingOutput, FamFile, RelDegree, RelInfoandSizes", )
    def test_OutputtoSet_DropEnds(self):
        # Start and end IDs are dropped given rel degree 3 but everyone else stays
        rel_degree1 = "3"
        output1 = {
            "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
            "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
            "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
            "InfType": ["4th", "4th", "4th", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "4th",],
        }
        df1 = pd.DataFrame(output1)
        expected_rel_info1 = [[0], [1, 2], [1, 2, 5], [3], [1, 4], [2, 5], [5, 6], [7]]
        expected_rel_size1 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        result1 = sut.convert_king_output_to_rel_info(king_output=df1, fam_filename=fam_file, rel_degree=rel_degree1)

        assert result1[0] == expected_rel_info1
        assert np.array_equal(result1[1], expected_rel_size1)
        
    def test_OutputtoSet_DropBetween(self):    
        # Start is UN so dropped and last ID is kept given rel degree 2. Some pairs are dropped in between. 
        rel_degree2 = "2"
        output2 = {
            "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
            "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
            "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
            "InfType": ["UN", "UN", "UN", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "PO",],
        }
        
        df2 = pd.DataFrame(output2)
        expected_rel_info2 = [[0], [1, 2], [1, 2, 5], [3], [1, 4], [2, 5, 7], [5, 6], [5,7]]
        expected_rel_size2 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 2.0])
        result2 = sut.convert_king_output_to_rel_info(king_output=df2, fam_filename=fam_file, rel_degree=rel_degree2)

        assert result2[0] == expected_rel_info2
        assert np.array_equal(result2[1], expected_rel_size2)
     
    def test_OutputtoSet_FS(self):    
        # Using output2 but checking rel degree = FS. 
        rel_degree3 = "FS"
        output3 = {
            "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
            "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
            "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
            "InfType": ["UN", "UN", "UN", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "PO",],
        }
        df3 = pd.DataFrame(output3)
        expected_rel_info3 = [[0], [1, 2], [1, 2, 5], [3], [4], [2, 5], [6], [7]]
        expected_rel_size3 = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0])
        result3 = sut.convert_king_output_to_rel_info(king_output=df3, fam_filename=fam_file, rel_degree=rel_degree3)

        assert result3[0] == expected_rel_info3
        assert np.array_equal(result3[1], expected_rel_size3)  
        
    def test_OutputtoSet_NoObs(self):
        rel_degree4 = "1"
        output4 = {
            "ID1": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
            "ID2": [2, 3, 4, 3, 5, 5, 6, 5, 7, 7, 8],
            "FID1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "FID2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Kinship": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,],
            "InfType": ["2nd", "3rd", "3rd", "4th", "2nd", "3rd", "4th", "4th", "3rd", "2nd", "3rd",],
        }
        df4 = pd.DataFrame(output4)
        expected_rel_info4 = [[0], [1], [2], [3], [4], [5], [6], [7]]
        expected_rel_size4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result4 = sut.convert_king_output_to_rel_info(king_output=df4, fam_filename=fam_file, rel_degree=rel_degree4)

        assert result4[0] == expected_rel_info4
        assert np.array_equal(result4[1], expected_rel_size4)               
        