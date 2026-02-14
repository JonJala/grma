"""
Testing of grma_lib.py
"""

import itertools as it
import os
import sys

import numpy as np
import pytest
import pandas as pd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_directory = os.path.abspath(os.path.join(main_directory, "test"))
data_directory = os.path.abspath(os.path.join(test_directory, "data"))
sys.path.append(main_directory)
pd.options.mode.copy_on_write = True #https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#

import bedbimfam
import grma_lib as sut

import helper
import tcs

##############

# rng = np.random.default_rng(seed=0)


# testcase_name = "toy_example_1"
# testcase_dir = os.path.join(data_directory, testcase_name)
# fam_file = os.path.join(testcase_dir, f"{testcase_name}.fam")

# TODO(jonbjala) Many more tests will need to be written


def make_bim(bim_filename: str, M: int):
    rs_width = len(str(M))

    bedbimfam.write_bim_file(bim_filename=bim_filename, chrs=np.ones(M),
                             rsid=np.array([f'RS{snp_id:0{rs_width}d}' for snp_id in range(M)]),
                             bp=np.array([10*snp_id for snp_id in range(M)]),
                             a1=['G'] * M,
                             a2=['A'] * M)


def make_fam(fam_filename: str, N: int, pheno: np.ndarray=None):
    iids=np.array([i for i in range(N)])
    fids=np.zeros(N, dtype=int)
    bedbimfam.write_fam_file(fam_filename=fam_filename, fid=fids, iid=iids, pheno=pheno.ravel())


make_bed=bedbimfam.write_bed_file


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
def king_output_disconnected_rels():
    KING_DF_LENGTH = len(ALLOWED_INFTYPES)
    
    king_output = {
        "ID1": [i for i in range(1, 2 * KING_DF_LENGTH, 2)],
        "ID2": [i for i in range(2, 2 * KING_DF_LENGTH + 1, 2)],
        "FID1": [1] * KING_DF_LENGTH,
        "FID2": [1] * KING_DF_LENGTH,
        "Kinship": [0.50] * KING_DF_LENGTH,
        "InfType": ALLOWED_INFTYPES,
    }
    df = pd.DataFrame(king_output)
    return df

def generate_king_output_same_degree(inftype: str):
    NUM_RELS = 7
    NUM_ROWS = NUM_RELS * NUM_RELS

    king_output = {
        "ID1": [i for i in range(1, NUM_RELS + 1) for _ in range(NUM_RELS)],
        "ID2": [i for _ in range(NUM_RELS) for i in range(1, NUM_RELS + 1)],
        "FID1": [1] * NUM_ROWS,
        "FID2": [1] * NUM_ROWS,
        "Kinship": [0.50] * NUM_ROWS,
        "InfType": [inftype] * NUM_ROWS,
    }
    df = pd.DataFrame(king_output)
    df = df.loc[df["ID1"] != df["ID2"]]
    return df    


@pytest.fixture()
def king_output_same_degree(request):
    # Allowed InfTypes are ["Dup/MZTwin", "FS", "PO" "1", "2", "3", "4", "UN"]
    return generate_king_output_same_degree(inftype=request.param)

def fam_file_format(num_rows: int):
    # Sets rng to simulate num_rows number of phenotypes
    rng = np.random.default_rng(seed=0)
    fam_file = {
        "FID": [1] * num_rows,
        "IID": [i for i in range(1, num_rows + 1)],
        "IIDF": [0] * num_rows,
        "IIDM": [0] * num_rows,
        "SEX": [1] * num_rows,
        "PHENO": list(rng.integers(1, 6, size=num_rows).astype(float)),
    }
    fam_df = pd.DataFrame(fam_file)
    return fam_df

@pytest.fixture
def short_fam_file():
    return fam_file_format(num_rows=8)

@pytest.fixture
def long_fam_file():
    return fam_file_format(num_rows=20)

# List of InfTypes that are available in the InfType column of a King output file.
ALLOWED_INFTYPES = [key for key in sut.INF_TO_DEG_MAP.keys()]

# Getting MAX_ID of king_output_same_degree to use in tests
DF2 = generate_king_output_same_degree(inftype="Dup/MZTwin") # InfType doesn't matter here, as we only care about max_ID for range in iid parametrization. 
DF2_MAX_ID = max(DF2['ID1'].max(), DF2['ID2'].max())

class TestKingOutputtoRelInfo:
    def test__FS(self, base_king_output, short_fam_file):
        # Checking rel degree = FS protocol works as expected.
        rel_degree = "FS"
        df = base_king_output
        df["InfType"] = ["UN", "UN", "UN", "FS", "2nd", "3rd", "Dup/MZTwin", "4th", "3rd", "2nd", "PO",]

        expected_rel_info = [[0], [1, 2], [1, 2, 5], [3], [4], [2, 5], [6], [7]]
        expected_rel_size = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0])
        actual_rel_info, actual_rel_sizes = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=short_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size) 

    def test__no_obs__all_singletons(self, base_king_output, short_fam_file):
        rel_degree = "1"
        df = base_king_output
        df["InfType"] = ["2nd", "3rd", "3rd", "4th", "2nd", "3rd", "4th", "4th", "3rd", "2nd", "3rd",]

        expected_rel_info = [[0], [1], [2], [3], [4], [5], [6], [7]]
        expected_rel_size = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        actual_rel_info, actual_rel_sizes, se_info = sut.convert_king_output_to_rel_info(king_output=df, fam_filename=short_fam_file, rel_degree=rel_degree)

        assert actual_rel_info == expected_rel_info
        assert np.array_equal(actual_rel_sizes, expected_rel_size)          

    # test__iid_value_doesnt_matter makes sure that no matter where the excluded ID is (due to having relatives that are not close enough), the rel info object is calculated correctly.
    # It does this by iterating through 1 until the Max_ID, and setting every row that contains a degree equal to the counter to UN.
    # Since this DF holds individuals who are all fully connected with the rest of the pop and all with the same degree of relation, this ensures that the iid observed has length of the rel_list for iid = 1 (due to UN relatives), while the rest are still connected to everyone else (Max_ID - 1)
    # Since iid is a range, the test will be run DF2_MAX_ID times, each time with a single iid value.
    @pytest.mark.parametrize("iid", range(1, DF2_MAX_ID + 1))
    @pytest.mark.parametrize("king_output_same_degree", ["Dup/MZTwin"], indirect=True)
    def test__iid_value_doesnt_matter(self, king_output_same_degree, long_fam_file, iid):
        rel_degree = "3"
        fam_df = long_fam_file
        king_df= king_output_same_degree
        contains_iid = (king_df['ID1']==iid) | (king_df['ID2']==iid)
        king_df.loc[contains_iid, 'InfType'] = 'UN'

        actual_rel_info, actual_rel_sizes, se_info = sut.convert_king_output_to_rel_info(king_output=king_df, fam_filename=fam_df, rel_degree=rel_degree)

        assert len(actual_rel_info[iid-1]) == 1
        assert all(len(actual_rel_info[id]) == DF2_MAX_ID-1 for id in range(DF2_MAX_ID) if id != iid-1)  

    # test__varying_threshold__expected_results repeatedly raises the relatedness degree input on a df that has one pair of individuals per degree of relation
    # This allows us to test that rel_info is correct for all levels of relatedness inputs on a single df.
    @pytest.mark.parametrize("rel_index", range(len(sut.REL_DEG_INPUTS)))
    def test__varying_threshold__expected_results(self, king_output_disconnected_rels, long_fam_file, rel_index):
        fam_df = long_fam_file
        king_df= king_output_disconnected_rels

        expected_sizes = [[1]] * len(fam_df)
        expected_sizes[0:(4 + 2*rel_index)] = [["a", "b"]] * (4+ 2*rel_index)

        actual_rel_info, actual_rel_sizes, se_info = sut.convert_king_output_to_rel_info(king_output=king_df, fam_filename=fam_df, rel_degree=sut.REL_DEG_INPUTS[rel_index])

        assert len(list(it.chain(*actual_rel_info))) == len(list(it.chain(*expected_sizes)))
        assert all(len(actual_rel_info[i]) == len(expected_sizes[i]) for i in range(len(fam_df)))

    # test__varying_threshold_within_every_degree uses the fully connected df to vary both InfType values within the df and rel_degrees and ensures that rel sizes are correct.

    @pytest.mark.parametrize("king_output_same_degree", ALLOWED_INFTYPES, indirect=True)
    @pytest.mark.parametrize("rel_index", range(len(sut.REL_DEG_INPUTS)))
    def test__varying_threshold_within_every_degree__expected_results(self, king_output_same_degree, long_fam_file, rel_index):
        fam_df = long_fam_file
        king_df = king_output_same_degree
        inftype = king_df["InfType"][1]
        inftype_index = ALLOWED_INFTYPES.index(inftype)

        if inftype_index - 1 <= rel_index:
            num_singletons = len(fam_df) - DF2_MAX_ID
            expected_size = (DF2_MAX_ID * DF2_MAX_ID) + num_singletons
        else:
            expected_size = len(fam_df)

        actual_rel_info, actual_rel_sizes, se_info = sut.convert_king_output_to_rel_info(king_output=king_df, fam_filename=fam_df, rel_degree=sut.REL_DEG_INPUTS[rel_index])
        assert len(list(it.chain(*actual_rel_info))) == expected_size
    


class TestGRMA:

    @pytest.mark.parametrize("tc", tcs.TC_DATA)
    def test__end_to_end__expected_results(self, tmp_path, tc):
        M,N = tc[tcs.G].shape

        # Create bed/bim/fam files
        bed_filename = os.path.join(tmp_path, "temp.bed")
        bim_filename = os.path.join(tmp_path, "temp.bim")
        fam_filename = os.path.join(tmp_path, "temp.fam")
        make_bed(bed_filename=bed_filename, G=tc[tcs.G])
        make_bim(bim_filename=bim_filename, M=M)
        make_fam(fam_filename=fam_filename, N=N, pheno=tc[tcs.P])

        # Get rel_info and check against expected if that is specified
        rel_info, se_info = helper.mock_create_rel_info(king_df=tc[tcs.KING_DF],
                                               rel_thresh=tc[tcs.REL_THRESH])
        if tc.get(tcs.REL_INFO) is not None:
            expected_rel_info = tc[tcs.REL_INFO]
            assert rel_info == expected_rel_info

        if tc.get(tcs.SE_INFO) is not None:
            expected_se_info = tc[tcs.SE_INFO]
            assert se_info == expected_se_info


        # Get expected results
        expected_betas, expected_ses = tc[tcs.OUTPUT] if tc.get(tcs.OUTPUT) is not None else \
                                       helper.mock_grma(rel_input=rel_info, se_info=se_info,
                                                        G=tc[tcs.G], pheno=tc[tcs.P])
        test_betas, test_ses = helper.mock_grma(rel_input=rel_info, se_info=se_info,
                                                        G=tc[tcs.G], pheno=tc[tcs.P])

        # Get actual results
        result_df = sut.grma(rel_input=tc[tcs.KING_DF],
                             bed_file=bed_filename,
                             bim_file=bim_filename,
                             fam_file=fam_filename,
                             rel_degree=tc[tcs.REL_THRESH])

        actual_betas = result_df[sut.OUTPUT_BETA_COL].to_numpy()
        actual_ses = result_df[sut.OUTPUT_SE_COL].to_numpy()


        # Compare
        assert np.allclose(actual_betas, expected_betas, atol=0.001) if tc.get(tcs.OUTPUT) is not None else \
               np.allclose(actual_betas, expected_betas)
        assert np.allclose(actual_ses, expected_ses, atol=0.001) if tc.get(tcs.OUTPUT) is not None else \
               np.allclose(actual_ses, expected_ses)
