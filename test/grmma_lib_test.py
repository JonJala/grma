"""
Testing of grma_lib.py
"""

import os
import sys

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_directory)

import numpy as np
import pytest

import grma_lib as sut


rng = np.random.default_rng()
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))


# TODO(jonbjala) Many more tests will need to be written

class TestResidualizePhenotypes:

    @pytest.mark.parametrize("N", [1, 10])
    def test___all_singletons___return_zeros(self, N):
        phenotypes = rng.random(N)
        rel_info = [[person_num] for person_num in range(N)]
        rel_set_sizes = np.full(N, 1, dtype=int)

        actual_results = sut.residualize_phenotypes(phenotypes=phenotypes,
                                                    rel_info=rel_info,
                                                    rel_set_sizes=rel_set_sizes)

        assert np.allclose(actual_results, 0.0)


class TestResidualizeGenotypes:

    @pytest.mark.parametrize("M", [1, 10])
    @pytest.mark.parametrize("N", [1, 10])
    def test___all_singletons___return_zeros(self, M, N):
        genotypes = rng.random((M, N))
        rel_info = [[person_num] for person_num in range(N)]
        rel_set_sizes = np.full(N, 1, dtype=int)

        actual_results = sut.residualize_genotypes(genotypes=genotypes,
                                                   rel_info=rel_info,
                                                   rel_set_sizes=rel_set_sizes)

        assert np.allclose(actual_results, 0.0)



class TestGrma:

    def test___toy_example_1___expected_results(self):
        testcase_name = 'toy_example_1'
        testcase_dir = os.path.join(data_directory, testcase_name)

        expected_betas_file = os.path.join(testcase_dir, f"{testcase_name}.betas")
        bed_file = os.path.join(testcase_dir, f"{testcase_name}.bed")
        bim_file = os.path.join(testcase_dir, f"{testcase_name}.bim")
        fam_file = os.path.join(testcase_dir, f"{testcase_name}.fam")
        rel_file = os.path.join(testcase_dir, f"{testcase_name}.kin")
        

        actual_betas, actual_ses = sut.grma(rel_input=rel_file, bed_file=bed_file,
                                            bim_file=bim_file, fam_file=fam_file)

        expected_betas = np.loadtxt(expected_betas_file)


        assert np.allclose(actual_betas, expected_betas, atol=0.0001)
        # TODO(jonbjala) Compare SEs once expected values are available
