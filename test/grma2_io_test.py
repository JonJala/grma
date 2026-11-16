"""
Tests of grma2_io: GRMA's readers.

Every reader is checked for the same three things: IDs come back as text (leading zeros intact),
a file and the equivalent DataFrame give identical results, and bad values are errors that name
the rows affected.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma2_io as sut
import snipar_hdf5


def write(tmp_path, name, text):
    path = os.path.join(tmp_path, name)
    with open(path, "w") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------------------------------
class TestReadFam:

    FAM_TEXT = ("007 001 0 0 1 1.5\n"
                "007 002 001 0 2 -9\n"
                "8 3 0 0 0 NA\n"
                "8 4 0 0 0 0\n")

    def test__ids__read_as_text_with_leading_zeros(self, tmp_path):
        actual = sut.read_fam(write(tmp_path, "x.fam", self.FAM_TEXT))

        assert actual[bbf.FAM_FID_COL].tolist() == ["007", "007", "8", "8"]
        assert actual[bbf.FAM_IID_COL].tolist() == ["001", "002", "3", "4"]
        assert actual[bbf.FAM_IIDF_COL].tolist() == ["0", "001", "0", "0"]
        assert actual[bbf.FAM_SEX_COL].tolist() == ["1", "2", "0", "0"]

    def test__phenotypes__minus_nine_and_na_missing_zero_kept(self, tmp_path):
        actual = sut.read_fam(write(tmp_path, "x.fam", self.FAM_TEXT))[bbf.FAM_PHENO_COL]

        assert actual.iloc[0] == 1.5
        assert np.isnan(actual.iloc[1]) and np.isnan(actual.iloc[2])
        assert actual.iloc[3] == 0.0

    def test__dataframe_with_numeric_ids__same_as_text(self):
        frame = pd.DataFrame({bbf.FAM_FID_COL : [8, 8], bbf.FAM_IID_COL : [3, 4],
                              bbf.FAM_IIDF_COL : 0, bbf.FAM_IIDM_COL : 0, bbf.FAM_SEX_COL : 0,
                              bbf.FAM_PHENO_COL : [-9.0, 2.0]})

        actual = sut.read_fam(frame)

        assert actual[bbf.FAM_IID_COL].tolist() == ["3", "4"]
        assert np.isnan(actual[bbf.FAM_PHENO_COL].iloc[0])
        assert actual[bbf.FAM_PHENO_COL].iloc[1] == 2.0

    def test__file_and_dataframe__identical(self, tmp_path):
        from_file = sut.read_fam(write(tmp_path, "x.fam", self.FAM_TEXT))
        frame = from_file.copy()
        # As text, the way a frame passed in might hold it
        frame[bbf.FAM_PHENO_COL] = ["1.5", "-9", "NA", "0"]

        pd.testing.assert_frame_equal(sut.read_fam(frame), from_file)

    def test__dataframe_passed_in__not_modified(self):
        frame = pd.DataFrame({bbf.FAM_FID_COL : [8], bbf.FAM_IID_COL : [3], bbf.FAM_IIDF_COL : 0,
                              bbf.FAM_IIDM_COL : 0, bbf.FAM_SEX_COL : 0,
                              bbf.FAM_PHENO_COL : [-9.0]})
        before = frame.copy()

        sut.read_fam(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test__phenotype_not_a_number__raises_naming_the_sample(self, tmp_path):
        path = write(tmp_path, "x.fam", "8 3 0 0 0 1.0\n8 4 0 0 0 tall\n")

        with pytest.raises(ValueError, match="not numbers") as error:
            sut.read_fam(path)

        assert "8 4" in str(error.value)


# --------------------------------------------------------------------------------------------------
class TestReadPheno:

    def test__values_and_missing_codes(self, tmp_path):
        actual = sut.read_pheno(write(tmp_path, "p.txt", "007 001 2.5\n8 3 -9\n8 4 0\n"))

        assert actual[bbf.FAM_FID_COL].tolist() == ["007", "8", "8"]
        assert actual[bbf.FAM_IID_COL].tolist() == ["001", "3", "4"]
        assert actual[bbf.FAM_PHENO_COL].iloc[0] == 2.5
        assert np.isnan(actual[bbf.FAM_PHENO_COL].iloc[1])
        assert actual[bbf.FAM_PHENO_COL].iloc[2] == 0.0

    def test__phenotype_not_a_number__raises(self, tmp_path):
        with pytest.raises(ValueError, match="not numbers"):
            sut.read_pheno(write(tmp_path, "p.txt", "8 3 high\n"))


# --------------------------------------------------------------------------------------------------
class TestReadCovar:

    def test__covariates__named_in_order_as_floats(self, tmp_path):
        actual = sut.read_covar(write(tmp_path, "c.txt", "007 001 40 0.5\n8 3 NA 1\n"))

        assert actual.columns.tolist() == [bbf.FAM_FID_COL, bbf.FAM_IID_COL, "COVAR_1", "COVAR_2"]
        assert actual[bbf.FAM_FID_COL].tolist() == ["007", "8"]
        assert actual["COVAR_1"].iloc[0] == 40.0 and np.isnan(actual["COVAR_1"].iloc[1])
        assert actual["COVAR_2"].tolist() == [0.5, 1.0]

    def test__no_covariate_column__raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one covariate"):
            sut.read_covar(write(tmp_path, "c.txt", "8 3\n8 4\n"))

    def test__covariate_not_a_number__raises_naming_the_sample(self, tmp_path):
        with pytest.raises(ValueError, match="not numbers") as error:
            sut.read_covar(write(tmp_path, "c.txt", "8 3 40\n8 4 old\n"))

        assert "8 4" in str(error.value)


# --------------------------------------------------------------------------------------------------
class TestReadIdList:

    def test__first_two_columns_as_text__rest_ignored(self, tmp_path):
        actual = sut.read_id_list(write(tmp_path, "ids.txt", "007 001 extra\n8 3 more\n"))

        assert actual.columns.tolist() == [bbf.FAM_FID_COL, bbf.FAM_IID_COL]
        assert actual[bbf.FAM_FID_COL].tolist() == ["007", "8"]
        assert actual[bbf.FAM_IID_COL].tolist() == ["001", "3"]


# --------------------------------------------------------------------------------------------------
class TestReadKing:

    KING_TEXT = ("FID1\tID1\tFID2\tID2\tN_SNP\tKinship\tInfType\n"
                 "007\t001\t007\t002\t1000\t0.25\tFS\n"
                 "8\t3\t8\t4\t1000\t0.12\t2nd\n")

    def test__needed_columns_as_text(self, tmp_path):
        actual = sut.read_king(write(tmp_path, "k.kin0", self.KING_TEXT))

        assert actual.columns.tolist() == list(sut.KING_COLS)
        assert actual[sut.KING_FID1_COL].tolist() == ["007", "8"]
        assert actual[sut.KING_IID2_COL].tolist() == ["002", "4"]
        assert actual[sut.KING_REL_COL].tolist() == ["FS", "2nd"]

    def test__a_needed_column_missing__raises_naming_it(self, tmp_path):
        text = "FID1\tID1\tFID2\tID2\tKinship\n8\t3\t8\t4\t0.25\n"

        with pytest.raises(ValueError, match="InfType"):
            sut.read_king(write(tmp_path, "k.kin0", text))


# --------------------------------------------------------------------------------------------------
class TestReadBim:

    def test__types(self, tmp_path):
        actual = sut.read_bim(write(tmp_path, "x.bim", "1 rs007 0.5 100 A C\n1 7 0 200 G T\n"))

        assert actual[bbf.BIM_RSID_COL].tolist() == ["rs007", "7"]
        assert actual[bbf.BIM_CHR_COL].tolist() == ["1", "1"]
        assert actual[bbf.BIM_CM_COL].tolist() == [0.5, 0.0]
        assert actual[bbf.BIM_BP_COL].tolist() == [100, 200]
        assert actual[bbf.BIM_A1_COL].tolist() == ["A", "G"]


# --------------------------------------------------------------------------------------------------
class TestReadTable:

    def test__neither_path_nor_dataframe__raises(self):
        with pytest.raises(TypeError, match="fam file"):
            sut.read_fam(42)


# --------------------------------------------------------------------------------------------------
class TestReadSnpList:

    def test__first_column_as_text__rest_ignored(self, tmp_path):
        actual = sut.read_snp_list(write(tmp_path, "snps.txt", "rs1 extra\n007\n"))

        assert actual.columns.tolist() == [bbf.BIM_RSID_COL]
        assert actual[bbf.BIM_RSID_COL].tolist() == ["rs1", "007"]

    def test__only_numeric_looking_ids__still_text(self, tmp_path):
        # A column mixing "rs1" with numbers is read as text whatever the reader asks for; one of
        # numbers alone is not, unless the reader insists
        actual = sut.read_snp_list(write(tmp_path, "snps.txt", "007\n8\n"))

        assert actual[bbf.BIM_RSID_COL].tolist() == ["007", "8"]


# --------------------------------------------------------------------------------------------------
class TestSniparReader:

    NUM_SNPS = 10
    FAMILIES = ["F1", "F2"]

    def write_file(self, tmp_path, chunk_snps=3):
        """A snipar file whose imputed value is 0.25 x family number + 0.01 x SNP number, so every
        value read can be traced to where it came from (and is exact in float16)"""
        imputed = np.array([[0.25 * (f + 1) + 0.01 * s for s in range(self.NUM_SNPS)]
                            for f in range(len(self.FAMILIES))])
        imputed = np.round(imputed * 64) / 64           # Exactly representable in float16
        counts = np.full(imputed.shape, 3.0)
        counts[1, 4] = 4.0
        f = np.round(np.linspace(0.1, 0.9, self.NUM_SNPS) * 64) / 64
        path = os.path.join(tmp_path, "imp" + snipar_hdf5.SNIPAR_FILE_SUFFIX)
        snipar_hdf5.write_snipar_file(
            path, imputed_gts=imputed, num_obs_par_alleles=counts, standard_f=f,
            parental_status=np.zeros((len(self.FAMILIES), 3), dtype=bool),
            pedigree=np.array([["F1", "k", "0", "0", "False", "False"]]),
            families=np.array(self.FAMILIES),
            bim_values=np.array([["1", f"rs{s}", "0", str(100 * s), "A", "C"]
                                 for s in range(self.NUM_SNPS)]),
            chunk_snps=chunk_snps)
        return path, imputed, counts, f

    def test__block_across_chunks__every_value_from_its_snp(self, tmp_path):
        path, imputed, counts, f = self.write_file(tmp_path)
        positions = np.array([1, 2, 3, 7])          # Crosses two chunk boundaries

        with sut.SniparReader(path, f) as reader:
            actual_imputed, actual_counts, actual_f = reader.read(positions, np.zeros(4, bool))

        assert np.allclose(actual_imputed, imputed[:, positions].T)
        assert np.allclose(actual_counts, counts[:, positions].T)
        assert np.allclose(actual_f, f[positions])

    def test__snp_the_file_lacks__nan(self, tmp_path):
        path, imputed, _, f = self.write_file(tmp_path)

        with sut.SniparReader(path, f) as reader:
            actual_imputed, actual_counts, actual_f = reader.read(np.array([0, -1, 5]),
                                                                  np.zeros(3, bool))

        assert np.all(np.isnan(actual_imputed[1])) and np.all(np.isnan(actual_counts[1]))
        assert np.isnan(actual_f[1])
        assert np.allclose(actual_imputed[[0, 2]], imputed[:, [0, 5]].T)

    def test__nothing_in_the_file__all_nan_one_column_per_family(self, tmp_path):
        path, _, _, f = self.write_file(tmp_path)

        with sut.SniparReader(path, f) as reader:
            actual_imputed, _, _ = reader.read(np.array([-1, -1]), np.zeros(2, bool))

        assert actual_imputed.shape == (2, len(self.FAMILIES))
        assert np.all(np.isnan(actual_imputed))

    def test__flipped_snp__counted_in_the_other_allele(self, tmp_path):
        path, imputed, counts, f = self.write_file(tmp_path)

        with sut.SniparReader(path, f) as reader:
            actual_imputed, actual_counts, actual_f = reader.read(np.array([2, 6]),
                                                                  np.array([False, True]))

        assert np.allclose(actual_imputed[0], imputed[:, 2])
        assert np.allclose(actual_imputed[1], 2.0 - imputed[:, 6])
        assert np.isclose(actual_f[1], 1.0 - f[6])
        assert np.allclose(actual_counts[1], counts[:, 6])        # Counts do not flip

    def test__blocks_in_order__each_chunk_read_once(self, tmp_path, monkeypatch):
        path, imputed, _, f = self.write_file(tmp_path, chunk_snps=4)
        reads = []
        real = snipar_hdf5.read_snipar_snp_range
        monkeypatch.setattr(snipar_hdf5, "read_snipar_snp_range",
                            lambda h, a, b: reads.append((a, b)) or real(h, a, b))

        with sut.SniparReader(path, f) as reader:
            first = reader.read(np.array([0, 1, 2]), np.zeros(3, bool))
            second = reader.read(np.array([3, 3, 5]), np.zeros(3, bool))   # Repeated position
            third = reader.read(np.array([6, 9]), np.zeros(2, bool))

        assert reads == [(0, 4), (4, 8), (8, 10)]
        assert np.allclose(second[0], imputed[:, [3, 3, 5]].T)
        assert np.allclose(third[0], imputed[:, [6, 9]].T) and first[0].shape == (3, 2)

    def test__closed_on_leaving_the_with_block(self, tmp_path):
        path, _, _, f = self.write_file(tmp_path)

        with sut.SniparReader(path, f) as reader:
            pass

        assert not reader._file.id.valid


# --------------------------------------------------------------------------------------------------
class TestReadGenotypeBlock:

    G = np.array([[0, 1, 2, np.nan], [2, 2, 0, 1], [1, 0, 1, 2], [0, 0, 2, 2], [1, 1, 1, 0]],
                 dtype=float)

    def test__array__chosen_snps_and_samples(self):
        actual = sut.read_genotype_block(self.G, 5, 4, np.array([1, 3, 4]), np.array([3, 0]))

        assert np.array_equal(actual, self.G[[1, 3, 4]][:, [3, 0]], equal_nan=True)

    def test__bed_file__same_as_the_array_missing_values_included(self, tmp_path):
        path = os.path.join(tmp_path, "g.bed")
        bbf.write_bed_file(path, self.G)

        actual = sut.read_genotype_block(path, 5, 4, np.array([1, 3, 4]), np.array([3, 1, 0]))

        # Starting after the file's first SNP, so that the offset into what is read matters
        assert np.array_equal(actual, self.G[[1, 3, 4]][:, [3, 1, 0]], equal_nan=True)


# --------------------------------------------------------------------------------------------------
# Copied from the original implementation's tests, with the function (one assertion adapted)

class TestCalculatePVals:

    @pytest.mark.parametrize("betas, ses, expected_p", [
        (np.array([1.96, 2.576]), np.array([1.0, 1.0]), np.array([0.05, 0.01])),
        (np.array([1.0, 1.0]), np.array([1.0/1.96, 1.0/2.576]), np.array([0.05, 0.01])),
        (np.array([1.96, 2.576]), np.array([-1.0, -1.0]), np.array([0.05, 0.01]))
    ])
    def test__precanned_tcs__expected_results(self, betas, ses, expected_p):
        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.allclose(actual_p.astype(float), expected_p, atol=0.00001)


    @pytest.mark.parametrize("rng_seed", [54, 87])
    def test__simple_inputs__expected_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100

        betas = 2.0 * rng.random(N) - 1.0
        ses = 2.0 * rng.random(N) + 10.0

        expected_p = 2.0 * (1.0 - stats.norm.cdf(np.abs(betas / ses)))

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.allclose(actual_p.astype(float), expected_p)


    @pytest.mark.parametrize("rng_seed", [316, 9999])
    def test__negative_z__same_results(self, rng_seed):
        rng = np.random.default_rng(seed=rng_seed)

        N = 100

        betas = 2.0 * rng.random(N) - 1.0
        ses = 2.0 * rng.random(N) + 10.0
        mask = rng.choice([True, False], size=N)

        first_p = sut.calculate_pvals(betas=betas, ses=ses)

        betas[mask] = -betas[mask]
        second_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.all(first_p == second_p)


    def test__include_very_small_p__expected_results(self):
        betas = np.array([39.0, -39.0])
        ses = np.array([1.0, 1.0])

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        # Adapted: the result may now hold None, so it is no longer a pure text array that
        # np.char.find can search
        assert all("-332" in p for p in actual_p)


    def test__include_p_equals_1__expected_results(self):
        betas = np.array([0.0, 0.0, 0.0])
        ses = np.array([1.0, -1.0, 93478.0])

        actual_p = sut.calculate_pvals(betas=betas, ses=ses)

        assert np.all(actual_p == "1.0e0")


    def test__no_estimate__no_p_value(self):
        # Changed from the original, which gave the text "nan": now None, written as missing.  A
        # missing estimate has a missing beta and SE.
        actual_p = sut.calculate_pvals(betas=np.array([np.nan, 1.0]),
                                       ses=np.array([np.nan, 1.0]))

        assert actual_p[0] is None and actual_p[1] is not None


# --------------------------------------------------------------------------------------------------
class TestBuildOutput:

    BIM = sut.read_bim(pd.DataFrame([("1", "rs1", 0.0, 100, "A", "C"),
                                     ("1", "rs2", 0.0, 200, "G", "T")],
                                    columns=list(bbf.BIM_COLS)))

    def test__bim_columns_then_beta_se_p_for_each_estimate(self):
        betas = np.array([[0.5, 0.4], [np.nan, -0.2]])
        ses = np.array([[0.1, 0.1], [np.nan, 0.05]])

        actual = sut.build_output(self.BIM, [("PO",), ("PO", "FS")], betas, ses)

        assert actual.columns.tolist() == list(bbf.BIM_COLS) + [
            "BETA_PO", "SE_PO", "P_PO", "BETA_PO_FS", "SE_PO_FS", "P_PO_FS"]
        assert actual["RSID"].tolist() == ["rs1", "rs2"]
        assert actual["BETA_PO_FS"].tolist() == [0.4, -0.2]
        assert pd.isna(actual["BETA_PO"].iloc[1]) and pd.isna(actual["P_PO"].iloc[1])
        assert float(actual["P_PO"].iloc[0]) == pytest.approx(2 * stats.norm.sf(5.0))


class TestWriteResults:

    def test__tab_separated__missing_values_written_as_na(self, tmp_path):
        bim = TestBuildOutput.BIM
        results = sut.build_output(bim, [("FS",)], np.array([[0.5], [np.nan]]),
                                   np.array([[0.1], [np.nan]]))
        path = os.path.join(tmp_path, "out.res")

        sut.write_results(results, path)

        lines = open(path).read().splitlines()
        assert lines[0].split("\t") == list(bbf.BIM_COLS) + ["BETA_FS", "SE_FS", "P_FS"]
        assert lines[2].split("\t")[-3:] == ["NA", "NA", "NA"]
        assert "nan" not in open(path).read().lower().replace("na\t", "").replace("\tna", "")

