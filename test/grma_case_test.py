"""
grma_lib.grma() on every case in tcs.py, at every relatedness degree.

    TestCaseEndToEnd     against grma_mock's independent implementation of the specification
    TestCaseInvariance   changes to the inputs that must leave the results unchanged, or (for
                         recoded alleles) exactly negated
    TestCaseEquivalence  cases that must give exactly the results of another
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma_lib as sut
import grma_mock
import tc_parental_snipar
import tcs


PALINDROMIC = ({"A", "T"}, {"C", "G"})


def run(key, degree, tmp_path, case=None, **extra):
    return sut.grma(**{**tcs.grma_kwargs(key, degree, tmp_path, case=case), **extra})


def value_columns(results):
    return [c for c in results.columns if c not in bbf.BIM_COLS]


def numeric(column):
    return pd.to_numeric(column.astype(object).where(column.notna(), np.nan))


def assert_same_results(actual, expected, rows=None):
    """The same estimates, SEs and p-values, up to floating-point rounding (see grma_lib_test)"""
    assert value_columns(actual) == value_columns(expected)
    for column in value_columns(actual):
        a, e = numeric(actual[column]), numeric(expected[column])
        if rows is not None:
            a, e = a[rows], e[rows]
        assert np.allclose(a, e, rtol=1e-9, atol=1e-12, equal_nan=True), column


def copy_case(key):
    case = dict(tcs.get(key))
    for name, value in case.items():
        if isinstance(value, pd.DataFrame):
            case[name] = value.copy()
        elif isinstance(value, np.ndarray):
            case[name] = value.copy()
        elif isinstance(value, dict):
            case[name] = {k : np.array(v, copy=True) for k, v in value.items()}
    return case


# --------------------------------------------------------------------------------------------------
class TestCaseEndToEnd:

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__results__the_mock_implementations(self, key, degree, tmp_path):
        actual = run(key, degree, tmp_path)
        rsids, expected = grma_mock.mock_grma(**tcs.mock_kwargs(key, degree))

        assert actual["RSID"].tolist() == rsids
        assert [c[len("BETA_"):] for c in actual.columns if c.startswith("BETA_")] == list(expected)
        for label, (betas, ses) in expected.items():
            assert np.allclose(actual["BETA_" + label], betas, rtol=1e-8, atol=1e-12,
                               equal_nan=True), label
            assert np.allclose(actual["SE_" + label], ses, rtol=1e-8, atol=1e-12,
                               equal_nan=True), label

    @pytest.mark.parametrize("degree", tcs.REL_DEGREES)
    def test__snipar_case__exercises_the_po_regression(self, degree, tmp_path):
        # A case meant to test the PO regression must actually produce PO estimates -- at most
        # SNPs, and in particular at the one snipar's file lacks, where only the children with
        # both parents genotyped can contribute
        actual = run("PARENTAL_SNIPAR", degree, tmp_path)

        missing_from_snipar = f"rs{tc_parental_snipar.MISSING_FROM_SNIPAR}"
        assert actual["BETA_PO"].notna().sum() >= len(actual) - 2
        assert actual.loc[actual["RSID"] == missing_from_snipar, "BETA_PO"].notna().all()


# --------------------------------------------------------------------------------------------------
class TestCaseInvariance:

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__phenotypes_shifted__same_results(self, key, degree, tmp_path):
        case = copy_case(key)
        for name in ("FAM_DF", "PHENO_DF"):
            if case[name] is not None:
                phenos = case[name]["PHENO"].astype(float)
                case[name]["PHENO"] = np.where(phenos == -9.0, -9.0, phenos + 7.5)

        assert_same_results(run(key, degree, tmp_path, case=case), run(key, degree, tmp_path))

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__alleles_swapped__those_estimates_negated(self, key, degree, tmp_path):
        # Swap A1 and A2 in the bim file and recode the genotypes to match.  snipar's file is left
        # as it is, so where it has the SNP, the flip is applied in reading it.  Palindromic SNPs
        # are left alone: their strand cannot be read from their letters.
        case = copy_case(key)
        bim = case["BIM_DF"]
        swap = [s for s in range(0, len(bim), 2)
                if {bim["A1"].iloc[s], bim["A2"].iloc[s]} not in PALINDROMIC]
        bim.loc[swap, ["A1", "A2"]] = bim.loc[swap, ["A2", "A1"]].to_numpy()
        case["G"][swap] = 2.0 - case["G"][swap]

        actual = run(key, degree, tmp_path, case=case)
        expected = run(key, degree, tmp_path)

        swapped = expected["RSID"].isin(bim["RSID"].iloc[swap]).to_numpy()
        for column in value_columns(expected):
            a, e = numeric(actual[column]), numeric(expected[column])
            if column.startswith("BETA_"):
                e = np.where(swapped, -e, e)
            assert np.allclose(a, e, rtol=1e-9, atol=1e-12, equal_nan=True), column

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__samples_renamed__same_results(self, key, degree, tmp_path):
        case = copy_case(key)
        fam = case["FAM_DF"]
        new_iid = {str(i) : f"person_{n}" for n, i in enumerate(fam["IID"])}
        new_fid = {str(f) : f"family_{n}" for n, f in enumerate(pd.unique(fam["FID"]))}

        def rename(df, fid_col, iid_col):
            df[fid_col] = df[fid_col].astype(str).map(new_fid)
            df[iid_col] = df[iid_col].astype(str).map(new_iid)

        rename(fam, "FID", "IID")
        for col in ("IIDF", "IIDM"):
            fam[col] = fam[col].astype(str).map(lambda x: new_iid.get(x, x))
        rename(case["REL_DF"], "FID1", "ID1")
        rename(case["REL_DF"], "FID2", "ID2")
        for name in ("PHENO_DF", "COVAR_DF", "ID_LIST"):
            if case[name] is not None:
                rename(case[name], "FID", "IID")
        if case["SNIPAR"] is not None:
            pedigree = case["SNIPAR"]["pedigree"].astype(object)
            for col in (1, 2, 3):
                pedigree[:, col] = [new_iid.get(x, x) for x in pedigree[:, col]]
            case["SNIPAR"]["pedigree"] = pedigree.astype(str)

        assert_same_results(run(key, degree, tmp_path, case=case), run(key, degree, tmp_path))

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__snps_renamed__same_results(self, key, degree, tmp_path):
        case = copy_case(key)
        new_rsid = {r : f"snp_{n}" for n, r in enumerate(case["BIM_DF"]["RSID"].astype(str))}
        case["BIM_DF"]["RSID"] = case["BIM_DF"]["RSID"].astype(str).map(new_rsid)
        if case["SNP_LIST"] is not None:
            case["SNP_LIST"]["RSID"] = case["SNP_LIST"]["RSID"].astype(str).map(new_rsid)
        if case["SNIPAR"] is not None:
            values = case["SNIPAR"]["bim_values"].astype(object)
            values[:, 1] = [new_rsid.get(r, r) for r in values[:, 1]]
            case["SNIPAR"]["bim_values"] = values.astype(str)

        assert_same_results(run(key, degree, tmp_path, case=case), run(key, degree, tmp_path))

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__singletons_added__same_results(self, key, degree, tmp_path):
        # Samples related to nobody are in no regression, so must not affect Omega's variance or
        # the phenotypes' centering.  The original implementation failed this.
        case = copy_case(key)
        rng = np.random.default_rng(31)
        new = [f"single_{n}" for n in range(3)]
        phenos = [5.0, -4.0, 9.0]
        fam = case["FAM_DF"]
        case["FAM_DF"] = pd.concat([fam.astype({"FID" : str, "IID" : str}), pd.DataFrame({
            "FID" : new, "IID" : new, "IIDF" : "0", "IIDM" : "0", "SEX" : "0",
            "PHENO" : phenos})], ignore_index=True)
        for name, extra in (("PHENO_DF", {"PHENO" : phenos}), ("ID_LIST", {})):
            if case[name] is not None:
                case[name] = pd.concat([case[name].astype({"FID" : str, "IID" : str}),
                                        pd.DataFrame({"FID" : new, "IID" : new, **extra})],
                                       ignore_index=True)
        if case["COVAR_DF"] is not None:
            covars = {c : rng.normal(size=3) for c in case["COVAR_DF"].columns[2:]}
            case["COVAR_DF"] = pd.concat([case["COVAR_DF"].astype({"FID" : str, "IID" : str}),
                                          pd.DataFrame({"FID" : new, "IID" : new, **covars})],
                                         ignore_index=True)
        case["G"] = np.column_stack([case["G"],
                                     rng.integers(0, 3, (case["G"].shape[0], 3)).astype(float)])

        assert_same_results(run(key, degree, tmp_path, case=case), run(key, degree, tmp_path))

    @pytest.mark.parametrize("key, degree", tcs.params())
    def test__one_snp_per_block__same_results(self, key, degree, tmp_path):
        assert_same_results(run(key, degree, tmp_path, snps_per_block=1),
                            run(key, degree, tmp_path))


# --------------------------------------------------------------------------------------------------
class TestCaseEquivalence:

    @pytest.mark.parametrize("key, degree", tcs.params(tcs.equivalent_keys()))
    def test__same_results_as_the_equivalent_case(self, key, degree, tmp_path):
        case = tcs.get(key)
        actual = run(key, degree, tmp_path)
        expected = run(case["EQUIVALENT_TO"], degree, tmp_path)

        excepted = actual["RSID"].isin(case["EQUIVALENT_EXCEPT"] or []).to_numpy()
        assert_same_results(actual, expected, rows=~excepted)

    @pytest.mark.parametrize("degree", tcs.REL_DEGREES)
    def test__hidden_mother__differs_where_snipar_lacks_the_snp(self, degree, tmp_path):
        # The exception is real: with the mother hidden, her children have no residual where snipar
        # lacks the SNP, so the PO estimate there changes
        actual = run("PARENTAL_SNIPAR_HIDDEN_MOTHER", degree, tmp_path)
        expected = run("PARENTAL_SNIPAR", degree, tmp_path)

        at = (actual["RSID"] == f"rs{tc_parental_snipar.MISSING_FROM_SNIPAR}").to_numpy()
        assert not np.allclose(numeric(actual["BETA_PO"])[at], numeric(expected["BETA_PO"])[at],
                               equal_nan=True)
