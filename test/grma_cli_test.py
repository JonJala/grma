"""
Tests of the command line, grma.py: real files in a temporary directory, run through main_func().

Each result file must be exactly what writing grma_lib.grma()'s results for the same inputs
gives, so these tests check what the command line adds -- finding filesets, pairing imputed files
with them, passing options on, writing the files, failing cleanly -- not the analysis itself.

The data is PARENTAL_SNIPAR's, split into two chromosomes: rs0-rs5 on chromosome 1 and rs6-rs11 on
chromosome 2, each a fileset with its own snipar file.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma
import grma_io
import grma_lib
import snipar_hdf5
import tc_parental_snipar as data

CHROMOSOMES = {"1" : list(range(0, 6)), "2" : list(range(6, 12))}


def write_dataset(directory, fam=data.FAM_DF):
    """Two filesets, one snipar file per chromosome, and a KING file.  The snipar files are named so
    that their alphabetical order is the reverse of their chromosomes', so that pairing them with
    the filesets by name would go wrong."""
    paths = {"rel" : os.path.join(directory, "rel.kin0"), "bfiles" : {}, "imputed" : {}}
    data.REL_DF.to_csv(paths["rel"], sep="\t", index=False)
    snipar_rows = [s for s in range(data.NUM_SNPS) if s != data.MISSING_FROM_SNIPAR]
    for chrom, snps in CHROMOSOMES.items():
        prefix = os.path.join(directory, f"data_chr{chrom}")
        bim = data.BIM_DF.iloc[snps]
        bbf.write_fam_file(prefix + ".fam", fam["FID"].to_numpy(), fam["IID"].to_numpy(),
                           pheno=fam["PHENO"].to_numpy())
        bbf.write_bim_file(prefix + ".bim", np.full(len(snps), chrom), bim["RSID"].to_numpy(),
                           bim["BP"].to_numpy(), bim["A1"].to_numpy(), bim["A2"].to_numpy())
        bbf.write_bed_file(prefix + ".bed", data.G[snps])
        paths["bfiles"][chrom] = prefix

        in_file = [n for n, s in enumerate(snipar_rows) if s in snps]
        snipar = dict(data.SNIPAR)
        snipar["imputed_gts"] = data.SNIPAR["imputed_gts"][:, in_file]
        snipar["num_obs_par_alleles"] = data.SNIPAR["num_obs_par_alleles"][:, in_file]
        snipar["standard_f"] = data.SNIPAR["standard_f"][in_file]
        bim_values = data.SNIPAR["bim_values"][in_file].copy()
        bim_values[:, 0] = chrom
        snipar["bim_values"] = bim_values
        name = {"1" : "imp_z", "2" : "imp_a"}[chrom]
        path = os.path.join(directory, name + snipar_hdf5.SNIPAR_FILE_SUFFIX)
        snipar_hdf5.write_snipar_file(path, **snipar)
        paths["imputed"][chrom] = path
    return paths


def read(path):
    with open(path) as f:
        return f.read()


def run_cli(*args):
    grma.main_func(["grma.py", *[str(a) for a in args]])


def expected_file(directory, paths, chrom, **options):
    """grma()'s results for one fileset, written as the command line writes them"""
    prefix = paths["bfiles"][chrom]
    results = grma_lib.grma(rel_file=paths["rel"], bed_file=prefix + ".bed",
                              bim_file=prefix + ".bim", fam_file=prefix + ".fam", **options)
    path = os.path.join(directory, f"expected_chr{chrom}.res")
    grma_io.write_results(results, path)
    return read(path)


def result_file(out_prefix, chrom):
    return read(f"{out_prefix}_data_chr{chrom}.res")


# --------------------------------------------------------------------------------------------------
class TestCommandLine:

    def test__two_filesets_with_imputed_parents__each_paired_by_chromosome(self, tmp_path):
        paths = write_dataset(tmp_path)
        out = os.path.join(tmp_path, "out")

        run_cli("--bfile", os.path.join(tmp_path, "data_chr*"), "--rel-pedigree", paths["rel"],
                "--imputed-parents", os.path.join(tmp_path, "imp_*.hdf5"), "--rel-thresh", "3",
                "--out", out)

        for chrom in CHROMOSOMES:
            expected = expected_file(tmp_path, paths, chrom, rel_degree=3,
                                     imputed_parental_file=paths["imputed"][chrom])
            assert result_file(out, chrom) == expected
        assert "BETA_PO_FS_2ND_3RD" in result_file(out, "1").splitlines()[0]

    def test__options__passed_on_to_the_analysis(self, tmp_path):
        paths = write_dataset(tmp_path)
        fam = data.FAM_DF
        pheno = os.path.join(tmp_path, "pheno.txt")
        pd.DataFrame({0 : fam["FID"], 1 : fam["IID"], 2 : fam["PHENO"] * 2 + 1}).to_csv(
            pheno, sep=" ", header=False, index=False)
        covar = os.path.join(tmp_path, "covar.txt")
        pd.DataFrame({0 : fam["FID"], 1 : fam["IID"], 2 : np.linspace(20, 70, len(fam))}).to_csv(
            covar, sep=" ", header=False, index=False)
        ids = os.path.join(tmp_path, "ids.txt")
        fam[fam["IID"] != "solo"][["FID", "IID"]].to_csv(ids, sep=" ", header=False, index=False)
        snps = os.path.join(tmp_path, "snps.txt")
        with open(snps, "w") as f:
            f.write("rs1\nrs4\nrs7\nrs8\n")
        out = os.path.join(tmp_path, "out")

        run_cli("--bfile", os.path.join(tmp_path, "data_chr*"), "--rel-pedigree", paths["rel"],
                "--imputed-parents", os.path.join(tmp_path, "imp_*.hdf5"), "--rel-thresh", "2",
                "--pheno", pheno, "--covar", covar, "--id-list", ids, "--snp-list", snps,
                "--snps-per-block", "1", "--out", out)

        for chrom in CHROMOSOMES:
            expected = expected_file(tmp_path, paths, chrom, rel_degree=2, pheno_file=pheno,
                                     covar_file=covar, id_list=ids, snp_list=snps,
                                     snps_per_block=1,
                                     imputed_parental_file=paths["imputed"][chrom])
            assert result_file(out, chrom) == expected
        assert len(result_file(out, "1").splitlines()) == 3        # Header, rs1, rs4

    def test__rel_thresh_fs__the_same_as_1(self, tmp_path):
        paths = write_dataset(tmp_path)
        common = ["--bfile", os.path.join(tmp_path, "data_chr1"), "--rel-pedigree", paths["rel"]]

        run_cli(*common, "--rel-thresh", "FS", "--out", os.path.join(tmp_path, "fs"))
        run_cli(*common, "--rel-thresh", "1", "--out", os.path.join(tmp_path, "one"))

        assert (result_file(os.path.join(tmp_path, "fs"), "1") ==
                result_file(os.path.join(tmp_path, "one"), "1"))

    def test__without_imputed_parents__no_po_regression(self, tmp_path):
        paths = write_dataset(tmp_path)
        out = os.path.join(tmp_path, "out")

        run_cli("--bfile", paths["bfiles"]["1"], "--rel-pedigree", paths["rel"],
                "--rel-thresh", "3", "--out", out)

        header = result_file(out, "1").splitlines()[0].split("\t")
        assert header[6:] == [f"{c}_{label}" for label in ("FS", "FS_2ND", "FS_2ND_3RD")
                              for c in ("BETA", "SE", "P")]
        assert result_file(out, "1") == expected_file(tmp_path, paths, "1", rel_degree=3)

    def test__missing_estimates__written_as_na(self, tmp_path):
        paths = write_dataset(tmp_path)
        out = os.path.join(tmp_path, "out")

        run_cli("--bfile", paths["bfiles"]["2"], "--rel-pedigree", paths["rel"],
                "--imputed-parents", paths["imputed"]["2"], "--out", out)

        rows = [line.split("\t") for line in result_file(out, "2").splitlines()]
        assert any("NA" in row for row in rows[1:])
        assert not any(value.lower() == "nan" for row in rows for value in row)


# --------------------------------------------------------------------------------------------------
class TestCommandLineErrors:

    def test__fam_files_disagree__exits_with_the_reason_logged(self, tmp_path):
        paths = write_dataset(tmp_path)
        changed = data.FAM_DF.copy()
        changed.loc[0, "PHENO"] = 99.0
        bbf.write_fam_file(paths["bfiles"]["2"] + ".fam", changed["FID"].to_numpy(),
                           changed["IID"].to_numpy(), pheno=changed["PHENO"].to_numpy())
        out = os.path.join(tmp_path, "out")

        with pytest.raises(SystemExit) as exit_info:
            run_cli("--bfile", os.path.join(tmp_path, "data_chr*"), "--rel-pedigree", paths["rel"],
                    "--out", out)

        assert exit_info.value.code == 1
        assert not os.path.exists(f"{out}_data_chr1.res")
        assert "fam" in read(out + ".log").lower()

    def test__fewer_imputed_files_than_filesets__exits_saying_so(self, tmp_path):
        paths = write_dataset(tmp_path)
        out = os.path.join(tmp_path, "out")

        with pytest.raises(SystemExit) as exit_info:
            run_cli("--bfile", os.path.join(tmp_path, "data_chr*"), "--rel-pedigree", paths["rel"],
                    "--imputed-parents", paths["imputed"]["1"], "--out", out)

        assert exit_info.value.code == 1
        assert "number of imputed parental genotype files (1)" in read(out + ".log")

    def test__two_imputed_files_for_one_chromosome__exits_naming_it(self, tmp_path):
        # The right number of files, but both for chromosome 1, so chromosome 2 has none
        paths = write_dataset(tmp_path)
        copy = os.path.join(tmp_path, "copy_of_chr1" + snipar_hdf5.SNIPAR_FILE_SUFFIX)
        with open(paths["imputed"]["1"], "rb") as source, open(copy, "wb") as target:
            target.write(source.read())
        out = os.path.join(tmp_path, "out")

        with pytest.raises(SystemExit) as exit_info:
            run_cli("--bfile", os.path.join(tmp_path, "data_chr*"), "--rel-pedigree", paths["rel"],
                    "--imputed-parents", paths["imputed"]["1"], copy, "--out", out)

        assert exit_info.value.code == 1
        assert "covers the same chromosome: ['1']" in read(out + ".log")

    def test__imputed_parents_matching_nothing__usage_error(self, tmp_path):
        paths = write_dataset(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            run_cli("--bfile", paths["bfiles"]["1"], "--rel-pedigree", paths["rel"],
                    "--imputed-parents", os.path.join(tmp_path, "nothing_*.hdf5"),
                    "--out", os.path.join(tmp_path, "out"))

        assert exit_info.value.code == 2
