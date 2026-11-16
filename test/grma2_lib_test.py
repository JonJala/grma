"""
Tests of grma2_lib: GRMA's planning and statistics.

So far: the sample table, the relatedness graph, duplicate samples, parents, groups, Omega, the
SNPs to analyse, the four regressions, the covariance of their estimates, their meta-analysis,
and grma2(), which runs it all.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import bedbimfam as bbf
import grma2_io
import grma2_lib as sut
import snipar_hdf5


def fam(iids, phenos, sexes=None):
    """A fam frame as grma2_io.read_fam() returns it, everyone in family F"""
    return grma2_io.read_fam(pd.DataFrame({
        bbf.FAM_FID_COL : "F", bbf.FAM_IID_COL : iids, bbf.FAM_IIDF_COL : "0",
        bbf.FAM_IIDM_COL : "0", bbf.FAM_SEX_COL : sexes if sexes is not None else "0",
        bbf.FAM_PHENO_COL : phenos}))


def samples_of(iids, in_analysis=None, sexes=None):
    """A sample table for the given IIDs; in_analysis lists who is in it (default everyone)"""
    phenos = [1.0 if in_analysis is None or iid in in_analysis else -9.0 for iid in iids]
    return sut.build_sample_table(fam(iids, phenos, sexes))


def king(rows):
    """A KING frame from (iid1, iid2, inftype) rows, everyone in family F"""
    return grma2_io.read_king(pd.DataFrame(
        [("F", a, "F", b, t) for a, b, t in rows], columns=list(grma2_io.KING_COLS)))


def graph_of(samples, rows):
    return sut.build_relatedness_graph(samples, king(rows))


def snipar(records, families):
    """snipar metadata holding a pedigree and a family list, as read_snipar_metadata() returns them

    records are (snipar FID, IID, father, mother, has_father, has_mother); the flags are booleans.
    """
    pedigree = np.array([[f, i, d, m, str(hf), str(hm)] for f, i, d, m, hf, hm in records])
    return {snipar_hdf5.SNIPAR_PEDIGREE_KEY : pedigree,
            snipar_hdf5.SNIPAR_FAMILIES_KEY : np.array(families)}


def resolved(iids, king_rows, in_analysis=None, sexes=None):
    """The sample table and graph, through resolve_duplicates()"""
    samples = samples_of(iids, in_analysis, sexes)
    return sut.resolve_duplicates(samples, graph_of(samples, king_rows))


def pairs(graph):
    """The graph as a set of (row 1, row 2, relationship), for order-free comparison"""
    return set(zip(graph[sut.GRAPH_ROW1_COL], graph[sut.GRAPH_ROW2_COL],
                   graph[sut.GRAPH_RELATIONSHIP_COL]))


def ids(iids):
    return pd.DataFrame({bbf.FAM_FID_COL : "F", bbf.FAM_IID_COL : iids})


# --------------------------------------------------------------------------------------------------
class TestBuildSampleTable:

    def test__fam_only__one_row_per_sample_in_fam_order(self):
        actual = sut.build_sample_table(fam(["c", "a", "b"], [1.0, 2.0, 3.0]))

        assert actual[sut.SAMPLE_IID_COL].tolist() == ["c", "a", "b"]
        assert actual[sut.SAMPLE_BED_COL].tolist() == [0, 1, 2]
        assert actual[sut.SAMPLE_PHENO_COL].tolist() == [1.0, 2.0, 3.0]
        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, True, True]

    def test__missing_fam_phenotype__no_phenotype_not_in_analysis(self):
        actual = sut.build_sample_table(fam(["a", "b"], [1.0, -9.0]))

        assert actual[sut.SAMPLE_HAS_PHENO_COL].tolist() == [True, False]
        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False]

    def test__phenotype_file__is_the_only_source(self):
        # b's fam phenotype is ignored once a phenotype file is given, since the file does not
        # list b; c is listed with a missing value
        pheno = grma2_io.read_pheno(ids(["a", "c"]).assign(**{bbf.FAM_PHENO_COL : [5.0, -9.0]}))

        actual = sut.build_sample_table(fam(["a", "b", "c"], [1.0, 2.0, 3.0]), pheno=pheno)

        assert actual[sut.SAMPLE_PHENO_COL].iloc[0] == 5.0
        assert actual[sut.SAMPLE_HAS_PHENO_COL].tolist() == [True, False, False]
        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False, False]

    def test__covariates__aligned_and_incomplete_ones_excluded(self):
        # Listed out of order; b has a missing covariate; c is absent from the file
        covar = grma2_io.read_covar(ids(["b", "a"]).assign(age=[np.nan, 40.0], pc1=[0.1, 0.2]))

        actual = sut.build_sample_table(fam(["a", "b", "c"], [1.0, 2.0, 3.0]), covar=covar)

        assert actual["COVAR_1"].iloc[0] == 40.0 and actual["COVAR_2"].iloc[0] == 0.2
        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False, False]
        # A covariate's absence affects only the analysis, not whether there is a phenotype
        assert actual[sut.SAMPLE_HAS_PHENO_COL].tolist() == [True, True, True]

    def test__id_list__restricts_the_analysis(self):
        actual = sut.build_sample_table(fam(["a", "b", "c"], [1.0, 2.0, 3.0]),
                                        id_list=grma2_io.read_id_list(ids(["c", "a"])))

        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False, True]

    def test__all_filters_together__a_sample_must_pass_every_one(self):
        pheno = grma2_io.read_pheno(ids(["a", "b", "c", "d"]).assign(
            **{bbf.FAM_PHENO_COL : [1.0, 2.0, 3.0, -9.0]}))
        covar = grma2_io.read_covar(ids(["a", "b", "d"]).assign(age=[1.0, 2.0, 4.0]))

        id_list = grma2_io.read_id_list(ids(["a", "c", "d"]))

        actual = sut.build_sample_table(fam(["a", "b", "c", "d"], [0.0] * 4), pheno=pheno,
                                        covar=covar, id_list=id_list)

        # a: passes all; b: not in ID list; c: no covariates; d: missing phenotype
        assert actual[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False, False, False]

    @pytest.mark.parametrize("which", ["fam", "pheno", "covar", "id_list"])
    def test__repeated_sample_in_any_input__raises_naming_it(self, which):
        inputs = dict(fam=fam(["a", "b"], [1.0, 2.0]), pheno=None, covar=None, id_list=None)
        repeated = ids(["a", "b", "b"])
        if which == "fam":
            inputs["fam"] = fam(["a", "b", "b"], [1.0, 2.0, 3.0])
        elif which == "pheno":
            inputs["pheno"] = grma2_io.read_pheno(repeated.assign(**{bbf.FAM_PHENO_COL : 1.0}))
        elif which == "covar":
            inputs["covar"] = grma2_io.read_covar(repeated.assign(age=1.0))
        else:
            inputs["id_list"] = grma2_io.read_id_list(repeated)

        with pytest.raises(ValueError, match="more than once") as error:
            sut.build_sample_table(**inputs)

        assert "F b" in str(error.value)

    def test__nobody_left_in_the_analysis__raises(self):
        with pytest.raises(ValueError, match="No sample is left"):
            sut.build_sample_table(fam(["a", "b"], [-9.0, -9.0]))

    def test__sex__carried_as_text(self):
        actual = sut.build_sample_table(fam(["a", "b"], [1.0, 2.0], sexes=["1", "2"]))

        assert actual[sut.SAMPLE_SEX_COL].tolist() == ["1", "2"]


# --------------------------------------------------------------------------------------------------
class TestBuildRelatednessGraph:

    def test__pairs__as_rows_in_order_with_relationships(self):
        samples = samples_of(["a", "b", "c", "d", "e", "f"])

        actual = graph_of(samples, [("b", "a", "PO"), ("a", "c", "FS"), ("d", "a", "2nd"),
                                    ("a", "e", "3rd"), ("f", "a", "Dup/MZTwin")])

        assert pairs(actual) == {(0, 1, sut.RELATIONSHIP_PO), (0, 2, sut.RELATIONSHIP_FS),
                                 (0, 3, sut.RELATIONSHIP_2ND), (0, 4, sut.RELATIONSHIP_3RD),
                                 (0, 5, sut.RELATIONSHIP_DUPLICATE)}

    def test__fourth_degree_and_unrelated__left_out(self):
        samples = samples_of(["a", "b", "c"])

        assert len(graph_of(samples, [("a", "b", "4th"), ("a", "c", "UN")])) == 0

    def test__sample_not_in_the_fam_file__pair_ignored(self, caplog):
        samples = samples_of(["a", "b"])

        with caplog.at_level(logging.INFO):
            actual = graph_of(samples, [("a", "b", "FS"), ("a", "stranger", "FS")])

        assert pairs(actual) == {(0, 1, sut.RELATIONSHIP_FS)}
        assert "not in the fam file" in caplog.text

    def test__neither_sample_in_the_analysis__pair_left_out(self):
        samples = samples_of(["a", "b", "c"], in_analysis={"a"})

        actual = graph_of(samples, [("b", "c", "FS"), ("a", "b", "PO")])

        # b-c can never be used; a-b can, since a is in the analysis
        assert pairs(actual) == {(0, 1, sut.RELATIONSHIP_PO)}

    def test__inftype_king_does_not_define__raises(self):
        samples = samples_of(["a", "b"])

        with pytest.raises(ValueError, match="does not define") as error:
            graph_of(samples, [("a", "b", "cousin")])

        assert "cousin" in str(error.value)

    def test__sample_paired_with_itself__raises(self):
        samples = samples_of(["a", "b"])

        with pytest.raises(ValueError, match="with themselves") as error:
            graph_of(samples, [("a", "a", "FS")])

        assert "a" in str(error.value)

    def test__pair_given_two_relationships__raises_naming_both(self):
        samples = samples_of(["a", "b"])

        with pytest.raises(ValueError, match="more than one relationship") as error:
            graph_of(samples, [("a", "b", "FS"), ("b", "a", "2nd")])

        assert "a-b" in str(error.value)
        assert "FS" in str(error.value) and "2nd" in str(error.value)

    def test__contradiction_between_two_unused_samples__ignored(self):
        samples = samples_of(["a", "b", "c"], in_analysis={"a"})

        actual = graph_of(samples, [("b", "c", "FS"), ("b", "c", "2nd"), ("a", "b", "PO")])

        assert pairs(actual) == {(0, 1, sut.RELATIONSHIP_PO)}

    def test__pair_repeated_either_way_round__kept_once_with_a_warning(self, caplog):
        samples = samples_of(["a", "b"])

        with caplog.at_level(logging.WARNING):
            actual = graph_of(samples, [("a", "b", "FS"), ("b", "a", "FS"), ("a", "b", "FS")])

        assert len(actual) == 1
        assert "more than once" in caplog.text and "a-b" in caplog.text


# --------------------------------------------------------------------------------------------------
class TestResolveDuplicates:

    def test__no_duplicates__nothing_dropped(self):
        samples, graph = resolved(["a", "b"], [("a", "b", "FS")])

        assert samples[sut.SAMPLE_DUPLICATE_OF_COL].tolist() == [-1, -1]
        assert pairs(graph) == {(0, 1, sut.RELATIONSHIP_FS)}

    def test__pair__the_earlier_kept_the_later_dropped(self):
        samples, graph = resolved(["a", "b", "c"], [("a", "b", "Dup/MZTwin"), ("b", "c", "FS"),
                                                    ("a", "c", "FS")])

        assert samples[sut.SAMPLE_DUPLICATE_OF_COL].tolist() == [-1, 0, -1]
        assert samples[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [True, False, True]
        # The duplicate pair itself, and every pair involving b, are gone
        assert pairs(graph) == {(0, 2, sut.RELATIONSHIP_FS)}

    def test__earlier_not_in_the_analysis__the_one_in_it_kept(self):
        samples, _ = resolved(["a", "b"], [("a", "b", "Dup/MZTwin")], in_analysis={"b"})

        assert samples[sut.SAMPLE_DUPLICATE_OF_COL].tolist() == [1, -1]
        assert samples[sut.SAMPLE_IN_ANALYSIS_COL].tolist() == [False, True]

    def test__chain_of_duplicates__one_cluster_one_kept(self):
        samples, _ = resolved(["a", "b", "c", "d"], [("c", "b", "Dup/MZTwin"),
                                                     ("b", "a", "Dup/MZTwin")])

        assert samples[sut.SAMPLE_DUPLICATE_OF_COL].tolist() == [-1, 0, 0, -1]

    def test__two_separate_pairs__one_kept_from_each(self):
        samples, _ = resolved(["a", "b", "c", "d"], [("a", "b", "Dup/MZTwin"),
                                                     ("c", "d", "Dup/MZTwin")])

        assert samples[sut.SAMPLE_DUPLICATE_OF_COL].tolist() == [-1, 0, -1, 2]

    def test__pair_left_with_neither_in_the_analysis__removed(self):
        # c is not in the analysis, and its only link to it was through b, now dropped
        samples, graph = resolved(["a", "b", "c"], [("a", "b", "Dup/MZTwin"), ("b", "c", "FS")],
                                  in_analysis={"a", "b"})

        assert len(graph) == 0


# --------------------------------------------------------------------------------------------------
class TestIdentifyParents:

    def parents_of(self, iids, king_rows, records, families, in_analysis=None, sexes=None):
        samples, graph = resolved(iids, king_rows, in_analysis, sexes)
        return sut.identify_parents(samples, graph, snipar(records, families))

    def test__no_snipar_file__nobody_with_a_warning(self, caplog):
        samples, graph = resolved(["a", "b"], [("a", "b", "PO")])

        with caplog.at_level(logging.WARNING):
            actual = sut.identify_parents(samples, graph, None)

        assert len(actual) == 0 and actual.columns.tolist() == list(sut.PARENTS_COLS)
        assert "parent-offspring regression is omitted" in caplog.text

    def test__both_parents_genotyped__both_rows_and_no_imputed_family(self):
        actual = self.parents_of(["dad", "mum", "kid"],
                                 [("dad", "kid", "PO"), ("mum", "kid", "PO")],
                                 [("S1", "kid", "dad", "mum", True, True)], ["OTHER"])

        assert actual.values.tolist() == [[2, 0, 1, -1]]

    def test__one_parent_genotyped__other_absent_and_imputed_family_found(self):
        actual = self.parents_of(["dad", "kid"], [("dad", "kid", "PO")],
                                 [("S1", "kid", "dad", "mum", True, False)], ["S0", "S1"])

        assert actual.values.tolist() == [[1, 0, -1, 1]]

    def test__child_not_in_the_analysis__left_out(self):
        actual = self.parents_of(["dad", "kid"], [("dad", "kid", "PO")],
                                 [("S1", "kid", "dad", "mum", True, False)], ["S1"],
                                 in_analysis={"dad"})

        assert len(actual) == 0

    def test__no_parent_genotyped__left_out(self):
        actual = self.parents_of(["kid"], [],
                                 [("S1", "kid", "dad", "mum", False, False)], ["S1"])

        assert len(actual) == 0

    def test__parent_zero__means_no_parent_even_with_a_sample_named_zero(self):
        # snipar writes "0" for a missing parent, as a fam file does.  A sample whose IID is "0"
        # must not be taken for that parent.
        actual = self.parents_of(["0", "mum", "kid"], [("mum", "kid", "PO")],
                                 [("S1", "kid", "0", "mum", False, True)], ["S1"])

        assert actual.values.tolist() == [[2, -1, 1, 0]]

    def test__parent_dropped_as_a_duplicate__replaced_by_the_one_kept(self):
        # mum and her twin are duplicates; the twin, later in the fam file, is dropped
        actual = self.parents_of(["mum", "twin", "kid"],
                                 [("mum", "twin", "Dup/MZTwin"), ("mum", "kid", "PO"),
                                  ("twin", "kid", "PO")],
                                 [("S1", "kid", "0", "twin", False, True)], ["S1"])

        assert actual.values.tolist() == [[2, -1, 0, 0]]

    def test__child_dropped_as_a_duplicate__not_replaced(self):
        actual = self.parents_of(["dad", "kid", "kid_dup"],
                                 [("kid", "kid_dup", "Dup/MZTwin"), ("dad", "kid", "PO"),
                                  ("dad", "kid_dup", "PO")],
                                 [("S1", "kid", "dad", "0", True, False),
                                  ("S1", "kid_dup", "dad", "0", True, False)], ["S1"])

        assert actual[sut.PARENTS_CHILD_COL].tolist() == [1]

    def test__iid_named_by_snipar_shared_by_two_samples__raises(self):
        samples = sut.build_sample_table(grma2_io.read_fam(pd.DataFrame({
            bbf.FAM_FID_COL : ["F1", "F2", "F1"], bbf.FAM_IID_COL : ["dad", "dad", "kid"],
            bbf.FAM_IIDF_COL : "0", bbf.FAM_IIDM_COL : "0", bbf.FAM_SEX_COL : "0",
            bbf.FAM_PHENO_COL : 1.0})))
        graph = sut.build_relatedness_graph(samples, king([]))

        with pytest.raises(ValueError, match="more than one sample") as error:
            sut.identify_parents(*sut.resolve_duplicates(samples, graph),
                                 snipar([("S1", "kid", "dad", "0", True, False)], ["S1"]))

        assert "dad" in str(error.value)

    def test__shared_iid_snipar_does_not_name__no_error(self):
        samples = sut.build_sample_table(grma2_io.read_fam(pd.DataFrame({
            bbf.FAM_FID_COL : ["F1", "F2", "F1", "F1"], bbf.FAM_IID_COL : ["x", "x", "dad", "kid"],
            bbf.FAM_IIDF_COL : "0", bbf.FAM_IIDM_COL : "0", bbf.FAM_SEX_COL : "0",
            bbf.FAM_PHENO_COL : 1.0})))
        rows = pd.DataFrame([("F1", "dad", "F1", "kid", "PO")], columns=list(grma2_io.KING_COLS))
        graph = sut.build_relatedness_graph(samples, grma2_io.read_king(rows))

        actual = sut.identify_parents(*sut.resolve_duplicates(samples, graph),
                                      snipar([("S1", "kid", "dad", "0", True, False)], ["S1"]))

        assert actual.values.tolist() == [[3, 2, -1, 0]]

    @pytest.mark.parametrize("has_father, has_mother", [(False, False), (True, True)])
    def test__snipar_disagrees_about_which_parents_are_genotyped__raises(self, has_father,
                                                                         has_mother):
        # Only dad is in the fam file; snipar says neither, or both
        with pytest.raises(ValueError, match="disagree about which parents") as error:
            self.parents_of(["dad", "kid"], [("dad", "kid", "PO")],
                            [("S1", "kid", "dad", "mum", has_father, has_mother)], ["S1"])

        assert "kid" in str(error.value)
        assert "same fileset" in str(error.value)

    def test__parent_king_does_not_confirm__raises_naming_child_and_parent(self):
        with pytest.raises(ValueError, match="does not classify as parent-offspring") as error:
            self.parents_of(["dad", "kid"], [("dad", "kid", "FS")],
                            [("S1", "kid", "dad", "0", True, False)], ["S1"])

        assert "kid (father dad)" in str(error.value)

    def test__child_listed_twice__raises(self):
        with pytest.raises(ValueError, match="more than once"):
            self.parents_of(["dad", "kid"], [("dad", "kid", "PO")],
                            [("S1", "kid", "dad", "0", True, False),
                             ("S1", "kid", "dad", "0", True, False)], ["S1"])

    def test__same_sex_parents__warning_only(self, caplog):
        with caplog.at_level(logging.WARNING):
            actual = self.parents_of(["p1", "p2", "kid"],
                                     [("p1", "kid", "PO"), ("p2", "kid", "PO")],
                                     [("S1", "kid", "p1", "p2", True, True)], ["S1"],
                                     sexes=["2", "2", "0"])

        assert len(actual) == 1
        assert "same recorded sex" in caplog.text

    def test__unknown_sexes__no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            self.parents_of(["p1", "p2", "kid"], [("p1", "kid", "PO"), ("p2", "kid", "PO")],
                            [("S1", "kid", "p1", "p2", True, True)], ["S1"])

        assert "same recorded sex" not in caplog.text

    def test__several_children__in_sample_table_order(self):
        actual = self.parents_of(["k2", "dad", "k1"], [("dad", "k1", "PO"), ("dad", "k2", "PO")],
                                 [("S1", "k1", "dad", "0", True, False),
                                  ("S1", "k2", "dad", "0", True, False)], ["S1"])

        assert actual[sut.PARENTS_CHILD_COL].tolist() == [0, 2]


# --------------------------------------------------------------------------------------------------
class TestAssignGroups:

    # Everyone is in family F.  kid has a genotyped parent and a sib; sib has only kid; nephew has
    # only a 2nd-degree relative; cousin only a 3rd-degree one; loner nobody; dad is kid's parent.
    IIDS = ["dad", "kid", "sib", "nephew", "cousin", "loner"]
    KING_ROWS = [("dad", "kid", "PO"), ("kid", "sib", "FS"), ("sib", "nephew", "2nd"),
                 ("nephew", "cousin", "3rd")]
    RECORDS = [("S1", "kid", "dad", "0", True, False)]

    def groups(self, rel_degree, in_analysis=None, king_rows=None):
        samples, graph = resolved(self.IIDS, king_rows or self.KING_ROWS, in_analysis)
        parents = sut.identify_parents(samples, graph, snipar(self.RECORDS, ["S1"]))
        actual = sut.assign_groups(samples, graph, parents, rel_degree=rel_degree)
        return dict(zip(actual[sut.SAMPLE_IID_COL], actual[sut.SAMPLE_GROUP_COL]))

    def test__first_group_qualified_for__exactly_one_each(self):
        actual = self.groups(rel_degree=3)

        # kid has a parent AND a sib: PO only.  dad has a child (PO pair) but no parent, sib or
        # 2nd/3rd-degree relative of his own: NOT_FOCAL.  sib has a 2nd-degree relative too: FS
        # only.
        assert actual == {"dad" : sut.GROUP_NOT_FOCAL, "kid" : sut.GROUP_PO, "sib" : sut.GROUP_FS,
                          "nephew" : sut.GROUP_2ND, "cousin" : sut.GROUP_3RD,
                          "loner" : sut.GROUP_NOT_FOCAL}

    def test__relative_outside_the_analysis__does_not_qualify(self):
        # nephew's 2nd-degree relative (sib) is out of the analysis, so nephew falls to his
        # 3rd-degree relative
        actual = self.groups(rel_degree=3, in_analysis={"dad", "kid", "nephew", "cousin", "loner"})

        assert actual["nephew"] == sut.GROUP_3RD
        assert actual["sib"] is None

    @pytest.mark.parametrize("rel_degree, nephew, cousin", [
        (0, sut.GROUP_NOT_FOCAL, sut.GROUP_NOT_FOCAL),
        (1, sut.GROUP_NOT_FOCAL, sut.GROUP_NOT_FOCAL),
        (2, sut.GROUP_2ND, sut.GROUP_NOT_FOCAL),
        (3, sut.GROUP_2ND, sut.GROUP_3RD)])
    def test__rel_degree__cuts_off_later_groups(self, rel_degree, nephew, cousin):
        actual = self.groups(rel_degree=rel_degree)

        # PO and FS run at every degree
        assert actual["kid"] == sut.GROUP_PO and actual["sib"] == sut.GROUP_FS
        assert actual["nephew"] == nephew and actual["cousin"] == cousin

    @pytest.mark.parametrize("rel_degree", [-1, 4])
    def test__rel_degree_out_of_range__raises(self, rel_degree):
        with pytest.raises(ValueError, match="between 0 and 3"):
            self.groups(rel_degree=rel_degree)

    def test__parent_outside_the_analysis__still_makes_the_child_po(self):
        actual = self.groups(rel_degree=3, in_analysis={"kid", "sib", "nephew", "cousin", "loner"})

        assert actual["kid"] == sut.GROUP_PO
        assert actual["dad"] is None


# --------------------------------------------------------------------------------------------------
# Omega

def planned(phenos, king_rows, records=(), families=(), rel_degree=3, covars=None):
    """The sample table through select_omega_sample(), with its graph and parents

    phenos maps each IID, in fam order, to its phenotype (-9 for missing); covars, where given, maps
    each IID to its covariate values.
    """
    iids = list(phenos)
    covar = None
    if covars is not None:
        rows = [["F", iid, *values] for iid, values in covars.items()]
        covar = grma2_io.read_covar(pd.DataFrame(rows))
    samples = sut.build_sample_table(fam(iids, list(phenos.values())), covar=covar)
    samples, graph = sut.resolve_duplicates(samples, sut.build_relatedness_graph(samples,
                                                                                 king(king_rows)))
    parents = sut.identify_parents(samples, graph,
                                   snipar(list(records), list(families)) if records else None)
    samples = sut.assign_groups(samples, graph, parents, rel_degree=rel_degree)
    return sut.select_omega_sample(samples, parents), graph, parents


def omega_members(samples):
    in_omega = samples[sut.SAMPLE_OMEGA_IDX_COL] != -1
    return samples.loc[in_omega, sut.SAMPLE_IID_COL].tolist()


class TestSelectOmegaSample:

    # dad is kid's genotyped parent but focal in no regression; kid is PO; sib is FS; loner is
    # related to nobody
    PHENOS = {"dad" : 1.0, "kid" : 2.0, "sib" : 3.0, "loner" : 4.0}
    KING_ROWS = [("dad", "kid", "PO"), ("dad", "sib", "PO"), ("kid", "sib", "FS")]
    RECORDS = [("S1", "kid", "dad", "0", True, False), ("S1", "sib", "dad", "0", True, False)]

    def test__focal_samples_and_parents_in__others_out(self):
        samples, _, _ = planned(self.PHENOS, self.KING_ROWS, self.RECORDS, ["S1"])

        assert omega_members(samples) == ["dad", "kid", "sib"]
        assert samples[sut.SAMPLE_OMEGA_IDX_COL].tolist() == [0, 1, 2, -1]

    def test__parent_outside_the_analysis__not_in_omega(self):
        samples, _, _ = planned({**self.PHENOS, "dad" : -9.0}, self.KING_ROWS, self.RECORDS,
                                ["S1"])

        assert omega_members(samples) == ["kid", "sib"]

    def test__sample_only_beyond_rel_degree__not_in_omega(self):
        phenos = {"a" : 1.0, "b" : 2.0, "c" : 3.0}
        samples, _, _ = planned(phenos, [("a", "b", "FS"), ("b", "c", "2nd")], rel_degree=1)

        # c's only relative is 2nd-degree, beyond the cut-off, so c is focal in nothing
        assert omega_members(samples) == ["a", "b"]

    def test__nobody_used_by_any_regression__raises(self):
        with pytest.raises(ValueError, match="No sample is used by any regression"):
            planned({"a" : 1.0, "b" : 2.0}, [])


class TestPreparePhenotypes:

    def test__no_covariates__centered_on_omega_samples_only(self):
        # loner is in the analysis but not in Omega, so must not affect the mean
        samples, _, _ = planned({"a" : 1.0, "b" : 5.0, "loner" : 100.0}, [("a", "b", "FS")])

        assert np.allclose(sut.prepare_phenotypes(samples), [-2.0, 2.0])

    def test__covariates__matches_a_separate_least_squares_solve(self):
        phenos = {"a" : 1.0, "b" : 4.0, "c" : 2.0, "d" : 7.0, "e" : 3.0}
        covars = {"a" : [30, 0.1], "b" : [40, -0.4], "c" : [35, 0.2], "d" : [50, 0.3],
                  "e" : [45, -0.1]}
        samples, _, _ = planned(phenos, [("a", "b", "FS"), ("c", "d", "FS"), ("d", "e", "FS")],
                                covars=covars)

        actual = sut.prepare_phenotypes(samples)

        y = np.array(list(phenos.values()))
        X = np.column_stack([np.array(list(covars.values()), dtype=float), np.ones(5)])
        expected = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
        assert np.allclose(actual, expected)

    def test__phenotypes_shifted_by_a_constant__unchanged(self):
        base = {"a" : 1.0, "b" : 5.0, "c" : 2.0}
        rows = [("a", "b", "FS"), ("b", "c", "FS")]
        samples, _, _ = planned(base, rows)
        shifted, _, _ = planned({k : v + 10.0 for k, v in base.items()}, rows)

        assert np.allclose(sut.prepare_phenotypes(samples), sut.prepare_phenotypes(shifted))

    def test__all_phenotypes_the_same__raises(self):
        samples, _, _ = planned({"a" : 3.0, "b" : 3.0}, [("a", "b", "FS")])

        with pytest.raises(ValueError, match="nothing to estimate"):
            sut.prepare_phenotypes(samples)

    def test__covariate_explains_everything__raises(self):
        samples, _, _ = planned({"a" : 2.0, "b" : 4.0}, [("a", "b", "FS")],
                                covars={"a" : [1.0], "b" : [2.0]})

        with pytest.raises(ValueError, match="nothing to estimate"):
            sut.prepare_phenotypes(samples)


class TestGetRelationshipCovariances:

    def test__mean_over_distinct_samples__each_counted_once(self):
        # p is in three PO pairs, with children valued 1, 2 and 6.  The mean counts p once:
        # (4 + 1 + 2 + 6) / 4 = 3.25.  Averaging over pair members instead would count p three
        # times and give 3.5.
        phenotypes = np.array([4.0, 1.0, 2.0, 6.0])
        first, second = np.array([0, 0, 0]), np.array([1, 2, 3])

        relationship = np.array([sut.RELATIONSHIP_PO] * 3)

        actual = sut.get_relationship_covariances(first, second, relationship, phenotypes)

        expected = np.mean([(4 - 3.25) * (1 - 3.25), (4 - 3.25) * (2 - 3.25),
                            (4 - 3.25) * (6 - 3.25)])
        assert actual == {sut.RELATIONSHIP_PO : pytest.approx(expected)}

    def test__each_relationship__its_own_mean_and_pairs(self):
        phenotypes = np.array([1.0, 3.0, 10.0, 20.0])
        actual = sut.get_relationship_covariances(
            np.array([0, 2]), np.array([1, 3]),
            np.array([sut.RELATIONSHIP_FS, sut.RELATIONSHIP_3RD]), phenotypes)

        assert actual[sut.RELATIONSHIP_FS] == pytest.approx((1 - 2) * (3 - 2))
        assert actual[sut.RELATIONSHIP_3RD] == pytest.approx((10 - 15) * (20 - 15))
        assert set(actual) == {sut.RELATIONSHIP_FS, sut.RELATIONSHIP_3RD}


class TestBuildOmega:

    @staticmethod
    def expected_unshrunk(samples, graph, phenotypes):
        """Omega built entry by entry, by a separate route"""
        n = len(phenotypes)
        omega = np.diag(np.full(n, np.var(phenotypes)))
        idx = samples[sut.SAMPLE_OMEGA_IDX_COL].to_numpy()
        by_relationship = {}
        for r1, r2, rel in zip(graph[sut.GRAPH_ROW1_COL], graph[sut.GRAPH_ROW2_COL],
                               graph[sut.GRAPH_RELATIONSHIP_COL]):
            if idx[r1] != -1 and idx[r2] != -1:
                by_relationship.setdefault(rel, []).append((idx[r1], idx[r2]))
        for rel, members in by_relationship.items():
            everyone = sorted({i for pair in members for i in pair})
            mean = np.mean(phenotypes[everyone])
            cov = np.mean([(phenotypes[i] - mean) * (phenotypes[j] - mean) for i, j in members])
            for i, j in members:
                omega[i, j] = omega[j, i] = cov
        return omega

    def test__definite_already__entries_as_built_by_hand(self):
        # Phenotypes chosen so that Omega is comfortably definite (smallest eigenvalue 0.43 x the
        # variance) and each relationship's covariance is clearly nonzero (0.23 to 0.44 x)
        phenos = {"dad" : -3.0, "kid" : -1.2, "sib" : -0.8, "cousin" : 0.9, "other" : 2.1}
        rows = [("dad", "kid", "PO"), ("dad", "sib", "PO"), ("kid", "sib", "FS"),
                ("sib", "cousin", "3rd"), ("cousin", "other", "FS")]
        records = [("S1", "kid", "dad", "0", True, False), ("S1", "sib", "dad", "0", True, False)]
        samples, graph, _ = planned(phenos, rows, records, ["S1"])
        phenotypes = sut.prepare_phenotypes(samples)

        actual = sut.build_omega(samples, graph, phenotypes).toarray()

        expected = self.expected_unshrunk(samples, graph, phenotypes)
        assert np.linalg.eigvalsh(expected).min() > 0      # So no shrinking was needed
        assert np.allclose(actual, expected)

    def test__relationship_beyond_rel_degree__still_in_omega(self):
        # At rel_degree 1, a and c are in FS regressions; their 3rd-degree link is not used for
        # residualization, but they are still phenotypically correlated
        phenos = {"a" : 1.0, "b" : 2.0, "c" : 4.0, "d" : 3.0}
        samples, graph, _ = planned(phenos, [("a", "b", "FS"), ("c", "d", "FS"),
                                             ("a", "c", "3rd")], rel_degree=1)

        actual = sut.build_omega(samples, graph, sut.prepare_phenotypes(samples)).toarray()

        assert actual[0, 2] != 0.0 and actual[0, 2] == actual[2, 0]

    def test__block_needing_shrinkage__shrunk_others_untouched(self):
        # Two families, each a parent with five children, at opposite phenotypic extremes: the PO
        # covariance is then nearly the variance, and a parent with m children gives eigenvalues
        # variance +/- covariance x sqrt(m), so each family's block is indefinite.  x-y is an
        # ordinary pair, which must be left as it is.
        kids1, kids2 = ["k1", "k2", "k3", "k4", "k5"], ["j1", "j2", "j3", "j4", "j5"]
        phenos = {"d1" : 10.0, "k1" : 9.5, "k2" : 10.5, "k3" : 10.0, "k4" : 9.8, "k5" : 10.2,
                  "d2" : -10.0, "j1" : -9.5, "j2" : -10.5, "j3" : -10.0, "j4" : -9.8,
                  "j5" : -10.2, "x" : -1.0, "y" : 1.0}
        rows = ([("d1", k, "PO") for k in kids1] + [("d2", k, "PO") for k in kids2] +
                [("x", "y", "FS")])
        records = ([("S1", k, "d1", "0", True, False) for k in kids1] +
                   [("S2", k, "d2", "0", True, False) for k in kids2])
        samples, graph, _ = planned(phenos, rows, records, ["S1", "S2"])
        phenotypes = sut.prepare_phenotypes(samples)

        actual = sut.build_omega(samples, graph, phenotypes).toarray()

        unshrunk = self.expected_unshrunk(samples, graph, phenotypes)
        family, pair = list(range(6)), [12, 13]
        assert np.linalg.eigvalsh(unshrunk[np.ix_(family, family)]).min() < 0
        assert np.linalg.eigvalsh(actual).min() > 0
        assert np.isclose(np.linalg.eigvalsh(actual[np.ix_(family, family)]).min(),
                          sut.DEFAULT_OMEGA_EPSILON, atol=1e-8)
        # Diagonal unchanged; the family's off-diagonal only scaled down; the pair untouched
        assert np.allclose(np.diag(actual), np.diag(unshrunk))
        assert 0 < actual[0, 1] / unshrunk[0, 1] < 1
        assert np.allclose(actual[np.ix_(pair, pair)], unshrunk[np.ix_(pair, pair)])

    def test__symmetric__and_zero_between_unrelated_samples(self):
        phenos = {"a" : 1.0, "b" : 2.0, "c" : 4.0, "d" : 3.0}
        samples, graph, _ = planned(phenos, [("a", "b", "FS"), ("c", "d", "FS")])

        actual = sut.build_omega(samples, graph, sut.prepare_phenotypes(samples)).toarray()

        assert np.allclose(actual, actual.T)
        assert actual[0, 2] == 0.0 and actual[1, 3] == 0.0


# --------------------------------------------------------------------------------------------------
# Copied unchanged from the original implementation's tests, with the function

class TestGetLambdaMin:

    @staticmethod
    def _banded_matrix(N: int, seed: int):
        rng = np.random.default_rng(seed)
        vals = rng.normal(size=N - 1)
        rows = np.concatenate([np.arange(N - 1), np.arange(1, N)])
        cols = np.concatenate([np.arange(1, N), np.arange(N - 1)])
        return sp.coo_array((np.concatenate([vals, vals]), (rows, cols)), shape=(N, N)).tocsr()

    @pytest.mark.parametrize("N", [sut.DENSE_LAMBDA_MIN_MAX_SIZE,
                                   sut.DENSE_LAMBDA_MIN_MAX_SIZE + 1,
                                   sut.DENSE_LAMBDA_MIN_MAX_SIZE + 100])
    def test__both_paths_agree_with_dense_truth(self, N):
        # get_lambda_min routes small matrices to a dense eigendecomposition and larger ones to
        # ARPACK.  Both sides of that boundary must agree with the dense answer, otherwise a
        # mistaken comparison would silently send everything down one path.
        B = self._banded_matrix(N, seed=99)

        expected = float(np.linalg.eigvalsh(B.toarray()).min())

        assert np.isclose(sut.get_lambda_min(B), expected, atol=1e-6)


    def test__repeated_calls_are_deterministic(self):
        # ARPACK starts from a random vector unless given one, which made repeated runs differ at
        # the ULP level.  get_lambda_min fixes the start vector; this pins that down.
        B = self._banded_matrix(sut.DENSE_LAMBDA_MIN_MAX_SIZE + 200, seed=7)

        results = [sut.get_lambda_min(B) for _ in range(5)]

        assert len(set(results)) == 1


class TestGetLambdaMinRouting:
    """Small matrices take the dense route, which is exact and faster at that size.  Both routes
    give the same answer, so only this test can tell which was taken."""

    def test__at_the_threshold__dense_route_without_arpack(self, monkeypatch):
        def no_arpack(*args, **kwargs):
            raise AssertionError("ARPACK was used for a matrix small enough for the dense route")
        monkeypatch.setattr(sut.spla, "eigsh", no_arpack)
        B = np.diag(np.linspace(-1.0, 1.0, sut.DENSE_LAMBDA_MIN_MAX_SIZE))

        assert np.isclose(sut.get_lambda_min(B), -1.0)

    def test__above_the_threshold__arpack_route(self, monkeypatch):
        calls = []
        real_eigsh = sut.spla.eigsh
        def counting_eigsh(*args, **kwargs):
            calls.append(1)
            return real_eigsh(*args, **kwargs)
        monkeypatch.setattr(sut.spla, "eigsh", counting_eigsh)
        B = sp.diags_array(np.linspace(-1.0, 1.0, sut.DENSE_LAMBDA_MIN_MAX_SIZE + 1)).tocsr()

        assert np.isclose(sut.get_lambda_min(B), -1.0, atol=1e-6)
        assert len(calls) > 0


class TestGetLambdaMinConvergence:
    """ARPACK can fail to converge.  get_lambda_min() retries with a larger workspace, and gives up
    with an error only after every attempt has failed."""

    BIG = sut.DENSE_LAMBDA_MIN_MAX_SIZE + 1

    def test__fails_then_converges__retried_with_a_larger_workspace(self, monkeypatch):
        workspaces = []
        real_eigsh = sut.spla.eigsh
        def flaky_eigsh(*args, **kwargs):
            workspaces.append(kwargs["ncv"])
            if len(workspaces) < 3:
                raise sut.spla.ArpackNoConvergence("no convergence", np.array([]), np.array([]))
            return real_eigsh(*args, **kwargs)
        monkeypatch.setattr(sut.spla, "eigsh", flaky_eigsh)
        B = sp.diags_array(np.linspace(-1.0, 1.0, self.BIG)).tocsr()

        assert np.isclose(sut.get_lambda_min(B), -1.0, atol=1e-6)
        assert len(workspaces) == 3 and workspaces == sorted(workspaces)

    def test__never_converges__raises(self, monkeypatch):
        def never(*args, **kwargs):
            raise sut.spla.ArpackNoConvergence("no convergence", np.array([]), np.array([]))
        monkeypatch.setattr(sut.spla, "eigsh", never)
        B = sp.diags_array(np.linspace(-1.0, 1.0, self.BIG)).tocsr()

        with pytest.raises(RuntimeError, match="failed to converge"):
            sut.get_lambda_min(B)


# Adapted from the original implementation's test: its input is built here directly, since the
# functions it used to build it belong to that implementation
class TestGetBlockAlphas:

    def test__gershgorin_screen_is_safe(self):
        # The Gershgorin screen skips blocks it can prove are definite.  Verify it never skips one
        # that needed shrinking, by checking every returned alpha against a direct per-block
        # eigendecomposition.
        rng = np.random.default_rng(1234)
        N = 60
        pheno = rng.normal(size=N)
        pairs = [(i, i + 1) for i in range(0, N - 1, 3)] + [(0, 3), (6, 9)]
        i1 = np.array([p[0] for p in pairs])
        i2 = np.array([p[1] for p in pairs])
        covariance = np.cov(np.concatenate([pheno[i1], pheno[i2]]),
                            np.concatenate([pheno[i2], pheno[i1]]), ddof=0)[0, 1]
        data = np.full(2 * len(pairs), covariance)
        off_diag = sp.coo_array((data, (np.concatenate([i1, i2]),
                                        np.concatenate([i2, i1]))), shape=(N, N)).tocsr()
        pheno_var = float(np.var(pheno))
        _, labels = csgraph.connected_components(off_diag, directed=False)

        alphas = sut.get_block_alphas(off_diag, pheno_var, labels)

        for c in range(labels.max() + 1):
            idx = np.flatnonzero(labels == c)
            if len(idx) < 2:
                continue
            block = off_diag[idx][:, idx].toarray()
            lmin = float(np.linalg.eigvalsh(block).min())
            expected = (min((sut.DEFAULT_OMEGA_EPSILON - pheno_var) / lmin, 1.0)
                        if lmin < 0.0 else 1.0)
            assert np.isclose(alphas[c], expected, atol=1e-6)

    def test__strongly_correlated_block__shrunk_to_the_floor(self):
        # A deliberately indefinite block alongside a definite one, so the screen and the shrinkage
        # are both exercised: the first must come back below 1 and land exactly on the floor
        off_diag = sp.csr_array(np.array([[0, 2, 2, 0, 0],
                                          [2, 0, 2, 0, 0],
                                          [2, 2, 0, 0, 0],
                                          [0, 0, 0, 0, 0.1],
                                          [0, 0, 0, 0.1, 0]], dtype=float))
        labels = np.array([0, 0, 0, 1, 1])

        alphas = sut.get_block_alphas(off_diag, 1.0, labels)

        assert alphas[0] < 1.0 and alphas[1] == 1.0
        shrunk = np.eye(3) + alphas[0] * off_diag[[0, 1, 2]][:, [0, 1, 2]].toarray()
        assert np.isclose(np.linalg.eigvalsh(shrunk).min(), sut.DEFAULT_OMEGA_EPSILON, atol=1e-12)

    def test__screened_block_not_indefinite__left_alone(self):
        # Defensive: an off-diagonal block has zeros on its diagonal, so any nonzero one has a
        # negative eigenvalue.  Only an all-zero block, with a variance at or below epsilon, passes
        # the screen without being indefinite -- and must then be left unshrunk.
        off_diag = sp.csr_array((2, 2))
        labels = np.array([0, 0])

        alphas = sut.get_block_alphas(off_diag, 0.0, labels)

        assert alphas.tolist() == [1.0]


# --------------------------------------------------------------------------------------------------
# The sibling, 2nd- and 3rd-degree regressions
#
# The fixture: sibships a (three) and b (two); n, a nephew of all of sibship a; half-sibs m1 and m2;
# c1, a cousin of b1; cousins c2 and c3.  a1 and b1 are cousins too, linking the families, so the
# regressions' samples overlap and their estimates are correlated.

RELATIVE_PHENOS = {"a1" : 1.2, "a2" : 0.4, "a3" : 1.9, "b1" : -0.7, "b2" : -1.5, "n" : 0.8,
                   "m1" : 2.2, "m2" : 1.1, "c1" : -0.3, "c2" : 0.6, "c3" : -2.0}
RELATIVE_ROWS = [("a1", "a2", "FS"), ("a1", "a3", "FS"), ("a2", "a3", "FS"), ("b1", "b2", "FS"),
                 ("n", "a1", "2nd"), ("n", "a2", "2nd"), ("n", "a3", "2nd"), ("m1", "m2", "2nd"),
                 ("c1", "b1", "3rd"), ("c2", "c3", "3rd"), ("a1", "b1", "3rd")]


def relative_plan(phenos=None, rows=None, rel_degree=3):
    samples, graph, _ = planned(phenos or RELATIVE_PHENOS, rows or RELATIVE_ROWS,
                                rel_degree=rel_degree)
    phenotypes = sut.prepare_phenotypes(samples)
    omega = sut.build_omega(samples, graph, phenotypes)
    regressions = sut.build_relative_regressions(samples, graph, phenotypes, omega,
                                                 rel_degree=rel_degree)
    cross = sut.build_cross_covariances(regressions, omega)
    return samples, graph, phenotypes, omega, regressions, cross


def genotypes_for(num_omega, num_snps=8, seed=4):
    """Random genotypes for the Omega samples.  The seed is one where every regression has some
    variation at every SNP, so every estimate exists; the no-variation case has its own test."""
    return np.random.default_rng(seed).integers(0, 3, size=(num_snps, num_omega)).astype(float)


def dense_regression(regression, omega, phenotypes, genotypes):
    """A regression's estimates by the textbook route: dense matrices, one pseudo-inverse"""
    A = regression.phenotype_map.toarray()
    Q = np.linalg.pinv(A @ omega.toarray() @ A.T, rcond=1e-10, hermitian=True)
    Z = genotypes @ A.T
    y = A @ phenotypes
    d = np.einsum("sf,fg,sg->s", Z, Q, Z)
    return (Z @ Q @ y) / d, d, Z, Q


class TestBuildResidualizationMatrix:

    def test__rows__own_share_and_relatives_shares_summing_to_zero(self):
        samples, graph, *_ = relative_plan()
        idx = dict(zip(samples[sut.SAMPLE_IID_COL], samples[sut.SAMPLE_OMEGA_IDX_COL]))

        focal, R = sut.build_residualization_matrix(samples, graph, sut.GROUP_FS)

        assert [samples[sut.SAMPLE_IID_COL][samples[sut.SAMPLE_OMEGA_IDX_COL] == f].item()
                for f in focal] == ["a1", "a2", "a3", "b1", "b2"]
        dense = R.toarray()
        assert np.allclose(dense.sum(axis=1), 0.0)
        a1 = dense[0]
        assert np.isclose(a1[idx["a1"]], 2 / 3)
        assert np.isclose(a1[idx["a2"]], -1 / 3) and np.isclose(a1[idx["a3"]], -1 / 3)
        assert np.count_nonzero(a1) == 3      # Not b1, though a1 and b1 are related: 3rd degree

    def test__relatives_from_any_group__used(self):
        # n's 2nd-degree relatives are all in the FS group, and still enter n's row
        samples, graph, *_ = relative_plan()
        idx = dict(zip(samples[sut.SAMPLE_IID_COL], samples[sut.SAMPLE_OMEGA_IDX_COL]))

        focal, R = sut.build_residualization_matrix(samples, graph, sut.GROUP_2ND)

        n_row = R.toarray()[list(focal).index(idx["n"])]
        assert np.isclose(n_row[idx["n"]], 3 / 4)
        assert all(np.isclose(n_row[idx[a]], -1 / 4) for a in ("a1", "a2", "a3"))


class TestCalculateGlsWeights:

    def test__a_pseudo_inverse__matching_one_taken_whole(self):
        _, _, _, omega, regressions, _ = relative_plan()
        for regression in regressions:
            A = regression.phenotype_map
            vcv = (A @ omega @ A.T).toarray()
            Q = regression.gls_weights.toarray()

            assert np.allclose(vcv @ Q @ vcv, vcv)
            assert np.allclose(Q, np.linalg.pinv(vcv, rcond=1e-10, hermitian=True))

    def test__empty__empty(self):
        assert sut.calculate_gls_weights(sp.csr_array((0, 0))).shape == (0, 0)

    def test__large_component__rounding_noise_not_inverted(self):
        """One component of 1,000 people: 250 sibships of 4, chained by weak cousin links.

        R removes each sibship's mean, so the covariance has 250 exactly-zero eigenvalues, which
        come out as rounding noise that grows with the block.  numpy's default cutoff is a fixed
        1e-15 of the largest singular value, and at this size enough of the noise exceeds it to
        give weights around 1e13 to 1e14.  1,000 rather than the few hundred where the failure
        begins: where it begins depends on the numerical library, and this is well past it.
        """

        rng = np.random.default_rng(seed=810)
        num_sibships, sibship_size = 250, 4
        N = num_sibships * sibship_size
        omega = np.identity(N)
        R = np.identity(N)
        for sibship in range(num_sibships):
            idx = np.arange(sibship * sibship_size, (sibship + 1) * sibship_size)
            omega[np.ix_(idx, idx)] = 0.4
            omega[idx, idx] = 1.0
            R[np.ix_(idx, idx)] -= 1.0 / sibship_size
            if sibship > 0:
                omega[idx[0], idx[0] - 1] = omega[idx[0] - 1, idx[0]] = 0.1
        omega *= rng.uniform(0.5, 3.0)
        vcv = R @ omega @ R.T

        actual = sut.calculate_gls_weights(sp.csr_array(vcv)).toarray()

        assert np.abs(actual).max() < 10.0
        assert np.allclose(vcv @ actual @ vcv, vcv, atol=1e-8)


class TestBuildRelativeRegressions:

    def test__each_group__its_regression_in_order(self):
        _, _, _, _, regressions, _ = relative_plan()

        assert [r.name for r in regressions] == [sut.GROUP_FS, sut.GROUP_2ND, sut.GROUP_3RD]
        assert [len(r.focal) for r in regressions] == [5, 3, 3]

    def test__phenotypes__the_residualized_ones(self):
        _, _, phenotypes, _, regressions, _ = relative_plan()

        for regression in regressions:
            assert np.allclose(regression.phenotypes, regression.phenotype_map @ phenotypes)

    def test__rel_degree__later_regressions_left_out_without_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _, _, _, _, regressions, _ = relative_plan(rel_degree=1)

        assert [r.name for r in regressions] == [sut.GROUP_FS]
        # Regressions the options exclude are not "empty": warning about them would mislead
        assert "No sample qualifies" not in caplog.text

    def test__empty_group__left_out_with_a_warning(self, caplog):
        rows = [r for r in RELATIVE_ROWS if r[2] != "2nd"]
        phenos = {k : v for k, v in RELATIVE_PHENOS.items() if k not in ("n", "m1", "m2")}

        with caplog.at_level(logging.WARNING):
            _, _, _, _, regressions, _ = relative_plan(phenos, rows)

        assert [r.name for r in regressions] == [sut.GROUP_FS, sut.GROUP_3RD]
        assert "No sample qualifies for the 2ND regression" in caplog.text


class TestReportedEstimates:

    @pytest.mark.parametrize("has_po, rel_degree, expected", [
        (True, 3, [("PO",), ("PO", "FS"), ("PO", "FS", "2ND"), ("PO", "FS", "2ND", "3RD")]),
        (True, 2, [("PO",), ("PO", "FS"), ("PO", "FS", "2ND")]),
        (True, 1, [("PO",), ("PO", "FS")]),
        (True, 0, [("PO",), ("PO", "FS")]),
        (False, 3, [("FS",), ("FS", "2ND"), ("FS", "2ND", "3RD")]),
        (False, 1, [("FS",)])])
    def test__cumulative__as_far_as_rel_degree_allows(self, has_po, rel_degree, expected):
        assert sut.reported_estimates(has_po, rel_degree) == expected


class TestRegressionEstimates:

    def test__estimates__match_a_dense_textbook_regression(self):
        samples, _, phenotypes, omega, regressions, _ = relative_plan()
        genotypes = genotypes_for(len(phenotypes))

        for regression in regressions:
            residuals = sut.relative_genotype_residuals(genotypes, regression)
            _, d, beta = sut.regression_statistics(residuals, regression)
            expected_beta, expected_d, _, _ = dense_regression(regression, omega, phenotypes,
                                                               genotypes)

            assert np.allclose(beta, expected_beta) and np.allclose(d, expected_d)

    def test__missing_genotype__zeroes_only_the_residuals_it_enters(self):
        samples, _, phenotypes, _, regressions, _ = relative_plan()
        idx = dict(zip(samples[sut.SAMPLE_IID_COL], samples[sut.SAMPLE_OMEGA_IDX_COL]))
        genotypes = genotypes_for(len(phenotypes))
        genotypes[0, idx["a2"]] = np.nan
        fs = regressions[0]

        residuals = sut.relative_genotype_residuals(genotypes, fs)

        # a2's own residual and its sibs' are lost at SNP 0.  b1's and b2's do not involve a2, so
        # they are exactly what R gives with a2's genotype left out; and every other SNP is intact.
        complete = genotypes_for(len(phenotypes))
        expected = complete @ fs.phenotype_map.T.toarray()
        assert np.all(residuals[0, :3] == 0.0)
        assert np.allclose(residuals[0, 3:], expected[0, 3:])
        assert np.allclose(residuals[1:], expected[1:])
        assert not np.isnan(residuals).any()

    def test__snp_with_no_variation__no_estimate(self):
        _, _, phenotypes, _, regressions, _ = relative_plan()
        genotypes = genotypes_for(len(phenotypes))
        genotypes[2] = 1.0

        for regression in regressions:
            residuals = sut.relative_genotype_residuals(genotypes, regression)
            _, d, beta = sut.regression_statistics(residuals, regression)

            assert d[2] == 0.0 and np.isnan(beta[2])
            assert np.all(residuals[2] == 0.0)


class TestCovarianceOfEstimates:

    def test__matches_w_omega_w_with_weights_over_all_of_omega(self):
        _, _, phenotypes, omega, regressions, cross = relative_plan()
        genotypes = genotypes_for(len(phenotypes))
        names = [r.name for r in regressions]
        stats = [sut.regression_statistics(sut.relative_genotype_residuals(genotypes, r), r)
                 for r in regressions]

        actual = sut.covariance_of_estimates([s[0] for s in stats], [s[1] for s in stats], names,
                                             cross)

        # The specification's formula: w = A'QZ / d over all of Omega, then C = W Omega W'
        W = []
        for regression in regressions:
            _, d, Z, Q = dense_regression(regression, omega, phenotypes, genotypes)
            W.append((regression.phenotype_map.toarray().T @ Q @ Z.T / d).T)   # (snps, omega)
        W = np.stack(W, axis=1)                                                  # (snps, k, omega)
        expected = np.einsum("sko,op,slp->skl", W, omega.toarray(), W)
        assert np.allclose(actual, expected)

    def test__diagonal__each_regressions_own_gls_variance(self):
        _, _, phenotypes, _, regressions, cross = relative_plan()
        genotypes = genotypes_for(len(phenotypes))
        stats = [sut.regression_statistics(sut.relative_genotype_residuals(genotypes, r), r)
                 for r in regressions]

        C = sut.covariance_of_estimates([s[0] for s in stats], [s[1] for s in stats],
                                        [r.name for r in regressions], cross)

        for k, (_, d, _) in enumerate(stats):
            assert np.allclose(C[:, k, k], 1.0 / d)

    def test__regression_without_an_estimate__its_row_and_column_nan(self):
        _, _, phenotypes, _, regressions, cross = relative_plan()
        genotypes = genotypes_for(len(phenotypes))
        stats = [sut.regression_statistics(sut.relative_genotype_residuals(genotypes, r), r)
                 for r in regressions]
        ds = [s[1].copy() for s in stats]
        ds[1][4] = 0.0

        C = sut.covariance_of_estimates([s[0] for s in stats], ds,
                                        [r.name for r in regressions], cross)

        assert np.all(np.isnan(C[4, 1, :])) and np.all(np.isnan(C[4, :, 1]))
        assert not np.isnan(C[4, 0, 2])


class TestMetaAnalyze:

    @staticmethod
    def by_hand(betas, C):
        """Each SNP separately, with an ordinary inverse over the estimates present"""
        estimates, ses = [], []
        for b, c in zip(betas, C):
            keep = ~np.isnan(b)
            inv = np.linalg.inv(c[np.ix_(keep, keep)])
            a = inv.sum(axis=1) / inv.sum()
            estimates.append(a @ b[keep])
            ses.append(np.sqrt(a @ c[np.ix_(keep, keep)] @ a))
        return np.array(estimates), np.array(ses)

    BETAS = np.array([[0.5, 0.3, -0.1], [1.0, 1.2, 0.8], [0.2, np.nan, 0.4]])
    C = np.array([[[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]] * 3)

    def test__optimal_weights__match_a_per_snp_calculation(self):
        C = self.C.copy()
        C[2, 1, :] = C[2, :, 1] = np.nan

        actual = sut.meta_analyze(self.BETAS, C)

        expected = self.by_hand(self.BETAS, C)
        assert np.allclose(actual[0], expected[0]) and np.allclose(actual[1], expected[1])

    def test__single_estimate__returned_as_it_is(self):
        estimate, se = sut.meta_analyze(self.BETAS[:, :1], self.C[:, :1, :1])

        assert np.allclose(estimate, self.BETAS[:, 0]) and np.allclose(se, 0.2)

    def test__every_estimate_missing__nan(self):
        estimate, se = sut.meta_analyze(np.full((1, 2), np.nan), np.full((1, 2, 2), np.nan))

        assert np.isnan(estimate[0]) and np.isnan(se[0])

    def test__singular_covariance__pseudo_inverse_still_gives_weights_summing_to_one(self):
        # Two identical estimates: C singular.  Any split of the weight is optimal; the result is
        # their common value, with their common variance.
        betas = np.array([[0.7, 0.7]])
        C = np.array([[[0.04, 0.04], [0.04, 0.04]]])

        estimate, se = sut.meta_analyze(betas, C)

        assert np.isclose(estimate[0], 0.7) and np.isclose(se[0], 0.2)


class TestEstimateBlock:

    def test__alleles_recoded__every_estimate_negated_ses_unchanged(self):
        _, _, phenotypes, _, regressions, cross = relative_plan()
        genotypes = genotypes_for(len(phenotypes))
        estimates = sut.reported_estimates(has_po=False)

        betas, ses = sut.estimate_block(genotypes, regressions, cross, estimates)
        recoded_betas, recoded_ses = sut.estimate_block(2.0 - genotypes, regressions, cross,
                                                        estimates)

        assert np.allclose(recoded_betas, -betas) and np.allclose(recoded_ses, ses)

    def test__phenotypes_shifted__estimates_unchanged(self):
        shifted = {k : v + 5.0 for k, v in RELATIVE_PHENOS.items()}
        plans = [relative_plan(), relative_plan(phenos=shifted)]
        genotypes = genotypes_for(len(plans[0][2]))
        estimates = sut.reported_estimates(has_po=False)

        results = [sut.estimate_block(genotypes, p[4], p[5], estimates) for p in plans]

        assert np.allclose(results[0][0], results[1][0]) and np.allclose(results[0][1],
                                                                          results[1][1])

    def test__first_estimate__the_first_regression_alone(self):
        _, _, phenotypes, _, regressions, cross = relative_plan()
        genotypes = genotypes_for(len(phenotypes))

        betas, ses = sut.estimate_block(genotypes, regressions, cross,
                                        sut.reported_estimates(has_po=False))

        _, d, beta = sut.regression_statistics(
            sut.relative_genotype_residuals(genotypes, regressions[0]), regressions[0])
        assert np.allclose(betas[:, 0], beta) and np.allclose(ses[:, 0], 1 / np.sqrt(d))

    def test__empty_regression__estimates_including_it_equal_those_without(self):
        rows = [r for r in RELATIVE_ROWS if r[2] != "2nd"]
        phenos = {k : v for k, v in RELATIVE_PHENOS.items() if k not in ("n", "m1", "m2")}
        _, _, phenotypes, _, regressions, cross = relative_plan(phenos, rows)

        betas, ses = sut.estimate_block(genotypes_for(len(phenotypes)), regressions, cross,
                                        sut.reported_estimates(has_po=False))

        # FS, FS+2ND, FS+2ND+3RD: with no 2ND regression, the first two coincide
        assert np.allclose(betas[:, 0], betas[:, 1]) and np.allclose(ses[:, 0], ses[:, 1])
        assert not np.allclose(betas[:, 1], betas[:, 2])

    def test__no_regressions__every_estimate_nan(self):
        betas, ses = sut.estimate_block(np.zeros((3, 2)), [], {},
                                        sut.reported_estimates(has_po=False, rel_degree=1))

        assert betas.shape == (3, 1) and np.all(np.isnan(betas)) and np.all(np.isnan(ses))


# --------------------------------------------------------------------------------------------------
# SNPs

def bim_of(rows):
    """A bim frame from (rsid, bp, a1, a2) rows, all on chromosome 1"""
    return grma2_io.read_bim(pd.DataFrame([("1", r, 0.0, bp, a1, a2) for r, bp, a1, a2 in rows],
                                          columns=list(bbf.BIM_COLS)))


def snipar_bim(rows):
    """snipar metadata holding only bim values, from (rsid, bp, allele1, allele2) rows"""
    return {snipar_hdf5.SNIPAR_BIM_VALUES_KEY :
                np.array([["1", r, "0", str(bp), a1, a2] for r, bp, a1, a2 in rows]),
            snipar_hdf5.SNIPAR_BIM_COLUMNS_KEY :
                np.array(["Chr", "id", "morgans", "coordinate", "allele1", "allele2"])}


class TestPlanSnps:

    BIM = [("rs1", 100, "A", "C"), ("rs2", 200, "G", "T"), ("rs3", 300, "A", "T"),
           ("rs4", 400, "C", "A")]

    def test__no_snipar_file__every_snp_no_alignment(self):
        actual = sut.plan_snps(bim_of(self.BIM))

        assert actual.bim_rows.tolist() == [0, 1, 2, 3]
        assert actual.snipar_positions is None and actual.flip is None

    def test__snp_list__only_listed_kept_unknown_ones_warned(self, caplog):
        snp_list = grma2_io.read_snp_list(pd.DataFrame({0 : ["rs3", "rs1", "rs99"]}))

        with caplog.at_level(logging.WARNING):
            actual = sut.plan_snps(bim_of(self.BIM), snp_list=snp_list)

        assert actual.bim_rows.tolist() == [0, 2]
        assert "1 SNP(s) in the SNP list are not in the bim file" in caplog.text

    def test__snp_list_naming_only_known_snps__no_warning(self, caplog):
        snp_list = grma2_io.read_snp_list(pd.DataFrame({0 : ["rs2", "rs4"]}))

        with caplog.at_level(logging.WARNING):
            actual = sut.plan_snps(bim_of(self.BIM), snp_list=snp_list)

        assert actual.bim_rows.tolist() == [1, 3]
        assert "not in the bim file" not in caplog.text

    def test__snp_list_naming_none__raises(self):
        snp_list = grma2_io.read_snp_list(pd.DataFrame({0 : ["rs99"]}))

        with pytest.raises(ValueError, match="No SNP is left to analyse"):
            sut.plan_snps(bim_of(self.BIM), snp_list=snp_list)

    def test__aligned__positions_and_flips(self, caplog):
        # rs1 as in the bim; rs2 swapped; rs3 (palindromic) usable, since no SNP shows opposite
        # strands; rs4 not in snipar's file
        snipar = snipar_bim([("rs1", 100, "A", "C"), ("rs2", 200, "T", "G"),
                             ("rs3", 300, "A", "T"), ("rs5", 500, "A", "G")])

        with caplog.at_level(logging.WARNING):
            actual = sut.plan_snps(bim_of(self.BIM), snipar_metadata=snipar)

        assert actual.bim_rows.tolist() == [0, 1, 2, 3]
        assert actual.snipar_positions.tolist() == [0, 1, 2, -1]
        assert actual.flip.tolist() == [False, True, False, False]
        assert "1 of 4 SNP(s) have no usable imputed parental data" in caplog.text

    def test__alleles_irreconcilable__no_usable_data(self):
        snipar = snipar_bim([("rs1", 100, "A", "G"), ("rs2", 200, "G", "T"),
                             ("rs3", 300, "A", "T"), ("rs4", 400, "C", "A")])

        actual = sut.plan_snps(bim_of(self.BIM), snipar_metadata=snipar)

        assert actual.snipar_positions.tolist() == [-1, 1, 2, 3]

    def test__opposite_strands__those_and_every_palindromic_snp_unusable(self, caplog):
        # rs2 is on the opposite strand (G/T against C/A); that makes rs3's strand unknowable
        snipar = snipar_bim([("rs1", 100, "A", "C"), ("rs2", 200, "C", "A"),
                             ("rs3", 300, "A", "T"), ("rs4", 400, "C", "A")])

        with caplog.at_level(logging.WARNING):
            actual = sut.plan_snps(bim_of(self.BIM), snipar_metadata=snipar)

        assert actual.snipar_positions.tolist() == [0, -1, -1, 3]
        assert "opposite strands" in caplog.text

    def test__rsid_at_a_different_position__raises(self):
        snipar = snipar_bim([("rs1", 101, "A", "C")])

        with pytest.raises(ValueError, match="different positions") as error:
            sut.plan_snps(bim_of(self.BIM[:1]), snipar_metadata=snipar)

        assert "rs1" in str(error.value)

    def test__rsid_repeated_in_snipar_and_named_by_the_bim__raises(self):
        snipar = snipar_bim([("rs1", 100, "A", "C"), ("rs1", 100, "A", "C")])

        with pytest.raises(ValueError, match="more than once") as error:
            sut.plan_snps(bim_of(self.BIM[:1]), snipar_metadata=snipar)

        assert "rs1" in str(error.value)

    def test__repeated_rsid_the_bim_does_not_name__no_error(self):
        snipar = snipar_bim([(".", 50, "A", "G"), (".", 60, "A", "G"), ("rs1", 100, "A", "C")])

        actual = sut.plan_snps(bim_of(self.BIM[:1]), snipar_metadata=snipar)

        assert actual.snipar_positions.tolist() == [2]

    def test__snipar_in_a_different_order__raises_naming_the_snp(self):
        snipar = snipar_bim([("rs2", 200, "G", "T"), ("rs1", 100, "A", "C")])

        with pytest.raises(ValueError, match="different order") as error:
            sut.plan_snps(bim_of(self.BIM[:2]), snipar_metadata=snipar)

        assert "rs2" in str(error.value)

    def test__bim_listing_a_snp_twice__repeated_position_accepted(self):
        bim = bim_of([("rs1", 100, "A", "C"), ("rs2", 200, "G", "T"), ("rs2", 200, "G", "T")])
        snipar = snipar_bim([("rs1", 100, "A", "C"), ("rs2", 200, "G", "T")])

        actual = sut.plan_snps(bim, snipar_metadata=snipar)

        assert actual.snipar_positions.tolist() == [0, 1, 1]


# --------------------------------------------------------------------------------------------------
# The PO regression
#
# Family 1: dad1 and mum1 both genotyped, children k1a and k1b.  mum1's sister aunt1 puts mum1 in
# the FS regression too, so that mum1 is in Omega however her motherhood is recorded.
# Family 2: dad2 genotyped but WITHOUT a phenotype -- read for his children's residuals, but not in
# Omega -- and children k2a, k2b; snipar family S2 holds the other parent's data.
# Family 3: dad3 genotyped, child k3; snipar family S3.
# s1 and s2: sibs without parents.  k3 and s1 are cousins, linking families.

PO_PHENOS = {"dad1" : 0.3, "mum1" : -0.8, "k1a" : 1.4, "k1b" : 0.2, "aunt1" : -1.1,
             "dad2" : -9.0, "k2a" : 0.9, "k2b" : -0.6, "dad3" : 1.7, "k3" : 2.1,
             "s1" : -1.9, "s2" : -0.4}
PO_KING = [("dad1", "k1a", "PO"), ("dad1", "k1b", "PO"), ("mum1", "k1a", "PO"),
           ("mum1", "k1b", "PO"), ("dad2", "k2a", "PO"), ("dad2", "k2b", "PO"),
           ("dad3", "k3", "PO"), ("k1a", "k1b", "FS"), ("k2a", "k2b", "FS"),
           ("mum1", "aunt1", "FS"), ("s1", "s2", "FS"), ("k3", "s1", "3rd")]
PO_RECORDS = [("S1", "k1a", "dad1", "mum1", True, True),
              ("S1", "k1b", "dad1", "mum1", True, True),
              ("S2", "k2a", "dad2", "mum2", True, False),
              ("S2", "k2b", "dad2", "mum2", True, False),
              ("S3", "k3", "dad3", "mum3", True, False)]


def po_plan(records=PO_RECORDS, families=("S2", "S3")):
    samples, graph, parents = planned(PO_PHENOS, PO_KING, records, list(families))
    phenotypes = sut.prepare_phenotypes(samples)
    omega = sut.build_omega(samples, graph, phenotypes)
    bed_cols, column_of_row = sut.plan_genotype_samples(samples, parents)
    po = sut.build_po_regression(samples, parents, phenotypes, omega, column_of_row)
    regressions = [po] + sut.build_relative_regressions(samples, graph, phenotypes, omega)
    cross = sut.build_cross_covariances(regressions, omega)
    return dict(samples=samples, graph=graph, parents=parents, phenotypes=phenotypes, omega=omega,
                bed_cols=bed_cols, column_of_row=column_of_row, regressions=regressions,
                cross=cross)


def po_data(num_loaded, families, num_snps=12, seed=5):
    """Genotypes for the loaded samples, and snipar data for the families: at each SNP, k is 3 or 4
    (with a few 2s and gaps), and the imputed value is what that k implies -- an allele plus f, or
    a genotype"""
    rng = np.random.default_rng(seed)
    genotypes = rng.integers(0, 3, size=(num_snps, num_loaded)).astype(float)
    k = rng.choice([3.0, 4.0], size=(num_snps, families))
    f = rng.uniform(0.1, 0.9, size=num_snps)
    imputed = np.where(k == 3, rng.integers(0, 2, size=k.shape) + f[:, None],
                       rng.integers(0, 3, size=k.shape).astype(float))
    k[0, 0] = 2.0
    k[1, 1] = np.nan
    imputed[1, 1] = np.nan
    return genotypes, (imputed, k, f)


class TestPlanGenotypeSamples:

    def test__omega_first_then_parents_outside_it(self):
        plan = po_plan()
        samples = plan["samples"]
        iid = samples[sut.SAMPLE_IID_COL].to_numpy()
        row = dict(zip(iid, range(len(iid))))

        bed_cols, column_of_row = plan["bed_cols"], plan["column_of_row"]

        num_omega = len(plan["phenotypes"])
        omega_rows = np.flatnonzero(samples[sut.SAMPLE_OMEGA_IDX_COL].to_numpy() != -1)
        assert bed_cols[:num_omega].tolist() == samples[sut.SAMPLE_BED_COL][omega_rows].tolist()
        # dad2 has no phenotype, so is not in Omega, but is read, after Omega's samples
        assert bed_cols.tolist()[num_omega:] == [samples[sut.SAMPLE_BED_COL][row["dad2"]]]
        assert column_of_row[row["dad2"]] == num_omega
        assert column_of_row[row["k1a"]] == samples[sut.SAMPLE_OMEGA_IDX_COL][row["k1a"]]


class TestBuildPoRegression:

    def test__children_their_phenotypes_and_their_block_of_omega(self):
        plan = po_plan()
        po, samples = plan["regressions"][0], plan["samples"]
        idx = dict(zip(samples[sut.SAMPLE_IID_COL], samples[sut.SAMPLE_OMEGA_IDX_COL]))
        children = [idx[c] for c in ("k1a", "k1b", "k2a", "k2b", "k3")]

        assert po.name == sut.GROUP_PO and po.focal.tolist() == children
        assert np.allclose(po.phenotypes, plan["phenotypes"][children])
        block = plan["omega"].toarray()[np.ix_(children, children)]
        assert np.allclose(po.gls_weights.toarray(), np.linalg.pinv(block, hermitian=True))

    def test__parents_located(self):
        plan = po_plan()
        po, col = plan["regressions"][0], plan["column_of_row"]
        row = dict(zip(plan["samples"][sut.SAMPLE_IID_COL], range(len(plan["samples"]))))

        assert po.parents.father.tolist() == [col[row["dad1"]]] * 2 + [col[row["dad2"]]] * 2 + [
            col[row["dad3"]]]
        assert po.parents.mother.tolist() == [col[row["mum1"]]] * 2 + [-1] * 3
        assert po.parents.family.tolist() == [-1, -1, 0, 0, 1]

    def test__no_parents__no_regression(self):
        samples, graph, parents = planned(RELATIVE_PHENOS, RELATIVE_ROWS)
        phenotypes = sut.prepare_phenotypes(samples)
        _, column_of_row = sut.plan_genotype_samples(samples, parents)

        assert sut.build_po_regression(samples, parents, phenotypes,
                                       sut.build_omega(samples, graph, phenotypes),
                                       column_of_row) is None


def po_residuals_by_hand(genotypes, snipar_block, po):
    """Each child at each SNP, case by case"""
    imputed, k, f = snipar_block
    out = np.zeros((genotypes.shape[0], len(po.focal)))
    for s in range(genotypes.shape[0]):
        for c, child in enumerate(po.focal):
            dad, mum, fam = po.parents.father[c], po.parents.mother[c], po.parents.family[c]
            g_c = genotypes[s, child]
            if dad != -1 and mum != -1:
                out[s, c] = g_c - (genotypes[s, dad] + genotypes[s, mum]) / 2
                continue
            g_obs = genotypes[s, dad if dad != -1 else mum]
            if fam == -1 or np.isnan(k[s, fam]):
                continue
            if k[s, fam] == 3:
                out[s, c] = g_c - g_obs / 2 - round(imputed[s, fam] - f[s])
            elif k[s, fam] == 4:
                out[s, c] = g_c - (g_obs + round(imputed[s, fam])) / 2
    return out


class TestPoGenotypeResiduals:

    def test__every_case__as_worked_case_by_case(self):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, block = po_data(len(plan["bed_cols"]), 2)

        actual = sut.po_genotype_residuals(genotypes, block, po)

        assert np.allclose(actual, po_residuals_by_hand(genotypes, block, po))
        # k = 2 (SNP 0, family S2) and a gap (SNP 1, family S3): missing, so zero
        assert np.all(actual[0, 2:4] == 0.0) and actual[1, 4] == 0.0

    def test__stored_as_float16__still_whole_numbers(self):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, (imputed, k, f) = po_data(len(plan["bed_cols"]), 2)
        stored = (imputed.astype(np.float16).astype(float), k, f.astype(np.float16).astype(float))

        actual = sut.po_genotype_residuals(genotypes, stored, po)

        assert np.allclose(actual, po_residuals_by_hand(genotypes, (imputed, k, f), po))

    def test__missing_genotype__that_residual_missing(self):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, block = po_data(len(plan["bed_cols"]), 2)
        genotypes[3, po.parents.mother[0]] = np.nan     # mum1, at SNP 3
        genotypes[4, po.focal[4]] = np.nan              # k3 itself, at SNP 4

        actual = sut.po_genotype_residuals(genotypes, block, po)

        assert np.all(actual[3, :2] == 0.0) and actual[4, 4] == 0.0
        assert not np.isnan(actual).any()

    def test__imputed_value_not_a_whole_number_of_alleles__missing_with_a_warning(self, caplog):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, (imputed, k, f) = po_data(len(plan["bed_cols"]), 2)
        imputed = imputed.copy()
        imputed[5, 1] += 0.4

        with caplog.at_level(logging.WARNING):
            actual = sut.po_genotype_residuals(genotypes, (imputed, k, f), po)

        assert actual[5, 4] == 0.0
        assert "not a whole number of alleles" in caplog.text

    def test__no_snipar_data__only_two_parent_children_have_residuals(self):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, _ = po_data(len(plan["bed_cols"]), 2)

        actual = sut.po_genotype_residuals(genotypes, None, po)

        assert np.all(actual[:, 2:] == 0.0) and np.any(actual[:, :2] != 0.0)


class TestPoGenotypeResidualsOneChild:
    """One child, built directly, for the cases the shared fixture has no example of"""

    @staticmethod
    def child_with(father, mother, family):
        return sut.Regression(name=sut.GROUP_PO, focal=np.array([0]),
                              phenotype_map=sp.csr_array(np.ones((1, 1))),
                              phenotypes=np.zeros(1), gls_weights=sp.csr_array(np.ones((1, 1))),
                              parents=sut.ParentColumns(father=np.array([father]),
                                                        mother=np.array([mother]),
                                                        family=np.array([family])))

    # Columns: the child, then the genotyped parent.  SNP 0 has k = 3 (transmitted allele 1, so
    # imputed = 1 + f); SNP 1 has k = 4 (the other parent's genotype is 2).
    GENOTYPES = np.array([[2.0, 1.0], [1.0, 0.0]])
    BLOCK = (np.array([[1.3], [2.0]]), np.array([[3.0], [4.0]]), np.array([0.3, 0.5]))

    def test__only_the_mother_genotyped__her_genotype_used(self):
        actual = sut.po_genotype_residuals(self.GENOTYPES, self.BLOCK,
                                           self.child_with(father=-1, mother=1, family=0))

        assert np.allclose(actual[:, 0], [2.0 - 1.0 / 2 - 1.0, 1.0 - (0.0 + 2.0) / 2])

    def test__one_parent_but_no_imputed_row__missing(self):
        actual = sut.po_genotype_residuals(self.GENOTYPES, self.BLOCK,
                                           self.child_with(father=1, mother=-1, family=-1))

        assert np.all(actual == 0.0)


class TestEstimateBlockWithPo:

    def test__po_estimates__match_a_dense_textbook_regression(self):
        plan = po_plan()
        po = plan["regressions"][0]
        genotypes, block = po_data(len(plan["bed_cols"]), 2)

        betas, ses = sut.estimate_block(genotypes, plan["regressions"], plan["cross"],
                                        sut.reported_estimates(has_po=True), snipar_block=block)

        Z = po_residuals_by_hand(genotypes, block, po)
        A = po.phenotype_map.toarray()
        Q = np.linalg.pinv(A @ plan["omega"].toarray() @ A.T, hermitian=True)
        d = np.einsum("sf,fg,sg->s", Z, Q, Z)
        assert np.allclose(betas[:, 0], (Z @ Q @ (A @ plan["phenotypes"])) / d)
        assert np.allclose(ses[:, 0], 1 / np.sqrt(d))

    def test__alleles_recoded_with_snipar__every_estimate_negated(self):
        # g -> 2 - g, imputed -> 2 - imputed, f -> 1 - f, as a flipped SNP is read: the PO
        # regression too must simply change sign.  The original implementation did not (notes
        # §13.7's second cause).
        plan = po_plan()
        genotypes, (imputed, k, f) = po_data(len(plan["bed_cols"]), 2)
        estimates = sut.reported_estimates(has_po=True)

        betas, ses = sut.estimate_block(genotypes, plan["regressions"], plan["cross"], estimates,
                                        snipar_block=(imputed, k, f))
        recoded_betas, recoded_ses = sut.estimate_block(
            2.0 - genotypes, plan["regressions"], plan["cross"], estimates,
            snipar_block=(2.0 - imputed, k, 1.0 - f))

        assert np.allclose(recoded_betas, -betas) and np.allclose(recoded_ses, ses)

    def test__mother_imputed_from_her_own_genotypes__same_as_genotyped(self):
        # Patrick's check: hide mum1 from the pedigree and give snipar her true genotypes as an
        # imputed parent with every allele observed (k = 4).  Family 1's children must get exactly
        # the results they get with mum1 genotyped.  mum1 stays in Omega either way, as aunt1's
        # sister, so nothing else changes.
        genotyped = po_plan()
        records = [(f, i, d, "hidden_mum1" if m == "mum1" else m, hd,
                    False if m == "mum1" else hm) for f, i, d, m, hd, hm in PO_RECORDS]
        imputed_mum = po_plan(records, families=("S1", "S2", "S3"))

        genotypes, (imputed, k, f) = po_data(len(genotyped["bed_cols"]), 2)
        mum1 = genotyped["regressions"][0].parents.mother[0]
        imputed3 = np.column_stack([genotypes[:, mum1], imputed])
        k3 = np.column_stack([np.full(len(k), 4.0), k])

        estimates = sut.reported_estimates(has_po=True)
        expected = sut.estimate_block(genotypes, genotyped["regressions"], genotyped["cross"],
                                      estimates, snipar_block=(imputed, k, f))
        actual = sut.estimate_block(genotypes, imputed_mum["regressions"], imputed_mum["cross"],
                                    estimates, snipar_block=(imputed3, k3, f))

        assert np.array_equal(imputed_mum["bed_cols"], genotyped["bed_cols"])
        assert np.allclose(actual[0], expected[0]) and np.allclose(actual[1], expected[1])


# --------------------------------------------------------------------------------------------------
# grma2(): planning and running, end to end
#
# The PO fixture's families, nine SNPs, and a snipar file on disk.  rs4 has G/T alleles; rs6's are
# swapped in snipar's file; rs7 is not in it at all.

NUM_SNPS = 9


def grma2_inputs(tmp_path, seed=8, chunk_snps=3):
    rng = np.random.default_rng(seed)
    iids = list(PO_PHENOS)
    fam_df = pd.DataFrame({bbf.FAM_FID_COL : "F", bbf.FAM_IID_COL : iids,
                           bbf.FAM_IIDF_COL : "0", bbf.FAM_IIDM_COL : "0", bbf.FAM_SEX_COL : "0",
                           bbf.FAM_PHENO_COL : list(PO_PHENOS.values())})
    king_df = pd.DataFrame([("F", a, "F", b, t) for a, b, t in PO_KING],
                           columns=list(grma2_io.KING_COLS))
    alleles = [("A", "C")] * NUM_SNPS
    alleles[4] = ("G", "T")
    bim_df = pd.DataFrame([("1", f"rs{s}", 0.0, 100 * (s + 1), *alleles[s])
                           for s in range(NUM_SNPS)], columns=list(bbf.BIM_COLS))
    genotypes = rng.integers(0, 3, size=(NUM_SNPS, len(iids))).astype(float)
    genotypes[2, 3] = np.nan

    # snipar: every SNP but rs7, with rs6's alleles swapped.  Values exactly representable in
    # float16, so that what is written is what is read.
    snipar_snps = [s for s in range(NUM_SNPS) if s != 7]
    f = np.round(rng.uniform(0.1, 0.9, len(snipar_snps)) * 64) / 64
    k = rng.choice([3.0, 4.0], size=(2, len(snipar_snps)))
    imputed = np.where(k == 3, rng.integers(0, 2, size=k.shape) + f,
                       rng.integers(0, 3, size=k.shape).astype(float))
    snipar_alleles = [alleles[s][::-1] if s == 6 else alleles[s] for s in snipar_snps]
    path = os.path.join(tmp_path, "imp" + snipar_hdf5.SNIPAR_FILE_SUFFIX)
    snipar_hdf5.write_snipar_file(
        path, imputed_gts=imputed, num_obs_par_alleles=k, standard_f=f,
        parental_status=np.zeros((2, 3), dtype=bool),
        pedigree=np.array([[f_, i, d, m, str(hd), str(hm)] for f_, i, d, m, hd, hm in PO_RECORDS]),
        families=np.array(["S2", "S3"]),
        bim_values=np.array([["1", f"rs{s}", "0", str(100 * (s + 1)), *a]
                             for s, a in zip(snipar_snps, snipar_alleles)]),
        chunk_snps=chunk_snps)
    return dict(fam_file=fam_df, rel_file=king_df, bim_file=bim_df, bed_file=genotypes,
                imputed_parental_file=path, rel_degree=3)


def without(inputs, *names):
    return {k : v for k, v in inputs.items() if k not in names}


def assert_same_results(actual, expected):
    """The same results, up to floating-point rounding

    Different block sizes give numpy's batched linear algebra different shapes, which can change
    the last bit of a result; p-values, written as text, would show it.  So the bim columns are
    compared exactly, and every estimate, SE and p-value numerically.
    """
    assert actual.columns.tolist() == expected.columns.tolist()
    bim_cols = list(bbf.BIM_COLS)
    pd.testing.assert_frame_equal(actual[bim_cols], expected[bim_cols])
    for column in actual.columns[len(bim_cols):]:
        a = pd.to_numeric(actual[column].astype(object).where(actual[column].notna(), np.nan))
        e = pd.to_numeric(expected[column].astype(object).where(expected[column].notna(), np.nan))
        assert np.allclose(a, e, rtol=1e-12, atol=0, equal_nan=True), column


class TestGrma2:

    def test__plan__the_steps_done_one_by_one(self, tmp_path):
        inputs = grma2_inputs(tmp_path)

        plan = sut.plan_analysis(**without(inputs, "bed_file"))

        expected = po_plan()
        assert [r.name for r in plan.regressions] == [r.name for r in expected["regressions"]]
        for actual_r, expected_r in zip(plan.regressions, expected["regressions"]):
            assert np.array_equal(actual_r.focal, expected_r.focal)
            assert np.allclose(actual_r.phenotypes, expected_r.phenotypes)
            assert np.allclose(actual_r.gls_weights.toarray(), expected_r.gls_weights.toarray())
        assert np.array_equal(plan.bed_cols, expected["bed_cols"])
        assert plan.estimates == sut.reported_estimates(has_po=True)
        assert plan.snps.snipar_positions.tolist() == [0, 1, 2, 3, 4, 5, 6, -1, 7]
        assert plan.snps.flip.tolist() == [False] * 6 + [True, False, False]

    def test__results__the_same_as_every_snp_estimated_at_once(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        plan = sut.plan_analysis(**without(inputs, "bed_file"))

        actual = sut.grma2(**inputs, snps_per_block=4)

        with grma2_io.SniparReader(inputs["imputed_parental_file"], plan.standard_f) as reader:
            block = reader.read(plan.snps.snipar_positions, plan.snps.flip)
        betas, ses = sut.estimate_block(inputs["bed_file"][:, plan.bed_cols], plan.regressions,
                                        plan.cross, plan.estimates, snipar_block=block)
        for e, combined in enumerate(plan.estimates):
            label = grma2_io.estimate_label(combined)
            assert np.allclose(actual["BETA_" + label], betas[:, e], equal_nan=True)
            assert np.allclose(actual["SE_" + label], ses[:, e], equal_nan=True)

    @pytest.mark.parametrize("snps_per_block", [1, 2, 5, 100])
    def test__block_size__makes_no_difference(self, tmp_path, snps_per_block):
        inputs = grma2_inputs(tmp_path)

        expected = sut.grma2(**inputs, snps_per_block=3)
        actual = sut.grma2(**inputs, snps_per_block=snps_per_block)

        assert_same_results(actual, expected)

    def test__from_files__the_same_as_from_frames(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        prefix = os.path.join(tmp_path, "data")
        fam, bim = inputs["fam_file"], inputs["bim_file"]
        bbf.write_fam_file(prefix + ".fam", fam[bbf.FAM_FID_COL].to_numpy(),
                           fam[bbf.FAM_IID_COL].to_numpy(), pheno=fam[bbf.FAM_PHENO_COL].to_numpy())
        bbf.write_bim_file(prefix + ".bim", bim[bbf.BIM_CHR_COL].to_numpy(),
                           bim[bbf.BIM_RSID_COL].to_numpy(), bim[bbf.BIM_BP_COL].to_numpy(),
                           bim[bbf.BIM_A1_COL].to_numpy(), bim[bbf.BIM_A2_COL].to_numpy())
        bbf.write_bed_file(prefix + ".bed", inputs["bed_file"])
        inputs["rel_file"].to_csv(prefix + ".kin0", sep="\t", index=False)

        from_files = sut.grma2(rel_file=prefix + ".kin0", bed_file=prefix + ".bed",
                               bim_file=prefix + ".bim", fam_file=prefix + ".fam", rel_degree=3,
                               imputed_parental_file=inputs["imputed_parental_file"])
        from_frames = sut.grma2(**inputs)

        assert_same_results(from_files, from_frames)

    def test__phenotype_file__the_same_as_those_phenotypes_in_the_fam_file(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        fam = inputs["fam_file"]
        new_phenos = fam[bbf.FAM_PHENO_COL].to_numpy() * 2.0 + 1.0
        new_phenos[fam[bbf.FAM_PHENO_COL].to_numpy() == -9.0] = -9.0
        pheno = pd.DataFrame({0 : fam[bbf.FAM_FID_COL], 1 : fam[bbf.FAM_IID_COL], 2 : new_phenos})

        actual = sut.grma2(**inputs, pheno_file=pheno)
        expected = sut.grma2(**{**inputs, "fam_file" : fam.assign(**{bbf.FAM_PHENO_COL :
                                                                       new_phenos})})

        assert_same_results(actual, expected)
        assert not np.allclose(actual["BETA_PO_FS"], sut.grma2(**inputs)["BETA_PO_FS"])

    def test__id_list__the_same_as_the_samples_left_out_lacking_phenotypes(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        fam = inputs["fam_file"]
        leave_out = fam[bbf.FAM_IID_COL].isin(["s2", "aunt1"]).to_numpy()
        ids = fam.loc[~leave_out, [bbf.FAM_FID_COL, bbf.FAM_IID_COL]]

        actual = sut.grma2(**inputs, id_list=ids)
        expected = sut.grma2(**{**inputs, "fam_file" : fam.assign(**{bbf.FAM_PHENO_COL : np.where(
            leave_out, -9.0, fam[bbf.FAM_PHENO_COL])})})

        assert_same_results(actual, expected)

    def test__covariates__the_same_as_the_phenotypes_residualized_on_them_first(self, tmp_path):
        # Residualizing on an intercept and covariates within Omega is a projection, so applying
        # it to phenotypes already residualized on the same covariates changes nothing: the run
        # with covariates must equal the run without them, given the pre-residualized phenotypes
        inputs = grma2_inputs(tmp_path)
        fam = inputs["fam_file"]
        age = np.linspace(20, 75, len(fam))
        covar = pd.DataFrame({0 : fam[bbf.FAM_FID_COL], 1 : fam[bbf.FAM_IID_COL], 2 : age})
        samples, _, _ = planned(PO_PHENOS, PO_KING, PO_RECORDS, ["S2", "S3"])
        in_omega = samples[sut.SAMPLE_OMEGA_IDX_COL].to_numpy() != -1
        y = fam[bbf.FAM_PHENO_COL].to_numpy()[in_omega]
        X = np.column_stack([age[in_omega], np.ones(in_omega.sum())])
        pre = fam[bbf.FAM_PHENO_COL].to_numpy().copy()
        pre[in_omega] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        actual = sut.grma2(**inputs, covar_file=covar)
        expected = sut.grma2(**{**inputs, "fam_file" : fam.assign(**{bbf.FAM_PHENO_COL : pre})})

        assert_same_results(actual, expected)
        assert not np.allclose(actual["BETA_PO_FS"], sut.grma2(**inputs)["BETA_PO_FS"])

    def test__snp_list__only_those_snps_with_the_same_results(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        snp_list = pd.DataFrame({0 : ["rs1", "rs6", "rs7", "rs8"]})

        actual = sut.grma2(**inputs, snp_list=snp_list)
        everything = sut.grma2(**inputs)

        assert actual["RSID"].tolist() == ["rs1", "rs6", "rs7", "rs8"]
        assert_same_results(actual, everything.iloc[[1, 6, 7, 8]].reset_index(drop=True))

    def test__snipar_reader__closed_after_the_run(self, tmp_path, monkeypatch):
        inputs = grma2_inputs(tmp_path)
        closed = []
        real_close = grma2_io.SniparReader.close
        monkeypatch.setattr(grma2_io.SniparReader, "close",
                            lambda self: closed.append(True) or real_close(self))

        sut.grma2(**inputs)

        assert closed == [True]

    @pytest.mark.parametrize("snipar, rel_degree, labels", [
        (True, 3, ["PO", "PO_FS", "PO_FS_2ND", "PO_FS_2ND_3RD"]),
        (True, 1, ["PO", "PO_FS"]),
        (False, 2, ["FS", "FS_2ND"])])
    def test__output_columns__from_the_options(self, tmp_path, caplog, snipar, rel_degree,
                                               labels):
        inputs = {**grma2_inputs(tmp_path), "rel_degree" : rel_degree}
        if not snipar:
            inputs = without(inputs, "imputed_parental_file")

        with caplog.at_level(logging.WARNING):
            actual = sut.grma2(**inputs)

        # This fixture's 2ND and 3RD groups are genuinely empty, which is warned about -- but only
        # where the options include them: a regression the options exclude is not an empty one
        for group, degree in ((sut.GROUP_2ND, 2), (sut.GROUP_3RD, 3)):
            warned = f"No sample qualifies for the {group} regression" in caplog.text
            assert warned == (rel_degree >= degree), group

        expected = list(bbf.BIM_COLS) + [f"{c}_{label}" for label in labels
                                         for c in ("BETA", "SE", "P")]
        assert actual.columns.tolist() == expected
        assert len(actual) == NUM_SNPS

    def test__snipar_file_without_po_children__warned_and_po_estimates_missing(self, tmp_path,
                                                                               caplog):
        inputs = grma2_inputs(tmp_path)
        # snipar's pedigree names no parent the fam file has, and records none as genotyped, so
        # no child has a parent and the PO regression has nobody
        path = inputs["imputed_parental_file"]
        metadata = snipar_hdf5.read_snipar_metadata(path)
        pedigree = metadata[snipar_hdf5.SNIPAR_PEDIGREE_KEY].copy()
        for column in (snipar_hdf5.SNIPAR_PED_FATHER_COL, snipar_hdf5.SNIPAR_PED_MOTHER_COL):
            pedigree[:, column] = "nobody"
        for column in (snipar_hdf5.SNIPAR_PED_HAS_FATHER_COL,
                       snipar_hdf5.SNIPAR_PED_HAS_MOTHER_COL):
            pedigree[:, column] = "False"
        os.remove(path)
        snipar_hdf5.write_snipar_file(
            path, imputed_gts=np.zeros((2, NUM_SNPS - 1)),
            num_obs_par_alleles=np.full((2, NUM_SNPS - 1), 3.0),
            standard_f=metadata[snipar_hdf5.SNIPAR_STANDARD_F_KEY],
            parental_status=np.zeros((2, 3), dtype=bool), pedigree=pedigree,
            families=metadata[snipar_hdf5.SNIPAR_FAMILIES_KEY],
            bim_values=metadata[snipar_hdf5.SNIPAR_BIM_VALUES_KEY])

        with caplog.at_level(logging.WARNING):
            actual = sut.grma2(**inputs)

        assert "No sample qualifies for the PO regression" in caplog.text
        assert actual["BETA_PO"].isna().all()
        # With no PO regression, PO+FS is the FS regression alone, which has estimates
        assert actual["BETA_PO_FS"].notna().all()

    def test__bad_input__raised_before_any_genotype_is_read(self, tmp_path):
        inputs = grma2_inputs(tmp_path)
        inputs["fam_file"] = pd.concat([inputs["fam_file"], inputs["fam_file"].iloc[[0]]],
                                       ignore_index=True)
        inputs["bed_file"] = os.path.join(tmp_path, "does_not_exist.bed")

        # A missing bed file would raise FileNotFoundError; the fam file's repeated sample is
        # reported first, because every check runs before the genotypes are read
        with pytest.raises(ValueError, match="more than once"):
            sut.grma2(**inputs)

    def test__rel_degree_given_as_text__accepted(self, tmp_path):
        inputs = grma2_inputs(tmp_path)

        expected = sut.grma2(**inputs)
        actual = sut.grma2(**{**inputs, "rel_degree" : "3"})

        assert_same_results(actual, expected)

