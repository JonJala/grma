"""
Tests of grma2_util: helpers shared by GRMA's modules.

TestDescribeOffenders is copied unchanged from the original implementation's tests, along with the
function itself.
"""

import os
import sys

import numpy as np

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)

import grma2_util as sut


class TestDescribeOffenders:

    # A message formatter, so the tests pin what the caller relies on -- the count being right and
    # the list being capped -- rather than the punctuation, which is free to change.
    #
    # The cases below pass an explicit limit so that they describe the BEHAVIOUR rather than
    # whatever MAX_REPORTED_OFFENDERS currently happens to be; two cases at the end cover the
    # default separately.
    LIMIT = 5

    def test__fewer_than_the_limit__all_of_them_named(self):
        ids = [f"IND{index}" for index in range(self.LIMIT - 2)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith(f"{self.LIMIT - 2} individual(s):")
        assert all(identifier in described for identifier in ids)
        assert "..." not in described


    def test__exactly_at_the_limit__still_all_of_them(self):
        # The boundary, since an off-by-one here would truncate a list that fits
        ids = [f"IND{index}" for index in range(self.LIMIT)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith(f"{self.LIMIT} individual(s):")
        assert all(identifier in described for identifier in ids)
        assert "..." not in described


    def test__over_the_limit__true_count_with_a_capped_sample(self):
        # The case this function exists for.  A fam file using something other than "0" for an
        # absent parent makes every individual an offender, and naming all of them turns one
        # error message into hundreds of thousands of lines.
        #
        # WHICH ones are named is deliberately not asserted: the docstring says that is arbitrary,
        # so pinning it here would contradict the contract.
        ids = [f"IND{index:06d}" for index in range(5000)]

        described = sut.describe_offenders(ids, limit=self.LIMIT)

        assert described.startswith("5000 individual(s), including:")
        assert described.count("IND") == self.LIMIT
        assert described.endswith(", ...]")


    def test__noun__used_in_the_phrase(self):
        # Callers name what they are reporting, so that the result reads as part of a sentence
        described = sut.describe_offenders(["rs1", "rs2"], noun="SNP(s)", limit=self.LIMIT)

        assert described.startswith("2 SNP(s):")


    def test__numpy_array__reads_the_same_as_a_list(self):
        # Several callers hold their IDs as numpy arrays, whose elements print their type
        # alongside their value unless they are converted first
        ids = [f"IND{index}" for index in range(self.LIMIT - 2)]

        assert (sut.describe_offenders(np.array(ids), limit=self.LIMIT) ==
                sut.describe_offenders(ids, limit=self.LIMIT))


    def test__nothing_to_report__says_so_without_failing(self):
        # Not reachable through the validators, which only call this once they have offenders,
        # but a formatter that falls over on an empty list is a poor one
        assert sut.describe_offenders([], limit=self.LIMIT).startswith("0 individual(s):")


    def test__no_limit_given__capped_at_the_module_default(self):
        # The callers all rely on the default, so it is worth knowing it is the constant rather
        # than something the function decided for itself
        ids = [f"IND{index:06d}" for index in range(sut.MAX_REPORTED_OFFENDERS + 1)]

        described = sut.describe_offenders(ids)

        assert described.count("IND") == sut.MAX_REPORTED_OFFENDERS
        assert described.endswith(", ...]")


    def test__no_limit_given__a_list_that_fits_is_not_capped(self):
        # The other side of the same boundary, at the default
        ids = [f"IND{index:06d}" for index in range(sut.MAX_REPORTED_OFFENDERS)]

        described = sut.describe_offenders(ids)

        assert described.count("IND") == sut.MAX_REPORTED_OFFENDERS
        assert "..." not in described
