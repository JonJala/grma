"""
PARENTAL_SNIPAR: families with every parental situation GRMA distinguishes, and a snipar file whose
values follow from the genotypes, as real snipar output would.

Genotypes come from simulated haplotypes: each founder has two, each child inherits one from each
parent, chosen at random.  So every child's genotype is a real Mendelian outcome, and snipar's value
for a missing parent can be written down exactly:

    - one of the missing parent's alleles known (k = 3): that allele plus the allele frequency f
    - both known (k = 4), because two children inherited different haplotypes from them: the
      parent's genotype
    - none known (k = 2): twice f

Families:
    A: dad_a and mum_a genotyped; children a1, a2, a3.  mum_a's sister aunt_a puts mum_a in the FS
       regression too, so that she is in Omega however her motherhood is recorded.  aunt_a is a
       2nd-degree relative of the children.
    B: dad_b genotyped but with no phenotype -- read for his children's residuals, not in Omega --
       and mum_b absent; children b1, b2.  Where they inherited different haplotypes from mum_b,
       snipar knows her genotype (k = 4); elsewhere one allele (k = 3).
    C: only mum_c genotyped; child c1.  Her missing partner's allele is known (k = 3) except at
       rs6 (k = 2).
    D: neither parent genotyped; children d1, d2, so no PO regression for them.  Their uncle u_d is
       2nd-degree only.
    z: a cousin of c1, 3rd-degree only.  solo: related to nobody.

SNPs: rs3's alleles are swapped in snipar's file; rs11 is not in it at all.  rs11 is one where
family A's children have nonzero residuals, so that the case tests that children with both parents
genotyped still contribute where snipar has nothing.  Values are exact in float16, as snipar stores
them: f is a multiple of 1/64.
"""

import numpy as np
import pandas as pd

import bedbimfam as bbf
import grma2_io


_rng = np.random.default_rng(seed=20261114)

NUM_SNPS = 12
_FREQ = np.round(_rng.uniform(0.15, 0.85, NUM_SNPS) * 64) / 64
_SNPS = np.arange(NUM_SNPS)

_haplotypes = {}            # IID -> (2, snps): row 0 inherited from the father, row 1 the mother
_from_mother = {}           # Child IID -> which of the mother's haplotypes was passed on, (snps,)


def _founder(iid):
    _haplotypes[iid] = (_rng.random((2, NUM_SNPS)) < _FREQ).astype(int)


def _child(iid, father, mother):
    from_father = _rng.integers(0, 2, NUM_SNPS)
    from_mother = _rng.integers(0, 2, NUM_SNPS)
    _haplotypes[iid] = np.vstack([_haplotypes[father][from_father, _SNPS],
                                  _haplotypes[mother][from_mother, _SNPS]])
    _from_mother[iid] = from_mother


# Parents before children, so that every child inherits from the haplotypes its parents really have
for founder in ("grandpa_a", "grandma_a", "dad_a", "dad_b", "mum_b", "dad_c", "mum_c", "dad_d",
                "mum_d", "u_d", "z", "solo"):
    _founder(founder)
_child("mum_a", "grandpa_a", "grandma_a")
_child("aunt_a", "grandpa_a", "grandma_a")
for kid, dad, mum in (("a1", "dad_a", "mum_a"), ("a2", "dad_a", "mum_a"), ("a3", "dad_a", "mum_a"),
                      ("b1", "dad_b", "mum_b"), ("b2", "dad_b", "mum_b"),
                      ("c1", "dad_c", "mum_c"), ("d1", "dad_d", "mum_d"), ("d2", "dad_d", "mum_d")):
    _child(kid, dad, mum)

# Who is in the fam file, in this order.  The grandparents, mum_b, dad_c, dad_d and mum_d are not.
IIDS = ["dad_a", "mum_a", "a1", "a2", "a3", "aunt_a", "dad_b", "b1", "b2", "mum_c", "c1", "d1",
        "d2", "u_d", "z", "solo"]
G = np.column_stack([_haplotypes[iid].sum(axis=0) for iid in IIDS]).astype(float)
G[4, IIDS.index("a2")] = np.nan                     # A missing genotype

# Phenotypes: an effect of rs0, a shared family effect, and noise.  dad_b has none.
_FAMILY_EFFECT = {"dad_a" : 0.6, "mum_a" : 0.6, "a1" : 0.6, "a2" : 0.6, "a3" : 0.6, "aunt_a" : 0.6,
                  "b1" : -0.4, "b2" : -0.4, "mum_c" : 0.2, "c1" : 0.2, "d1" : -0.8, "d2" : -0.8,
                  "u_d" : -0.8}
_phenos = np.round([0.5 * G[0, n] + _FAMILY_EFFECT.get(iid, 0.0) + _rng.normal(0.0, 1.0)
                    for n, iid in enumerate(IIDS)], 3)
_phenos[IIDS.index("dad_b")] = -9.0

FAM_DF = pd.DataFrame({bbf.FAM_FID_COL : "F", bbf.FAM_IID_COL : IIDS, bbf.FAM_IIDF_COL : "0",
                       bbf.FAM_IIDM_COL : "0", bbf.FAM_SEX_COL : "0", bbf.FAM_PHENO_COL : _phenos})

_PAIRS = ([(p, k, "PO") for p in ("dad_a", "mum_a") for k in ("a1", "a2", "a3")] +
          [("a1", "a2", "FS"), ("a1", "a3", "FS"), ("a2", "a3", "FS"), ("mum_a", "aunt_a", "FS")] +
          [("aunt_a", k, "2nd") for k in ("a1", "a2", "a3")] +
          [("dad_b", "b1", "PO"), ("dad_b", "b2", "PO"), ("b1", "b2", "FS")] +
          [("mum_c", "c1", "PO"), ("d1", "d2", "FS"), ("u_d", "d1", "2nd"), ("u_d", "d2", "2nd"),
           ("z", "c1", "3rd")])
REL_DF = pd.DataFrame([("F", a, "F", b, t) for a, b, t in _PAIRS],
                      columns=list(grma2_io.KING_COLS))

_ALLELES = [("A", "C"), ("G", "T"), ("C", "A"), ("A", "G"), ("T", "C"), ("G", "A"),
            ("C", "T"), ("A", "C"), ("T", "G"), ("G", "C"), ("A", "G"), ("C", "A")]
BIM_DF = pd.DataFrame([("1", f"rs{s}", 0.0, 1000 * (s + 1), *_ALLELES[s]) for s in _SNPS],
                      columns=list(bbf.BIM_COLS))


# snipar's values for each family with an imputed row ---------------------------------------------

def _missing_mother(kids, mother):
    """Values for a family whose mother is missing, from what her children show of her"""
    both_seen = np.array([len({_from_mother[k][s] for k in kids}) == 2 for s in _SNPS])
    transmitted = _haplotypes[mother][_from_mother[kids[0]], _SNPS]
    return (np.where(both_seen, _haplotypes[mother].sum(axis=0), transmitted + _FREQ),
            np.where(both_seen, 4.0, 3.0))


_b_imputed, _b_k = _missing_mother(["b1", "b2"], "mum_b")
# c1's missing parent is the father: the allele he passed on is c1's paternal haplotype
_c_imputed, _c_k = _haplotypes["c1"][0] + _FREQ, np.full(NUM_SNPS, 3.0)
_c_imputed[6], _c_k[6] = 2.0 * _FREQ[6], 2.0
# Both of family D's parents are missing; GRMA does not use such a family, so any value will do
_d_imputed, _d_k = np.full(NUM_SNPS, 1.0), np.zeros(NUM_SNPS)

# snipar's file: every SNP but MISSING_FROM_SNIPAR; SWAPPED_IN_SNIPAR's alleles swapped, so its
# values are counted in the other allele (2 - value, 1 - f)
MISSING_FROM_SNIPAR = 11
SWAPPED_IN_SNIPAR = 3
_SNIPAR_SNPS = [s for s in _SNPS if s != MISSING_FROM_SNIPAR]
_SWAPPED = SWAPPED_IN_SNIPAR


def _snipar_coding(values, flipped_as):
    out = values[_SNIPAR_SNPS].copy()
    out[_SNIPAR_SNPS.index(_SWAPPED)] = flipped_as - out[_SNIPAR_SNPS.index(_SWAPPED)]
    return out


SNIPAR = dict(
    imputed_gts=np.vstack([_snipar_coding(v, 2.0) for v in (_b_imputed, _c_imputed, _d_imputed)]),
    num_obs_par_alleles=np.vstack([k[_SNIPAR_SNPS] for k in (_b_k, _c_k, _d_k)]),
    standard_f=_snipar_coding(_FREQ, 1.0),
    parental_status=np.zeros((3, 3), dtype=bool),
    pedigree=np.array([["A", k, "dad_a", "mum_a", "True", "True"] for k in ("a1", "a2", "a3")] +
                      [["B", k, "dad_b", "mum_b", "True", "False"] for k in ("b1", "b2")] +
                      [["C", "c1", "dad_c", "mum_c", "False", "True"]] +
                      [["D", k, "dad_d", "mum_d", "False", "False"] for k in ("d1", "d2")]),
    families=np.array(["B", "C", "D"]),
    bim_values=np.array([["1", f"rs{s}", "0", str(1000 * (s + 1)),
                          *(_ALLELES[s][::-1] if s == _SWAPPED else _ALLELES[s])]
                         for s in _SNIPAR_SNPS]))

# Every child's genotype is one allele from each parent's haplotypes -- the case's central property
for _kid, _dad, _mum in (("a1", "dad_a", "mum_a"), ("b1", "dad_b", "mum_b"),
                         ("c1", "dad_c", "mum_c"), ("mum_a", "grandpa_a", "grandma_a")):
    assert all(_haplotypes[_kid][0, s] in _haplotypes[_dad][:, s] and
               _haplotypes[_kid][1, s] in _haplotypes[_mum][:, s] for s in _SNPS)
