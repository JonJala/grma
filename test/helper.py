import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import scipy.sparse.linalg as spla

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
import grma
import grma_lib
import bedbimfam as bbf

'''
This contains a straightforward, unscalable version of GRMA to use to test against in small enough
examples.

'''

# Assumes IDs are integers in the range of 0 to N (and that FID is unused)
def mock_create_rel_info(king_df: pd.DataFrame, fam_df: pd.DataFrame, rel_deg: int, N: int):
    ids = sorted(set(king_df[grma_lib.KING_IID1_COL].tolist()).union(
                 set(king_df[grma_lib.KING_IID2_COL].tolist())))

    rel_thresh_val = rel_deg

    fam_df_copy = fam_df.copy()
    fam_df_copy["INDEX"] = list(range(N))

    rel_df = king_df.copy()
    rel_df = pd.merge(left=rel_df, right=fam_df_copy, left_on=[grma_lib.KING_FID1_COL, grma_lib.KING_IID1_COL],
                      right_on=[bbf.FAM_FID_COL, bbf.FAM_IID_COL])
    rel_df.rename(columns={"INDEX": "INDEX1"}, inplace=True)
    rel_df = pd.merge(left=rel_df, right=fam_df_copy, left_on=[grma_lib.KING_FID2_COL, grma_lib.KING_IID2_COL],
                      right_on=[bbf.FAM_FID_COL, bbf.FAM_IID_COL])
    rel_df.rename(columns={"INDEX": "INDEX2"}, inplace=True)

    # Find the closest relationship levels for everyone
    lowest_rel = [5] * N

    for index, row in rel_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]

        if cur_rel < lowest_rel[row["INDEX1"]]:
            lowest_rel[row["INDEX1"]] = cur_rel
        if cur_rel < lowest_rel[row["INDEX2"]]:
            lowest_rel[row["INDEX2"]] = cur_rel


    # Construct the rel_info object and se_info objects
    rel_info = [[i] for i in range(N)]
    for index, row in rel_df.iterrows():
        cur_rel = grma_lib.INFTYPE_TO_DEG_MAP[row[grma_lib.KING_REL_COL]]

        if cur_rel == lowest_rel[row["INDEX1"]] and cur_rel <= rel_thresh_val:
            rel_info[row["INDEX1"]].append(row["INDEX2"])

        if cur_rel == lowest_rel[row["INDEX2"]] and cur_rel <= rel_thresh_val:
            rel_info[row["INDEX2"]].append(row["INDEX1"])

    # Sort the rel_info entries
    for i, rel in enumerate(rel_info):
        rel_info[i] = sorted(rel)

    return rel_info


def mock_shrink_offdiag(matrix: sp.csr_array, weight: float = 0.75):
    """Independent per-block reimplementation of the shrinkage applied in calculate_omega().

    Deliberately does NOT use the closed form (epsilon - variance) / lambda_min that production
    uses.  Instead it binary-searches, within each connected component, for the largest alpha
    that keeps that component positive definite.  Agreement between the two is therefore a real
    cross-check rather than a restatement of the same formula.

    Note that this makes no attempt to replicate the Gershgorin screen in get_block_alphas().
    That screen only decides which blocks can skip an eigendecomposition -- it never changes an
    alpha -- so examining every block here is equivalent by construction, and means a faulty
    screen shows up as a disagreement rather than being replicated into the expected value.
    The same reasoning applies to the dense/ARPACK routing in get_lambda_min(): this uses a dense
    eigendecomposition throughout, so a mistake in that routing is also visible here.
    """

    diag = matrix.diagonal()
    off_diag = matrix.copy()
    off_diag.setdiag(0)
    off_diag.eliminate_zeros()

    n_comp, labels = connected_components(off_diag, directed=False)
    dense_off = off_diag.toarray()
    alphas = np.ones(n_comp)

    for c in range(n_comp):
        idx = np.nonzero(labels == c)[0]
        if len(idx) < 2:
            continue

        block_off = dense_off[np.ix_(idx, idx)]
        block_diag = np.diag(diag[idx])

        if np.linalg.eigvalsh(block_diag + block_off).min() > 0.0:
            continue

        alpha_min = 0.0
        alpha_max = 1.0
        for _ in range(35):
            alpha = weight * alpha_max + (1.0 - weight) * alpha_min
            if np.linalg.eigvalsh(block_diag + alpha * block_off).min() > 0.0:
                alpha_min = alpha
            else:
                alpha_max = alpha

        alphas[c] = alpha_min

    # Both members of a pair are in the same component, so scaling row i by its own component's
    # alpha scales every entry correctly.
    return sp.csr_array(np.diag(diag) + dense_off * alphas[labels][:, None])


def mock_calc_omega(fam_df: pd.DataFrame, N: int, rel_df: pd.DataFrame,
                    unresidualized_phenotypes: np.ndarray):

    FAM_INDEX1_COL = "Fam Index 1"
    FAM_INDEX2_COL = "Fam Index 2"

    master_df = rel_df.copy()


    fam_idx = pd.MultiIndex.from_frame(fam_df[[bbf.FAM_FID_COL, bbf.FAM_IID_COL]])

    master_df[FAM_INDEX1_COL] = fam_idx.get_indexer(
        pd.MultiIndex.from_arrays([rel_df[grma_lib.KING_FID1_COL], rel_df[grma_lib.KING_IID1_COL]])
    )
    master_df[FAM_INDEX2_COL] = fam_idx.get_indexer(
        pd.MultiIndex.from_arrays([rel_df[grma_lib.KING_FID2_COL], rel_df[grma_lib.KING_IID2_COL]])
    )

    master_df[grma_lib.KING_REL_COL] = master_df[grma_lib.KING_REL_COL].map(grma_lib.INFTYPE_TO_DEG_MAP)

    master_df = master_df[[FAM_INDEX1_COL, FAM_INDEX2_COL, grma_lib.KING_REL_COL]]
    master_df = master_df[master_df[grma_lib.KING_REL_COL] <= grma_lib.SE_RELATEDNESS]

    rel_to_cov_dict = dict()
    for rel_deg in range(grma_lib.SE_RELATEDNESS + 1):
        temp_df = master_df[master_df[grma_lib.KING_REL_COL] == rel_deg]

        ind1 = np.concatenate([temp_df[FAM_INDEX1_COL].to_numpy(int), temp_df[FAM_INDEX2_COL].to_numpy(int)])
        ind2 = np.concatenate([temp_df[FAM_INDEX2_COL].to_numpy(int), temp_df[FAM_INDEX1_COL].to_numpy(int)])

        covar = np.cov(unresidualized_phenotypes[ind1], unresidualized_phenotypes[ind2], ddof=0)[0, 1] \
            if len(temp_df) > 0 else np.nan

        rel_to_cov_dict[rel_deg] = covar


    master_df["COV"] = master_df[grma_lib.KING_REL_COL].map(rel_to_cov_dict)

    variance = np.var(unresidualized_phenotypes)

    rows = np.concatenate([master_df[FAM_INDEX1_COL].to_numpy(int), master_df[FAM_INDEX2_COL].to_numpy(int), np.arange(N)])
    cols = np.concatenate([master_df[FAM_INDEX2_COL].to_numpy(int), master_df[FAM_INDEX1_COL].to_numpy(int), np.arange(N)])
    data = np.concatenate([master_df["COV"].to_numpy(), master_df["COV"].to_numpy(), variance * np.ones(N)])

    omega = sp.coo_array((data, (rows, cols)), shape=(N, N)).tocsr()
    shrunk_omega = mock_shrink_offdiag(omega)

    return shrunk_omega


#################################
def mock_gls_estimates(genotypes, gls_weights, weighted_phenotypes):
    """Works the GLS estimates out one SNP at a time

    The library does this with two einsums over a whole block at once, so the obvious scalar form
    is a genuinely separate route rather than a restatement of it.

    Args:
        genotypes: (snps, samples) residualized genotypes
        gls_weights: (samples, samples) weight matrix, sparse or dense
        weighted_phenotypes: (samples,) phenotypes already multiplied by the weights

    Returns:
        A tuple (betas, ses, XtWX) of (snps,) arrays, matching the library's return order
    """

    weights = gls_weights.toarray() if sp.issparse(gls_weights) else gls_weights

    XtWX, ses, betas = [], [], []
    for snp in genotypes:
        denominator = snp @ weights @ snp
        XtWX.append(denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            ses.append(1.0 / np.sqrt(denominator))
            betas.append((snp @ weighted_phenotypes) / denominator)

    return np.array(betas), np.array(ses), np.array(XtWX)


#################################
def mock_snipar_imputation(rng: np.random.Generator, num_snps: int, neither_observed: np.ndarray,
                           frequency_range: tuple = (0.05, 0.45)):
    """Builds imputed parental values the way snipar builds them

    This runs the construction FORWARDS -- start from the parental alleles snipar managed to
    observe, fill the rest in with the population frequency -- so that GRMA's correction, which
    runs it backwards, can be checked against the alleles it ought to recover.  Writing it out
    independently rather than reusing the library's arithmetic is the point: agreement then means
    the inversion is right, where a restatement of the same expression would prove nothing.

    What snipar stores depends on how many parents it had, and the two forms count different
    alleles.  Both were confirmed against a file generated from snipar's own example data.

        neither parent genotyped: imputed_par_gts holds the AVERAGE of the two parents, so all
            four parental alleles are in scope and the frequency contribution is halved along
            with everything else

        one parent genotyped: imputed_par_gts holds only the OTHER parent's genotype.  Two of the
            k observed alleles belong to the genotyped parent and are not in it at all, so only
            the remaining k - 2 are, and nothing is halved

    Either way the result is a genotype on a 0 to 2 scale, which is worth checking: a model that
    produced something outside that range would not be describing snipar.

    Args:
        rng: Seeded generator, so that a failure can be reproduced
        num_snps: Number of SNPs to generate
        neither_observed: Per family, whether snipar had no genotyped parent to work from
        frequency_range: Bounds to draw allele frequencies from.  The default stays clear of 0
            and of 0.5 so that a correction which was dropped, halved or doubled shows up as a
            clear difference rather than a marginal one.  Anything testing behaviour that depends
            on the frequency itself -- resolving a palindromic SNP, say, where values near 0.5
            are the hard case -- will want to say so explicitly.

    Returns:
        A tuple (imputed_gts, num_obs_par_alleles, standard_f, expected_contribution) where the
        first three are what the file would hold and the last is what
        calculate_imputed_contribution() should return for them
    """

    num_families = len(neither_observed)

    standard_f = rng.uniform(frequency_range[0], frequency_range[1], num_snps)

    # Where neither parent was genotyped, snipar can see as few as two of the four parental
    # alleles.  Where one was, its two alleles are always seen, so the count starts at three.
    num_obs_par_alleles = np.where(neither_observed[:, None],
                                   rng.choice([2.0, 3.0, 4.0], size=(num_families, num_snps)),
                                   rng.choice([3.0, 4.0], size=(num_families, num_snps)))

    # How many of the alleles that went INTO the imputed value carried the counted allele: all
    # four for an averaged pair, but only the ones belonging to the ungenotyped parent otherwise
    alleles_in_scope = np.where(neither_observed[:, None],
                                num_obs_par_alleles, num_obs_par_alleles - 2.0)
    observed_alleles = rng.integers(0, alleles_in_scope.astype(int) + 1).astype(float)

    # The alleles snipar could not see contribute the population frequency instead
    divisor = np.where(neither_observed, 2.0, 1.0)[:, None]
    imputed_gts = (observed_alleles + standard_f[None, :] * (4.0 - num_obs_par_alleles)) / divisor

    # GRMA removes the frequency and, for an averaged pair, doubles what is left so that it counts
    # for both parents.  What should come back is the allele count it started from.
    expected_contribution = observed_alleles.T

    return imputed_gts, num_obs_par_alleles, standard_f, expected_contribution


##################################

#################################
# Stage-by-stage mocks, chained by test/tcs.py to derive any intermediate a case does not state.
#
# Each is a separate route to one stage's result, written for clarity over speed.  None of them
# consumes an intermediate a case STATES -- only the case's inputs and its stated facts (AFFECTED,
# PARENTAL_MEANS, RETAINED_RSIDS) -- so that a stated intermediate compared against this chain is a
# genuine cross-check rather than the stated value checked against itself.

def _id_key(fid, iid):
    """Sample key compared as strings, since fixtures mix integer and string IDs"""
    return (str(fid), str(iid))


def mock_analysis_sample(fam_df: pd.DataFrame, id_list: pd.DataFrame = None,
                         pheno_df: pd.DataFrame = None, covar_df: pd.DataFrame = None
                         ) -> pd.DataFrame:
    """The samples kept for analysis, in fam file order, with their final phenotypes

    Kept means: in the id_list, the phenotype file and the covariate file wherever those are given,
    with no missing covariate, and with a non-missing phenotype -- taken from the phenotype file
    where there is one, otherwise from the fam file.  Phenotypes are then residualized on an
    intercept plus any covariates.  Production uses lstsq for that; this solves the normal
    equations, or takes the mean where there are no covariates -- separate routes to the same fit.

    Returns:
        The kept rows of fam_df, reindexed from 0, with FAM_PHENO_COL holding final phenotypes
    """

    keys = [_id_key(f, i) for f, i in zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL])]
    keep = set(keys)
    phenos = dict(zip(keys, fam_df[bbf.FAM_PHENO_COL].astype(float)))

    if id_list is not None:
        keep &= {_id_key(f, i) for f, i in zip(id_list.iloc[:, 0], id_list.iloc[:, 1])}

    if pheno_df is not None:
        from_file = {_id_key(f, i): float(p)
                     for f, i, p in pheno_df.iloc[:, :3].itertuples(index=False)}
        keep &= set(from_file)
        phenos = {k: from_file.get(k, np.nan) for k in keys}

    covars = None
    if covar_df is not None:
        covars = {_id_key(row[0], row[1]): np.asarray(row[2:], dtype=float)
                  for row in covar_df.itertuples(index=False)}
        covars = {k: v for k, v in covars.items() if not np.isnan(v).any()}
        keep &= set(covars)

    rows = [n for n, k in enumerate(keys) if k in keep and not np.isnan(phenos[k])]
    analysis_df = fam_df.iloc[rows].reset_index(drop=True).copy()
    y = np.array([phenos[keys[n]] for n in rows], dtype=float)

    # Residualized on an intercept plus any covariates (notes §13.5).  Without covariates that is
    # just the mean, taken explicitly here where production runs the same lstsq either way.
    if covars is not None:
        X = np.column_stack([np.vstack([covars[keys[n]] for n in rows]), np.ones(len(rows))])
        y = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    else:
        y = y - np.mean(y)

    analysis_df[bbf.FAM_PHENO_COL] = y
    return analysis_df


def mock_fam_trios(fam_df: pd.DataFrame, analysis_df: pd.DataFrame) -> list:
    """IIDs of the analysis samples whose fam file lists two parents who are both samples

    Parents are looked up within the child's FID, and "0" means no parent.  A parent need not be
    in the analysis sample.  No KING validation: this is only for cross-checking a case's stated
    AFFECTED, and a case whose pedigree KING contradicts should fail loudly in production anyway.
    """

    samples = {_id_key(f, i) for f, i in zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL])}
    analysis = {_id_key(f, i)
                for f, i in zip(analysis_df[bbf.FAM_FID_COL], analysis_df[bbf.FAM_IID_COL])}

    trios = []
    for fid, iid, dad, mum in zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL],
                                  fam_df[bbf.FAM_IIDF_COL], fam_df[bbf.FAM_IIDM_COL]):
        if (_id_key(fid, iid) in analysis and str(dad) != "0" and str(mum) != "0" and
                _id_key(fid, dad) in samples and _id_key(fid, mum) in samples):
            trios.append(str(iid))
    return trios


def mock_fam_parental_means(fam_df: pd.DataFrame, G: np.ndarray, affected_iids: list) -> dict:
    """The mean of each affected individual's two fam-file parents, from RAW genotypes

    Returns:
        dict of IID -> (snps,) array over every SNP of G
    """

    pos = {_id_key(f, i): n for n, (f, i) in
           enumerate(zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL]))}
    rows = {str(i): (f, d, m) for f, i, d, m in zip(
        fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL],
        fam_df[bbf.FAM_IIDF_COL], fam_df[bbf.FAM_IIDM_COL])}

    means = {}
    for iid in affected_iids:
        fid, dad, mum = rows[str(iid)]
        means[str(iid)] = (G[:, pos[_id_key(fid, dad)]] + G[:, pos[_id_key(fid, mum)]]) / 2.0
    return means


def mock_restrict_rel_df(rel_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """The relatedness pairs between analysis samples, which are the only ones that play a part"""

    kept = {_id_key(f, i) for f, i in zip(analysis_df[bbf.FAM_FID_COL],
                                          analysis_df[bbf.FAM_IID_COL])}
    mask = [(_id_key(f1, i1) in kept and _id_key(f2, i2) in kept) for f1, i1, f2, i2 in zip(
        rel_df[grma_lib.KING_FID1_COL], rel_df[grma_lib.KING_IID1_COL],
        rel_df[grma_lib.KING_FID2_COL], rel_df[grma_lib.KING_IID2_COL])]
    return rel_df[np.asarray(mask, dtype=bool)].reset_index(drop=True)


def mock_R(rel_info: list, analysis_df: pd.DataFrame, affected_iids: list) -> np.ndarray:
    """R row by row: an affected sample passes through unchanged; everyone else is centered on
    their group, which may include affected relatives at their DIFFERENCED values"""

    is_affected = analysis_df[bbf.FAM_IID_COL].astype(str).isin(affected_iids).to_numpy()
    R = np.identity(len(rel_info))
    for person, group in enumerate(rel_info):
        if not is_affected[person]:
            R[person, group] -= 1.0 / len(group)
    return R


def mock_resid_genos(G: np.ndarray, fam_df: pd.DataFrame, analysis_df: pd.DataFrame,
                     rel_info: list, affected_iids: list, parental_means: dict,
                     retained_snps: np.ndarray) -> np.ndarray:
    """Residualized genotypes, (retained snps, analysis samples), NaN set to zero

    Every parental mean is subtracted from a copy of the RAW genotypes, so someone who is both
    affected and a parent is differenced against raw values.  Residualization then uses explicit
    group means, so that a missing value spoils only the rows whose groups actually contain it --
    as with production's sparse R, and unlike a dense product, where NaN times zero would spread it
    everywhere.
    """

    all_keys = [_id_key(f, i) for f, i in zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL])]
    iid_pos = {str(i): n for n, i in enumerate(fam_df[bbf.FAM_IID_COL])}

    G = G.copy()
    for iid in affected_iids:
        G[:, iid_pos[iid]] -= parental_means[iid]

    positions = [all_keys.index(_id_key(f, i)) for f, i in
                 zip(analysis_df[bbf.FAM_FID_COL], analysis_df[bbf.FAM_IID_COL])]
    G = G[np.asarray(retained_snps)][:, positions]

    is_affected = analysis_df[bbf.FAM_IID_COL].astype(str).isin(affected_iids).to_numpy()
    resid = G.copy()
    for person, group in enumerate(rel_info):
        if not is_affected[person]:
            resid[:, person] = G[:, person] - np.mean(G[:, group], axis=1)
    resid[np.isnan(resid)] = 0.0
    return resid


def mock_snipar_from_trios(fam_df: pd.DataFrame, G: np.ndarray, bim_df: pd.DataFrame,
                           trio_iids: list, mode: str) -> dict:
    """snipar data describing a case's fam-file trios, built from the TRUE genotypes in the bed

    For the invariance check of notes §15.3: GRMA run on this file must give exactly the results of
    fam-file trio mode.  That only holds under three conditions, all met here:

      - every family has num_observed_parental_alleles = 4, so production removes no frequency
        fill-in and recovers the stored truth exactly
      - an "imputed" parent is named by an ID that resolves to nobody in the fam file, so it is not
        ALSO taken from the bed (notes §13.4); the parent stays in the fileset, so omega and every
        other row of R are unchanged
      - a missing genotype stays missing (NaN), so it propagates as it does in trio mode

    The file describes exactly the given trios and nobody else, since anyone else it described
    could become affected in imputed mode when they are not in trio mode.

    Args:
        fam_df: The case's fam file
        G: (snps, samples) genotypes, every SNP of bim_df and every sample of fam_df
        bim_df: The case's bim file; the file shares its SNPs, order and alleles
        trio_iids: IIDs of the children to describe -- the case's fam-mode AFFECTED
        mode: "one" -- father observed, mother imputed as her true genotype; or
              "both" -- neither observed, snipar's row holding the parents' true average

    Returns:
        Keyword arguments for snipar_hdf5.write_snipar_file()
    """

    if mode not in ("one", "both"):
        raise ValueError(f"mode must be 'one' or 'both', not {mode!r}")

    pos = {_id_key(f, i): n for n, (f, i) in
           enumerate(zip(fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL]))}
    rows = {str(i): (f, str(d), str(m)) for f, i, d, m in zip(
        fam_df[bbf.FAM_FID_COL], fam_df[bbf.FAM_IID_COL],
        fam_df[bbf.FAM_IIDF_COL], fam_df[bbf.FAM_IIDM_COL])}

    # One snipar family per sibship, named so that they cannot collide with any sample ID
    sibships = {}
    for iid in trio_iids:
        fid, dad, mum = rows[str(iid)]
        sibships.setdefault((str(fid), dad, mum), []).append(str(iid))

    families, imputed, pedigree = [], [], []
    for (fid, dad, mum), kids in sibships.items():
        family = f"snipar_{fid}_{dad}_{mum}"
        g_dad, g_mum = G[:, pos[(fid, dad)]], G[:, pos[(fid, mum)]]
        if mode == "one":
            imputed.append(g_mum)
            named = (dad, f"{mum}_hidden", "True", "False")
        else:
            imputed.append((g_dad + g_mum) / 2.0)
            named = (f"{dad}_hidden", f"{mum}_hidden", "False", "False")
        families.append(family)
        pedigree += [[family, kid, *named] for kid in kids]

    imputed = np.array(imputed)
    status = [True, False, True] if mode == "one" else [False, False, False]
    bim_values = np.array([["1", r, "0", str(bp), a1, a2] for r, bp, a1, a2 in zip(
        bim_df[bbf.BIM_RSID_COL], bim_df[bbf.BIM_BP_COL],
        bim_df[bbf.BIM_A1_COL], bim_df[bbf.BIM_A2_COL])])

    return dict(imputed_gts=imputed,
                num_obs_par_alleles=np.full(imputed.shape, 4.0),
                standard_f=np.full(G.shape[0], 0.3),   # Irrelevant at 4 alleles observed
                parental_status=np.array([status] * len(families)),
                pedigree=np.array(pedigree), families=np.array(families),
                bim_values=bim_values,
                # Narrow, so that a case's blocks cross chunk boundaries
                chunk_snps=min(2, G.shape[0]))
