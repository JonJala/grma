"""
An independent implementation of GRMA, written from the specification, for checking grma2_lib.

It shares no code with grma2_lib, and deliberately works differently wherever it can: dense
matrices instead of sparse ones, plain loops over samples and SNPs instead of vectorized arrays, an
eigendecomposition of each block of Omega instead of a screen followed by ARPACK, one
pseudo-inverse of each regression's whole covariance instead of a block-wise one, and the covariance
of the estimates computed literally as W Omega W'.  Agreement with grma2_lib is then a real
cross-check, not a restatement.

It assumes inputs that pass grma2_lib's checks: it validates nothing.

The one entry point is mock_grma2(), taking the same inputs as grma2_lib.grma2() but always as
DataFrames and arrays, plus snipar's data as arrays rather than a file.
"""

import numpy as np

MISSING_PHENOTYPE = -9.0
# Below this, a relative regression's residual is rounding noise
RESIDUAL_TOLERANCE = 1e-9
# snipar's values must be this close to a whole number of alleles
SNIPAR_TOLERANCE = 0.01
EPSILON = 1e-10                     # Omega's floor

RELATIONSHIP_OF_INFTYPE = {"PO" : "PO", "FS" : "FS", "2nd" : "2ND", "3rd" : "3RD",
                           "Dup/MZTwin" : "DUP"}
REGRESSIONS = ("PO", "FS", "2ND", "3RD")
MIN_DEGREE = {"FS" : 0, "2ND" : 2, "3RD" : 3}


def _number(value):
    """A phenotype or covariate as a float, NaN where missing"""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if x == MISSING_PHENOTYPE else x


def mock_grma2(*, fam, king, bim, genotypes, rel_degree, pheno=None, covar=None, id_list=None,
               snp_list=None, snipar=None):
    """GRMA's estimates, from the specification

    Args:
        fam, king, bim: DataFrames, with their formats' column names
        genotypes: (bim snps, fam samples), NaN where missing
        rel_degree: 0 to 3
        pheno, covar, id_list, snp_list: DataFrames as grma2_io reads them, or None
        snipar: snipar's data as given to snipar_hdf5.write_snipar_file(), or None

    Returns:
        A tuple (rsids, estimates): the analysed SNPs' rsids, in bim order; and label -> (betas,
        ses), one value per analysed SNP, NaN where there is none
    """

    ids = [(str(f), str(i)) for f, i in zip(fam["FID"], fam["IID"])]
    n_fam = len(ids)
    where = {key : n for n, key in enumerate(ids)}

    # -- Who is in the analysis ---------------------------------------------------------------
    if pheno is not None:
        by_id = {(str(f), str(i)) : _number(p) for f, i, p in
                 zip(pheno["FID"], pheno["IID"], pheno["PHENO"])}
        phenotype = np.array([by_id.get(key, np.nan) for key in ids])
    else:
        phenotype = np.array([_number(p) for p in fam["PHENO"]])
    in_analysis = ~np.isnan(phenotype)
    covariates = None
    if covar is not None:
        covar_cols = [c for c in covar.columns if c not in ("FID", "IID")]
        by_id = {(str(f), str(i)) : [_number(row[c]) for c in covar_cols]
                 for f, i, (_, row) in zip(covar["FID"], covar["IID"], covar.iterrows())}
        covariates = np.array([by_id.get(key, [np.nan] * len(covar_cols)) for key in ids])
        in_analysis &= ~np.isnan(covariates).any(axis=1)
    if id_list is not None:
        listed = {(str(f), str(i)) for f, i in zip(id_list["FID"], id_list["IID"])}
        in_analysis &= np.array([key in listed for key in ids])

    # -- Pairs: relationships between fam samples, at least one in the analysis ---------------
    pairs = {}
    for f1, i1, f2, i2, t in zip(king["FID1"], king["ID1"], king["FID2"], king["ID2"],
                                 king["InfType"]):
        a, b = where.get((str(f1), str(i1))), where.get((str(f2), str(i2)))
        if a is None or b is None or t not in RELATIONSHIP_OF_INFTYPE:
            continue
        if in_analysis[a] or in_analysis[b]:
            pairs[(min(a, b), max(a, b))] = RELATIONSHIP_OF_INFTYPE[t]

    # Duplicates: in each cluster keep the earliest in the analysis, else the earliest
    kept_for = list(range(n_fam))
    def root(x):
        while kept_for[x] != x:
            x = kept_for[x]
        return x
    for (a, b), r in pairs.items():
        if r == "DUP":
            kept_for[root(max(a, b))] = root(min(a, b))
    clusters = {}
    for x in range(n_fam):
        clusters.setdefault(root(x), []).append(x)
    replacement = list(range(n_fam))
    for members in clusters.values():
        if len(members) > 1:
            eligible = [m for m in members if in_analysis[m]]
            keep = min(eligible) if eligible else min(members)
            for m in members:
                if m != keep:
                    replacement[m] = keep
                    in_analysis[m] = False
    dropped = {m for m in range(n_fam) if replacement[m] != m}
    pairs = {p : r for p, r in pairs.items() if r != "DUP" and not (set(p) & dropped)
             and (in_analysis[p[0]] or in_analysis[p[1]])}

    def related(x, relationship):
        return [b if a == x else a for (a, b), r in pairs.items()
                if r == relationship and x in (a, b)]

    # -- Parents, from snipar's pedigree ------------------------------------------------------
    by_iid = {}
    for n, (_, iid) in enumerate(ids):
        by_iid.setdefault(iid, []).append(n)
    children = {}                    # child -> (father, mother, snipar family row)
    if snipar is not None:
        families = sorted(snipar["families"])
        order = np.argsort(np.array(snipar["families"]), kind="stable")
        for record in snipar["pedigree"]:
            fid, iid, father, mother = record[0], record[1], record[2], record[3]
            if iid not in by_iid or not in_analysis[by_iid[iid][0]]:
                continue
            def parent(p):
                if p == "0" or p not in by_iid:
                    return -1
                return replacement[by_iid[p][0]]
            f, m = parent(father), parent(mother)
            if f == -1 and m == -1:
                continue
            row = families.index(fid) if fid in families else -1
            children[by_iid[iid][0]] = (f, m, row)
        # write_snipar_file sorts families; read the per-family rows in that order
        imputed_by_family = np.asarray(snipar["imputed_gts"])[order]
        counts_by_family = np.asarray(snipar["num_obs_par_alleles"])[order]

    # -- Groups --------------------------------------------------------------------------------
    group = [None] * n_fam
    for x in range(n_fam):
        if not in_analysis[x]:
            continue
        if x in children:
            group[x] = "PO"
        elif any(in_analysis[y] for y in related(x, "FS")):
            group[x] = "FS"
        elif rel_degree >= 2 and any(in_analysis[y] for y in related(x, "2ND")):
            group[x] = "2ND"
        elif rel_degree >= 3 and any(in_analysis[y] for y in related(x, "3RD")):
            group[x] = "3RD"
        else:
            group[x] = "NOT_FOCAL"

    # -- Omega ---------------------------------------------------------------------------------
    parents_used = {p for f, m, _ in children.values() for p in (f, m) if p != -1}
    omega_members = [x for x in range(n_fam) if in_analysis[x] and
                     (group[x] not in (None, "NOT_FOCAL") or x in parents_used)]
    pos = {x : k for k, x in enumerate(omega_members)}
    n_om = len(omega_members)

    y = phenotype[omega_members]
    X = np.ones((n_om, 1))
    if covariates is not None:
        X = np.column_stack([covariates[omega_members], X])
    y = y - X @ np.linalg.solve(X.T @ X, X.T @ y)

    variance = float(np.mean((y - y.mean()) ** 2))
    covariance = {}
    for r in REGRESSIONS:
        rp = [(pos[a], pos[b]) for (a, b), rr in pairs.items() if rr == r and a in pos and b in pos]
        if rp:
            everyone = sorted({k for p in rp for k in p})
            mean = np.mean(y[everyone])
            covariance[r] = np.mean([(y[a] - mean) * (y[b] - mean) for a, b in rp])
    off = np.zeros((n_om, n_om))
    for (a, b), r in pairs.items():
        if a in pos and b in pos and r in covariance:
            off[pos[a], pos[b]] = off[pos[b], pos[a]] = covariance[r]

    # Shrink each connected block of the off-diagonal just enough to keep it positive definite
    label = list(range(n_om))
    def find(x):
        while label[x] != x:
            x = label[x]
        return x
    for a in range(n_om):
        for b in range(n_om):
            if off[a, b] != 0:
                label[find(a)] = find(b)
    for block_root in set(find(x) for x in range(n_om)):
        block = [x for x in range(n_om) if find(x) == block_root]
        if len(block) < 2:
            continue
        lowest = np.linalg.eigvalsh(off[np.ix_(block, block)]).min()
        if lowest < 0:
            alpha = min((EPSILON - variance) / lowest, 1.0)
            off[np.ix_(block, block)] *= alpha
    omega = variance * np.eye(n_om) + off

    # -- The regressions: phenotype maps ------------------------------------------------------
    maps = {}
    po_children = [x for x in omega_members if group[x] == "PO"]
    if po_children:
        A = np.zeros((len(po_children), n_om))
        for row, x in enumerate(po_children):
            A[row, pos[x]] = 1.0
        maps["PO"] = A
    for r in ("FS", "2ND", "3RD"):
        if rel_degree < MIN_DEGREE[r]:
            continue
        focal = [x for x in omega_members if group[x] == r]
        if not focal:
            continue
        R = np.zeros((len(focal), n_om))
        for row, x in enumerate(focal):
            members = [x] + [z for z in related(x, r) if z in pos]
            for z in members:
                R[row, pos[z]] -= 1.0 / len(members)
            R[row, pos[x]] += 1.0
        maps[r] = R

    # -- SNPs ----------------------------------------------------------------------------------
    rows = list(range(len(bim)))
    if snp_list is not None:
        listed = set(str(s) for s in snp_list["RSID"])
        rows = [s for s in rows if str(bim["RSID"].iloc[s]) in listed]
    snipar_at, snipar_flip = {}, {}
    if snipar is not None:
        values = snipar["bim_values"]
        index = {str(v[1]) : k for k, v in enumerate(values)}
        for s in rows:
            rsid = str(bim["RSID"].iloc[s])
            if rsid not in index:
                continue
            a1, a2 = str(bim["A1"].iloc[s]), str(bim["A2"].iloc[s])
            s1, s2 = str(values[index[rsid]][4]), str(values[index[rsid]][5])
            if (a1, a2) == (s1, s2):
                snipar_at[s], snipar_flip[s] = index[rsid], False
            elif (a1, a2) == (s2, s1):
                snipar_at[s], snipar_flip[s] = index[rsid], True

    # -- Per SNP --------------------------------------------------------------------------------
    names = [r for r in REGRESSIONS if r in maps]
    allowed = [r for r in REGRESSIONS[1:] if rel_degree >= MIN_DEGREE[r]]
    included = (["PO"] if snipar is not None else []) + allowed
    estimates = [tuple(included[:n + 1]) for n in range(len(included))]
    results = {"_".join(e) : (np.full(len(rows), np.nan), np.full(len(rows), np.nan))
               for e in estimates}
    phenos_of = {r : maps[r] @ y for r in names}
    weights_of = {r : np.linalg.pinv(maps[r] @ omega @ maps[r].T, rcond=1e-10, hermitian=True)
                  for r in names}

    for out, s in enumerate(rows):
        g = genotypes[s]
        Z = {}
        for r in names:
            if r == "PO":
                z = np.zeros(len(po_children))
                for row, x in enumerate(po_children):
                    f, m, fam_row = children[x]
                    if f != -1 and m != -1:
                        value = g[x] - (g[f] + g[m]) / 2
                    else:
                        if s not in snipar_at or fam_row == -1:
                            continue
                        observed = g[f if f != -1 else m]
                        k = counts_by_family[fam_row][snipar_at[s]]
                        v = float(imputed_by_family[fam_row][snipar_at[s]])
                        freq = float(snipar["standard_f"][snipar_at[s]])
                        if snipar_flip[s]:
                            v, freq = 2.0 - v, 1.0 - freq
                        if k == 3 and abs((v - freq) - round(v - freq)) <= SNIPAR_TOLERANCE:
                            value = g[x] - observed / 2 - round(v - freq)
                        elif k == 4 and abs(v - round(v)) <= SNIPAR_TOLERANCE:
                            value = g[x] - (observed + round(v)) / 2
                        else:
                            continue
                    z[row] = 0.0 if np.isnan(value) else value
                Z[r] = z
            else:
                z = np.zeros(maps[r].shape[0])
                for row in range(len(z)):
                    used = np.flatnonzero(maps[r][row])
                    members = [omega_members[k] for k in used]
                    if np.isnan(g[members]).any():
                        continue
                    value = maps[r][row, used] @ g[members]
                    z[row] = 0.0 if abs(value) < RESIDUAL_TOLERANCE else value
                Z[r] = z

        beta, w = {}, {}
        for r in names:
            d = Z[r] @ weights_of[r] @ Z[r]
            if d > 0:
                beta[r] = (Z[r] @ weights_of[r] @ phenos_of[r]) / d
                w[r] = maps[r].T @ weights_of[r] @ Z[r] / d       # Over all of Omega
        for e in estimates:
            present = [r for r in e if r in beta]
            if not present:
                continue
            W = np.array([w[r] for r in present])
            C = W @ omega @ W.T
            inverse = np.linalg.pinv(C, hermitian=True)
            a = inverse.sum(axis=1) / inverse.sum()
            label = "_".join(e)
            results[label][0][out] = a @ np.array([beta[r] for r in present])
            results[label][1][out] = np.sqrt(a @ C @ a)

    return [str(bim["RSID"].iloc[s]) for s in rows], results
