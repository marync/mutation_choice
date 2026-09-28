import numpy as np
from scipy.special import ndtri

MAX_TABLE_BYTES = 5 * 1024 * 1024


def log(msg):
    print(msg, flush=True)


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def make_nk_landscape(p, K, sigma, rng):
    if K == 1:
        nb = np.arange(p)[:, None].astype(np.int64)
    else:
        nb = np.empty((p, K), dtype=np.int64)
        for i in range(p):
            nb[i] = rng.choice(p, size=K, replace=False)

    tbytes = p * (2 ** K) * 8
    # table entries unscaled (plain sigma); 1/sqrt(p) normalization applied in nk_fitness
    if tbytes <= MAX_TABLE_BYTES:
        tbl = rng.normal(loc=0.0, scale=sigma, size=(p, 2 ** K))
        return {"p": p, "K": K, "sigma": sigma, "neigh": nb, "mode": "dense", "table": tbl}
    else:
        salt = int(rng.integers(0, 2 ** 62, dtype=np.int64))
        return {"p": p, "K": K, "sigma": sigma, "neigh": nb, "mode": "hashed", "salt": salt}


def _pattern_index(bits):
    K = bits.shape[-1]
    w = (1 << np.arange(K)).astype(np.int64)
    return (bits.astype(np.int64) * w).sum(axis=-1)


def _hash_bits_directly(bits2d, hidx, salt):
    n, K = bits2d.shape
    h = np.full(n, np.uint64(salt), dtype=np.uint64) ^ (hidx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    mult = np.uint64(6364136223846793005)
    bu = bits2d.astype(np.uint64)
    for c in range(K):
        h = h * mult + bu[:, c] + np.uint64(1442695040888963407)
        h ^= h >> np.uint64(29)
    h ^= h >> np.uint64(32)
    h = h * np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(33)
    return h


def _zobrist_value(salt, hidx, pidx):
    # deterministic uint64 R[host, position] via splitmix64-style mix; lets a
    # single bit flip update a fingerprint with one XOR instead of a full rehash
    key = (np.uint64(salt)
           ^ (hidx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
           ^ (pidx.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9) + np.uint64(1)))
    key ^= key >> np.uint64(30); key = key * np.uint64(0xBF58476D1CE4E5B9)
    key ^= key >> np.uint64(27); key = key * np.uint64(0x94D049BB133111EB)
    key ^= key >> np.uint64(31)
    return key


def _hash_bits_zobrist(bits2d, hidx_flat, salt):
    n, K = bits2d.shape
    pidx = np.broadcast_to(np.arange(K), (n, K))
    hidx2d = np.broadcast_to(hidx_flat[:, None], (n, K))
    R = _zobrist_value(salt, hidx2d, pidx)
    masked = np.where(bits2d == 1, R, np.uint64(0))
    return np.bitwise_xor.reduce(masked, axis=-1)


def precompute_founder_flip_state_zobrist(founder, lc):
    # K>62 setup: founder fingerprint (p,) + (p_loci, p_hosts) Zobrist table
    K = lc["K"]
    if K <= 62:
        raise ValueError("precompute_founder_flip_state_zobrist only supports K > 62")
    p = lc["p"]
    nb = lc["neigh"]
    fbits = ((founder[nb] + 1) // 2).astype(np.int64)
    hidx2d = np.broadcast_to(np.arange(p)[:, None], (p, K))
    pidx2d = np.broadcast_to(np.arange(K)[None, :], (p, K))
    R = _zobrist_value(lc["salt"], hidx2d, pidx2d)
    f_fp = np.bitwise_xor.reduce(np.where(fbits == 1, R, np.uint64(0)), axis=-1)

    hidx_flat = np.repeat(np.arange(p), K)
    lidx_flat = nb.ravel()
    zmat = np.zeros((p, p), dtype=np.uint64)
    zmat[lidx_flat, hidx_flat] = R.ravel()
    return f_fp, zmat


def fast_batch_fitness_from_mutations_zobrist(mut_batch, f_fp, zmat, lc):
    p = lc["p"]
    n = mut_batch.shape[0]
    shifts = zmat[mut_batch]
    fp_shift = np.bitwise_xor.reduce(shifts, axis=1)
    new_fp = f_fp[None, :].astype(np.uint64) ^ fp_shift

    u = (new_fp >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = ndtri(u)
    contrib = z * lc["sigma"]
    return contrib.sum(axis=1) / np.sqrt(p)


def _hash_normal(salt, lidx, pidx, sigma):
    key = (np.uint64(salt) ^ (lidx.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
           ^ (pidx.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)))
    key ^= key >> np.uint64(30); key = key * np.uint64(0xBF58476D1CE4E5B9)
    key ^= key >> np.uint64(27); key = key * np.uint64(0x94D049BB133111EB)
    key ^= key >> np.uint64(31)
    u = (key >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = ndtri(u)
    return z * sigma


def precompute_founder_flip_state(founder, lc):
    K = lc["K"]
    if K > 62:
        raise ValueError("precompute_founder_flip_state only supports K <= 62")
    p = lc["p"]
    nb = lc["neigh"]
    fbits = ((founder[nb] + 1) // 2).astype(np.int64)
    f_pat = _pattern_index(fbits)

    w_pos = (1 - 2 * fbits) * (1 << np.arange(K, dtype=np.int64))
    hidx = np.repeat(np.arange(p), K)
    lidx = nb.ravel()
    dmat = np.zeros((p, p), dtype=np.int64)  
    dmat[lidx, hidx] = w_pos.ravel()
    return f_pat, dmat


def fast_batch_fitness_from_mutations(mut_batch, f_pat, dmat, lc):
    p = lc["p"]
    n = mut_batch.shape[0]
    shift = dmat[mut_batch].sum(axis=1)
    new_pat = f_pat[None, :].astype(np.int64) + shift

    if lc["mode"] == "dense":
        tbl = lc["table"]
        contrib = tbl[np.arange(p)[None, :].repeat(n, axis=0), new_pat]
    else:
        lidx = np.broadcast_to(np.arange(p), (n, p))
        contrib = _hash_normal(lc["salt"], lidx, new_pat, lc["sigma"])

    return contrib.sum(axis=1) / np.sqrt(p)


def nk_fitness(genotypes, lc):
    # fitness = (1/sqrt(p)) * sum_i f_i(x_i)
    genotypes = np.atleast_2d(genotypes)
    n, p = genotypes.shape
    K = lc["K"]
    nb = lc["neigh"]
    bits = ((genotypes[:, nb] + 1) // 2).astype(np.int64)

    if lc["mode"] == "dense":
        pat = _pattern_index(bits)
        tbl = lc["table"]
        contrib = tbl[np.arange(p)[None, :].repeat(n, axis=0), pat]
    else:
        lidx = np.broadcast_to(np.arange(p), (n, p))
        if K <= 62:
            pat = _pattern_index(bits)
            contrib = _hash_normal(lc["salt"], lidx, pat, lc["sigma"])
        else:
            bits_flat = bits.reshape(n * p, K)
            lidx_flat = lidx.reshape(n * p)
            h = _hash_bits_zobrist(bits_flat, lidx_flat, lc["salt"])
            u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = ndtri(u)
            contrib = (z * lc["sigma"]).reshape(n, p)

    return contrib.sum(axis=1) / np.sqrt(p)


def one_step_neighbors(founder):
    p = founder.shape[0]
    nbrs = np.tile(founder, (p, 1))
    idx = np.arange(p)
    nbrs[idx, idx] *= -1
    return nbrs


def top_muts(founder, lc, tot_muts):
    Y0 = nk_fitness(founder[None, :], lc)[0]
    eff = nk_fitness(one_step_neighbors(founder), lc) - Y0
    return np.argsort(eff)[-tot_muts:]


def bottom_muts(founder, lc, tot_muts):
    Y0 = nk_fitness(founder[None, :], lc)[0]
    eff = nk_fitness(one_step_neighbors(founder), lc) - Y0
    return np.argsort(eff)[:tot_muts]


def combinatorial_library_flip(founder, mut_sites):
    m = len(mut_sites)
    lib = np.tile(founder, (2 ** m, 1))
    for pat in range(2 ** m):
        for bit, locus in enumerate(mut_sites):
            if (pat >> bit) & 1:
                lib[pat, locus] *= -1
    return lib


def nk_adaptive_walk_with_landscape_retry(p, K, sigma, tot_muts, walk_ntries, rng,
                                           force_mode=None, max_outer_retries=50):
    for _o in range(max_outer_retries):
        lc = make_nk_landscape(p, K, sigma, rng=rng)
        founder = rng.choice([-1, 1], size=p)

        for _a in range(walk_ntries):
            curr = founder.copy()
            Ycurr = nk_fitness(curr[None, :], lc)[0]
            fixloci = []
            Ytraj = [Ycurr]
            for _s in range(tot_muts):
                nbrs = one_step_neighbors(curr)
                Ynbr = nk_fitness(nbrs, lc)
                s_all = Ynbr - Ycurr
                ben = np.where(s_all > 0)[0]
                if ben.size == 0:
                    break
                locus = rng.choice(ben)
                curr = nbrs[locus].copy()
                Ycurr = float(Ynbr[locus])
                fixloci.append(int(locus))
                Ytraj.append(Ycurr)
            if len(fixloci) == tot_muts:
                return lc, founder, np.array(Ytraj), fixloci, True
        # all walk_ntries attempts failed; redraw landscape+founder
    return None, None, None, None, False


def sample_conditioned_mutation_set_threshold(founder, traj_sites, lc, sites, epsilon, nbatch, ntries, rng):
    p = founder.shape[0]
    tot_muts = len(traj_sites)
    tgt_geno = founder.copy()
    tgt_geno[list(traj_sites)] *= -1
    tgt_fit = nk_fitness(tgt_geno[None, :], lc)[0]

    fast = lc["K"] <= 62
    fast_z = lc["K"] > 62
    if fast:
        f_pat, dmat = precompute_founder_flip_state(founder, lc)
    elif fast_z:
        f_fp, zmat = precompute_founder_flip_state_zobrist(founder, lc)

    for _t in range(ntries):
        cand = np.array([rng.choice(sites, size=tot_muts, replace=False) for _ in range(nbatch)])
        if fast:
            cand_fit = fast_batch_fitness_from_mutations(cand, f_pat, dmat, lc)
        elif fast_z:
            cand_fit = fast_batch_fitness_from_mutations_zobrist(cand, f_fp, zmat, lc)
        else:
            genos = np.tile(founder, (nbatch, 1))
            rows = np.repeat(np.arange(nbatch), tot_muts)
            genos[rows, cand.ravel()] *= -1
            cand_fit = nk_fitness(genos, lc)
        diffs = np.abs(cand_fit - tgt_fit)
        best = np.argmin(diffs)
        if diffs[best] < epsilon:
            return cand[best]
    return None


# ===================== additional metrics: model fitting, local maxima, accessible paths =====================
from itertools import combinations, permutations


def build_epistasis_design(X, d):
    X = np.asarray(X)
    N, K = X.shape
    terms = [()]
    for r in range(1, d + 1):
        terms.extend(combinations(range(K), r))
    A = np.ones((N, len(terms)), dtype=float)
    for col, S in enumerate(terms[1:], start=1):
        A[:, col] = np.prod(X[:, S], axis=1)
    return A, terms


def fit_epistasis_coeffs(X, F, d, ridge_lambda=0.0):
    A, terms = build_epistasis_design(X, d)
    F = np.asarray(F, dtype=float)
    if ridge_lambda == 0.0:
        beta, *_ = np.linalg.lstsq(A, F, rcond=None)
    else:
        ATA = A.T @ A
        beta = np.linalg.solve(ATA + ridge_lambda * np.eye(ATA.shape[0]), A.T @ F)
    return beta, terms


def predict_from_betas(X_new, beta, terms):
    X_new = np.asarray(X_new)
    A_new = np.ones((X_new.shape[0], len(terms)), dtype=float)
    for col, S in enumerate(terms[1:], start=1):
        A_new[:, col] = np.prod(X_new[:, S], axis=1)
    return A_new @ np.asarray(beta, dtype=float)


def count_total_local_maxima(lib, fitness):
    lib = np.asarray(lib)
    fitness = np.asarray(fitness)
    g2i = {tuple(g): i for i, g in enumerate(lib)}
    L = lib.shape[1]
    vsites, amap = [], {}
    for j in range(L):
        vals = np.unique(lib[:, j])
        if len(vals) == 2:
            vsites.append(j)
            amap[j] = vals
    n_max = 0
    for i, g in enumerate(lib):
        fi = fitness[i]
        is_max = True
        for j in vsites:
            nb = g.copy()
            a0, a1 = amap[j]
            nb[j] = a1 if g[j] == a0 else a0
            idx = g2i.get(tuple(nb))
            if idx is None:
                continue
            if fitness[idx] > fi:
                is_max = False
                break
        if is_max:
            n_max += 1
    return n_max


def count_accessible_paths_to_full_mutant(lib, fitness, founder, strict=True):
    lib = np.asarray(lib)
    fitness = np.asarray(fitness)
    founder = np.asarray(founder)
    g2i = {tuple(g): i for i, g in enumerate(lib)}
    vsites = [j for j in range(lib.shape[1]) if len(np.unique(lib[:, j])) == 2]
    full_mut = founder.copy()
    for j in vsites:
        vals = np.unique(lib[:, j])
        full_mut[j] = vals[1] if founder[j] == vals[0] else vals[0]
    n_acc = 0
    for order in permutations(vsites):
        curr = founder.copy()
        prev_fit = fitness[g2i[tuple(curr)]]
        ok = True
        for site in order:
            curr[site] = full_mut[site]
            idx = g2i.get(tuple(curr))
            if idx is None:
                ok = False
                break
            cfit = fitness[idx]
            if (cfit <= prev_fit) if strict else (cfit < prev_fit):
                ok = False
                break
            prev_fit = cfit
        if ok:
            n_acc += 1
    return n_acc
