import sys
import time
import numpy as np
from copy import deepcopy
from sklearn.metrics import r2_score
sys.path.insert(0, '.')
from nk_pipeline import (
    one_step_neighbors, combinatorial_library_flip, fit_epistasis_coeffs,
    predict_from_betas, count_total_local_maxima, count_accessible_paths_to_full_mutant,
    mean_sem, log
)

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x):
    # deterministic bit-mixing step, turns a linear hash of a genotype into
    # something that looks like an independent random draw
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    z = z ^ (z >> np.uint64(31))
    return z


def make_hoc_landscape(p, sigma, rng):
    # every genotype gets independent random fitness
    mult = rng.integers(1, 2**63, size=p, dtype=np.uint64)
    mult = mult | np.uint64(1)  
    salt = np.uint64(rng.integers(0, 2**63, dtype=np.uint64))
    return {"p": p, "sigma": sigma, "mult": mult, "salt": salt}


def hoc_fitness(genos, lc, average=True):
    p, sigma = lc["p"], lc["sigma"]
    mult, salt = lc["mult"], lc["salt"]
    G = np.asarray(genos)
    if G.ndim == 1:
        G = G[None, :]
    if G.shape[1] != p:
        raise ValueError(f"expected genotype length p={p}, got {G.shape[1]}")
    B = (G == 1).astype(np.uint64) if np.min(G) < 0 else G.astype(np.uint64)

    with np.errstate(over="ignore"):
        h = (B * mult[None, :]).sum(axis=1, dtype=np.uint64) + salt
        z1 = _splitmix64(h)
        z2 = _splitmix64(z1)

    u1 = (z1 >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u2 = (z2 >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u1 = np.clip(u1, 1e-300, 1.0)
    normal = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    return normal * sigma


def hoc_adaptive_walk_single(founder, lc, tot_muts, max_attempts, rng):
    Y0 = hoc_fitness(founder[None, :], lc)
    curr = deepcopy(founder)
    for _a in range(max_attempts):
        fixloci = []
        Ycurr = float(Y0[0])
        Ytraj = [Ycurr]
        for _s in range(tot_muts):
            nbrs = one_step_neighbors(curr)
            Ynbr = hoc_fitness(nbrs, lc)
            s_all = Ynbr - Ycurr
            ben = np.where(s_all > 0)[0]
            if ben.size == 0:
                break
            locus = rng.choice(ben)
            curr = nbrs[locus].copy()
            Ycurr = float(Ynbr[locus])
            fixloci.append(int(locus))
            Ytraj.append(Ycurr)
        if len(Ytraj) == tot_muts + 1:
            return np.array(Ytraj), np.array(fixloci, dtype=int), True
    return None, None, False


def hoc_top_muts(curr, lc, tot_muts):
    nbrs = one_step_neighbors(curr)
    Ynbr = hoc_fitness(nbrs, lc)
    return np.argsort(Ynbr)[-tot_muts:]


def hoc_bottom_muts(curr, lc, tot_muts):
    nbrs = one_step_neighbors(curr)
    Ynbr = hoc_fitness(nbrs, lc)
    return np.argsort(Ynbr)[:tot_muts]


def hoc_endpoint_fitness(founder, mut_sites, lc):
    g = np.array(founder, copy=True)
    g[list(mut_sites)] *= -1
    return hoc_fitness(g[None, :], lc)[0]


def hoc_sample_best_conditioned_mutation_set(founder, traj_sites, lc, sites, n_samples, rng):
    founder = np.asarray(founder)
    traj_sites = tuple(traj_sites)
    k = len(traj_sites)
    tgt_fit = hoc_endpoint_fitness(founder, traj_sites, lc)

    cand = np.array([rng.choice(sites, size=k, replace=False) for _ in range(n_samples)])
    G = np.tile(founder, (n_samples, 1))
    rows = np.arange(n_samples)[:, None]
    G[rows, cand] *= -1

    diffs = np.abs(hoc_fitness(G, lc) - tgt_fit)
    best = np.argmin(diffs)
    return tuple(sorted(cand[best]))


def run_hoc(reps, p, tot_muts, max_attempts, sigma, n_cond_samples, methods, seed):
    t0_all = time.time()
    log(f"--- HoC model, replicates={reps} ---")

    rep_seeds = np.random.SeedSequence(seed).spawn(reps + 1)
    founder_rng = np.random.default_rng(rep_seeds[0])
    rngs = [np.random.default_rng(s) for s in rep_seeds[1:]]

    bgs_rep = founder_rng.choice([-1, 1], size=(reps, p))

    r1_vals = {m: [] for m in methods}
    r2_vals = {m: [] for m in methods}
    lm_vals = {m: [] for m in methods}
    path_vals = {m: [] for m in methods}
    n_noreach = 0

    t0 = time.time()
    prog_every = max(1, reps // 10)

    for r in range(reps):
        lc = make_hoc_landscape(p, sigma, rng=rngs[r])
        founder = bgs_rep[r]

        Ytraj, fixloci, ok = hoc_adaptive_walk_single(founder, lc, tot_muts, max_attempts, rngs[r])
        if not ok:
            n_noreach += 1
            continue

        cond_sites = hoc_sample_best_conditioned_mutation_set(
            founder, tuple(fixloci), lc, np.arange(p), n_cond_samples, rngs[r])

        for method in methods:
            if method == "traj":
                mut_sites = fixloci
            elif method == "rand":
                mut_sites = rngs[r].choice(p, size=tot_muts, replace=False)
            elif method == "topm":
                mut_sites = hoc_top_muts(founder, lc, tot_muts)
            elif method == "botm":
                mut_sites = hoc_bottom_muts(founder, lc, tot_muts)
            elif method == "cond":
                mut_sites = cond_sites

            lib = combinatorial_library_flip(founder, mut_sites)
            Ylib = hoc_fitness(lib, lc)

            beta1, terms1 = fit_epistasis_coeffs(lib, Ylib, d=1, ridge_lambda=0)
            r1_vals[method].append(r2_score(Ylib, predict_from_betas(lib, beta1, terms1)))

            beta2, terms2 = fit_epistasis_coeffs(lib, Ylib, d=2, ridge_lambda=0)
            r2_vals[method].append(r2_score(Ylib, predict_from_betas(lib, beta2, terms2)))

            lm_vals[method].append(count_total_local_maxima(lib, Ylib))
            path_vals[method].append(count_accessible_paths_to_full_mutant(lib, Ylib, founder, strict=True))

        del lc

        if (r + 1) % prog_every == 0 or (r + 1) == reps:
            log(f"HoC: {r+1}/{reps} replicates processed "
                f"({time.time()-t0:.1f}s elapsed, {n_noreach} failed so far)")

    out = {"R1_2": {}, "R2_2": {}, "localmax": {}, "paths": {}}
    for method in methods:
        m1, s1 = mean_sem(r1_vals[method])
        m2, s2 = mean_sem(r2_vals[method])
        m3, s3 = mean_sem(lm_vals[method])
        m4, s4 = mean_sem(path_vals[method])
        out["R1_2"][method] = (m1, s1)
        out["R2_2"][method] = (m2, s2)
        out["localmax"][method] = (m3, s3)
        out["paths"][method] = (m4, s4)
        log(f"HoC: method='{method}' summary: R1^2={m1:.3f}, R2^2={m2:.3f}, localmax={m3:.2f}, paths={m4:.2f}")

    return out, n_noreach


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    tot_muts = 4
    max_attempts = 1000
    sigma = 1.0
    n_cond_samples = 5000
    methods = ["traj", "rand", "topm", "botm", "cond"]
    seed = 999999999  # independent of the K-sweep's master_seed

    out, n_noreach = run_hoc(reps, p, tot_muts, max_attempts, sigma, n_cond_samples, methods, seed)

    flat = {"n_noreach": n_noreach}
    for metric in out:
        for method in methods:
            m, s = out[metric][method]
            flat[f"{metric}_{method}_mean"] = m
            flat[f"{metric}_{method}_sem"] = s

    np.savez(f"partial_HoC_L{p}_final.npz", **flat)
    log(f"Saved partial_HoC_L{p}_final.npz")
