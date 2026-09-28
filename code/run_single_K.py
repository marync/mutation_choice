import sys
import time
import numpy as np
from sklearn.metrics import r2_score
sys.path.insert(0, '.')
from nk_pipeline import (
    make_nk_landscape, nk_adaptive_walk_with_landscape_retry, sample_conditioned_mutation_set_threshold,
    top_muts, bottom_muts, combinatorial_library_flip, nk_fitness,
    fit_epistasis_coeffs, predict_from_betas, count_total_local_maxima,
    count_accessible_paths_to_full_mutant, mean_sem, log
)

def run_one_K(K, k_idx, n_K, reps, p, tot_muts, walk_ntries, sigma, methods, k_seed,
              cond_epsilon=0.01, cond_nbatch=10000, cond_ntries=20, force_mode=None):
    t_k0 = time.time()
    log(f"--- K={K} ({k_idx+1}/{n_K}) replicates={reps} mode={force_mode or 'auto'} ---")

    rep_seeds = k_seed.spawn(reps)
    rngs = [np.random.default_rng(s) for s in rep_seeds]

    r1_vals = {m: [] for m in methods}
    r2_vals = {m: [] for m in methods}
    lm_vals = {m: [] for m in methods}
    path_vals = {m: [] for m in methods}
    n_noreach = 0

    t0 = time.time()
    prog_every = max(1, reps // 10)

    # landscape+founder held fixed across up to walk_ntries attempts redrawn only if all fail
    for r in range(reps):
        lc, founder, Ytraj, fixloci, ok = nk_adaptive_walk_with_landscape_retry(
            p, K, sigma, tot_muts, walk_ntries, rngs[r], force_mode=force_mode)
        if not ok:
            n_noreach += 1
            continue

        cond_sites = sample_conditioned_mutation_set_threshold(
            founder, tuple(fixloci), lc, np.arange(p),
            epsilon=cond_epsilon, nbatch=cond_nbatch, ntries=cond_ntries, rng=rngs[r])
        if cond_sites is None:
            # no candidate found within tolerance, treat as a failed replicate
            n_noreach += 1
            continue

        for method in methods:
            if method == "traj":
                mut_sites = fixloci
            elif method == "rand":
                mut_sites = rngs[r].choice(p, size=tot_muts, replace=False)
            elif method == "topm":
                mut_sites = top_muts(founder, lc, tot_muts)
            elif method == "botm":
                mut_sites = bottom_muts(founder, lc, tot_muts)
            elif method == "cond":
                mut_sites = cond_sites

            lib = combinatorial_library_flip(founder, mut_sites)
            Ylib = nk_fitness(lib, lc)

            beta1, terms1 = fit_epistasis_coeffs(lib, Ylib, d=1, ridge_lambda=0)
            r1_vals[method].append(r2_score(Ylib, predict_from_betas(lib, beta1, terms1)))

            beta2, terms2 = fit_epistasis_coeffs(lib, Ylib, d=2, ridge_lambda=0)
            r2_vals[method].append(r2_score(Ylib, predict_from_betas(lib, beta2, terms2)))

            lm_vals[method].append(count_total_local_maxima(lib, Ylib))
            path_vals[method].append(count_accessible_paths_to_full_mutant(lib, Ylib, founder, strict=True))

        del lc

        if (r + 1) % prog_every == 0 or (r + 1) == reps:
            log(f"K={K}: {r+1}/{reps} replicates processed "
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
        log(f"K={K}: method='{method}' summary: R1^2={m1:.3f}, R2^2={m2:.3f}, localmax={m3:.2f}, paths={m4:.2f}")

    log(f"=== K={K} complete in {time.time()-t_k0:.2f}s ({n_noreach}/{reps} failed to reach walk length) ===")
    return out, n_noreach


if __name__ == "__main__":
    K = int(sys.argv[1])
    k_idx = int(sys.argv[2])
    n_K = int(sys.argv[3])
    reps = int(sys.argv[4])
    p = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    tot_muts = 4
    walk_ntries = 20
    sigma = 1.0
    cond_epsilon = 1e-3 * sigma 
    cond_nbatch = 10000
    cond_ntries = 20
    methods = ["traj", "rand", "topm", "botm", "cond"]

    master_seed = 20260731
    k_seeds = np.random.SeedSequence(master_seed).spawn(n_K)
    k_seed = k_seeds[k_idx]

    out, n_noreach = run_one_K(K, k_idx, n_K, reps, p, tot_muts, walk_ntries,
                                sigma, methods, k_seed,
                                cond_epsilon=cond_epsilon, cond_nbatch=cond_nbatch, cond_ntries=cond_ntries)

    flat = {"n_noreach": n_noreach}
    for metric in out:
        for method in methods:
            m, s = out[metric][method]
            flat[f"{metric}_{method}_mean"] = m
            flat[f"{metric}_{method}_sem"] = s

    np.savez(f"partial_K{K}_p{p}.npz", **flat)
    log(f"Saved partial_K{K}_p{p}.npz")
