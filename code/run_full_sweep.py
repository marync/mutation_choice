import sys
sys.path.insert(0, '.')
import numpy as np
from run_single_K import run_one_K
from hoc_pipeline import run_hoc
from nk_pipeline import log

K_VALUES = [1, 3, 7, 11, 15, 25, 40, 50, 55, 63, 75, 85, 95]

TOT_MUTS = 4
WALK_NTRIES = 20
SIGMA = 1.0
COND_EPSILON = 1e-3 * SIGMA
COND_NBATCH = 10000
COND_NTRIES = 20
METHODS = ["traj", "rand", "topm", "botm", "cond"]
MASTER_SEED = 20
HOC_SEED = 99


def run_full_K_sweep(reps, p):
    k_seeds = np.random.SeedSequence(MASTER_SEED).spawn(len(K_VALUES))

    combined = {"K_values": np.array(K_VALUES)}
    metrics = ["R1_2", "R2_2", "localmax", "paths"]
    for metric in metrics:
        for method in METHODS:
            combined[f"{metric}_{method}_mean"] = []
            combined[f"{metric}_{method}_sem"] = []

    for K, k_seed in zip(K_VALUES, k_seeds):
        out, n_noreach = run_one_K(K, K_VALUES.index(K), len(K_VALUES), reps, p,
                                    TOT_MUTS, WALK_NTRIES, SIGMA, METHODS, k_seed,
                                    cond_epsilon=COND_EPSILON, cond_nbatch=COND_NBATCH,
                                    cond_ntries=COND_NTRIES)
        for metric in metrics:
            for method in METHODS:
                m, s = out[metric][method]
                combined[f"{metric}_{method}_mean"].append(m)
                combined[f"{metric}_{method}_sem"].append(s)

    for k in combined:
        if k != "K_values":
            combined[k] = np.array(combined[k])

    np.savez(f"LK_results_L{p}_final.npz", **combined)
    log(f"Saved LK_results_L{p}_final.npz")


def run_full_hoc(reps, p):
    out, n_noreach = run_hoc(reps, p, TOT_MUTS, 1000, SIGMA, 5000, METHODS, HOC_SEED)
    flat = {"n_noreach": n_noreach}
    for metric in out:
        for method in METHODS:
            m, s = out[metric][method]
            flat[f"{metric}_{method}_mean"] = m
            flat[f"{metric}_{method}_sem"] = s
    np.savez(f"partial_HoC_L{p}_final.npz", **flat)
    log(f"Saved partial_HoC_L{p}_final.npz")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    run_full_K_sweep(reps, p)
    run_full_hoc(reps, p)
    log("Full sweep complete.")
