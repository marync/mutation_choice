import sys, json, itertools
sys.path.insert(0, '.')
import numpy as np
from nk_pipeline import (
    make_nk_landscape, nk_fitness, one_step_neighbors, top_muts, bottom_muts,
    nk_adaptive_walk_with_landscape_retry, sample_conditioned_mutation_set_threshold,
    _pattern_index, _hash_normal, _hash_bits_zobrist, mean_sem, log
)
from scipy.special import ndtri

p = 100
sigma = 1.0
tot_muts = 4
walk_ntries = 20
cond_epsilon = 1e-3
cond_nbatch = 10000
cond_ntries = 20

pairs = list(itertools.combinations(range(tot_muts), 2))  # all 6 pairs among the 4 loci


def host_contrib_vec(hosts, geno, lc):
    # per-host fitness contribution f_i for a given genotype (founder or mutant)
    if len(hosts) == 0:
        return np.array([])
    hosts = np.asarray(hosts)
    K = lc["K"]
    bits0 = ((geno[lc["neigh"][hosts]] + 1) // 2)
    if lc["mode"] == "dense":
        pat0 = _pattern_index(bits0)
        return lc["table"][hosts, pat0]
    else:
        if K <= 62:
            pat0 = _pattern_index(bits0)
            return _hash_normal(lc["salt"], hosts, pat0, lc["sigma"])
        else:
            h = _hash_bits_zobrist(bits0.astype(np.int64), hosts, lc["salt"])
            u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = ndtri(u)
            return z * lc["sigma"]


def run(K_vals, reps, seed, out_path):
    ss = np.random.SeedSequence(seed)
    k_seeds = ss.spawn(len(K_vals))
    tags = ['topm', 'bot4', 'rand', 'traj', 'cond']
    out = {'K': [], 'n_noreach': []}
    for tag in tags:
        for m in ['N_i', 'N_ij', 'f_i', 'f_ij', 'delta_f_i', 'delta_f_ij']:
            out[f'{m}_{tag}'] = []
            out[f'{m}_{tag}_sem'] = []

    for K, k_seed in zip(K_vals, k_seeds):
        rep_seeds = k_seed.spawn(reps)
        rngs = [np.random.default_rng(s) for s in rep_seeds]

        vals = {tag: {'N_i': [], 'N_ij': [], 'f_i': [], 'f_ij': [],
                      'delta_f_i': [], 'delta_f_ij': []} for tag in tags}
        n_noreach = 0

        for r in range(reps):
            rng = rngs[r]
            lc, founder, Ytraj, fixloci, ok = nk_adaptive_walk_with_landscape_retry(
                p, K, sigma, tot_muts, walk_ntries, rng)
            if not ok:
                n_noreach += 1
                continue

            cond_sites = sample_conditioned_mutation_set_threshold(
                founder, tuple(fixloci), lc, np.arange(p),
                epsilon=cond_epsilon, nbatch=cond_nbatch, ntries=cond_ntries, rng=rng)
            if cond_sites is None:
                n_noreach += 1
                continue

            nb = lc["neigh"]
            mem = np.zeros((p, p), dtype=np.int32)
            rows = np.repeat(np.arange(p), nb.shape[1])
            mem[rows, nb.ravel()] = 1
            M = mem.sum(axis=0)

            top4 = top_muts(founder, lc, tot_muts)
            bot4 = bottom_muts(founder, lc, tot_muts)
            rand4 = rng.choice(p, size=tot_muts, replace=False)
            traj = np.array(fixloci)
            cond = np.array(cond_sites)

            for idx, tag in [(top4, 'topm'), (bot4, 'bot4'), (rand4, 'rand'), (traj, 'traj'), (cond, 'cond')]:
                # N_i: single-locus connectivity, averaged over the 4 ascertained loci
                vals[tag]['N_i'].append(np.mean(M[idx]))

                # N_ij: pairwise shared connectivity, averaged over all 6 pairs
                pair_vals = [int(mem[:, idx[a]] @ mem[:, idx[b]]) for a, b in pairs]
                vals[tag]['N_ij'].append(np.mean(pair_vals))

                # any_hosts: touched by >=1 of the 4 loci; shared_hosts: touched by >=2
                hits = mem[:, idx].sum(axis=1)
                any_hosts = np.where(hits >= 1)[0]
                shared_hosts = np.where(hits >= 2)[0]

                mutant = founder.copy()
                mutant[idx] *= -1

                if len(any_hosts) > 0:
                    pre = host_contrib_vec(any_hosts, founder, lc)
                    vals[tag]['f_i'].append(np.mean(pre))
                    post = host_contrib_vec(any_hosts, mutant, lc)
                    vals[tag]['delta_f_i'].append(np.mean(post - pre))

                if len(shared_hosts) > 0:
                    pre = host_contrib_vec(shared_hosts, founder, lc)
                    vals[tag]['f_ij'].append(np.mean(pre))
                    post = host_contrib_vec(shared_hosts, mutant, lc)
                    vals[tag]['delta_f_ij'].append(np.mean(post - pre))

        out['K'].append(K)
        out['n_noreach'].append(n_noreach)
        for tag in tags:
            for m in ['N_i', 'N_ij', 'f_i', 'f_ij', 'delta_f_i', 'delta_f_ij']:
                mean, sem = mean_sem(vals[tag][m])
                out[f'{m}_{tag}'].append(mean)
                out[f'{m}_{tag}_sem'].append(sem)
        log(f"K={K:3d} n_noreach={n_noreach}/{reps} " +
            " ".join(f"N_ij_{tag}={out[f'N_ij_{tag}'][-1]:.2f}" for tag in tags))

    np.savez(out_path, **{k: np.array(v) for k, v in out.items()})
    log(f"saved {out_path}")


if __name__ == "__main__":
    K_batch = json.loads(sys.argv[1])
    reps = int(sys.argv[2])
    seed = int(sys.argv[3])
    out_path = sys.argv[4]
    run(K_batch, reps, seed, out_path)
