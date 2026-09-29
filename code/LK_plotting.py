import numpy as np
from math import comb

methods = ["traj", "rand", "topm", "botm", "cond"]

neigh_methods = [("topm", "M_top4", "Nij_topm", "baseline_topm"),
                  ("botm", "M_bot4", "Nij_bot4", "baseline_bot4"),
                  ("traj", "M_traj", "Nij_traj", "baseline_traj"),
                  ("cond", "M_cond", "Nij_cond", "baseline_cond")]

colordict = {
    "traj": "salmon",
    "cond": "dimgray",
    "rand": "lightgray",
    "topm": "gold",
    "botm": "cornflowerblue",
}
labels_full = {
    "traj": "adaptive",
    "cond": "conditional",
    "rand": "random",
    "topm": "beneficial",
    "botm": "deleterious",
}


LINE_LW = 1            # connecting line width
MARKER_SIZE = 10         # circle marker size
MARKER_EDGEWIDTH = .5   # circle marker black outline width
ERR_ELINEWIDTH = 1.6     # error bar line width
ERR_CAPSIZE = 0          # error bar cap size
ENDPOINT_PLUS_S = 20    # '+' (additive) marker area (matches old markersize=17)
ENDPOINT_X_S = 20       # 'x' (HoC) marker area (matches old markersize=13)
ENDPOINT_LINEWIDTH = 1   # '+' / 'x' marker stroke width

def A_m(m, M, L, K, sigma):
    total  = 0.0
    ubound = np.min ([M,K]) 
    for j in range(m, ubound + 1):
        total += (2.0 ** (-j)) * comb(M - m, j - m) * comb(L - M, K - j)
    total *= comb(M, m)
    denom = comb(L, K)
    return sigma ** 2 * total / denom if denom > 0 else 0.0


def theory_R1_and_R2minusR1(K, L, M, sigma):
    As = [A_m(m, M, L, K, sigma) for m in range(1, M + 1)]
    total = sum(As)
    if total <= 0:
        return np.nan, np.nan
    return As[0] / total, As[1] / total


def g(data, metric, method, stat):
    return data[f"{metric}_{method}_{stat}"]


def g_hoc(hoc, metric, method, stat):
    return float(hoc[f"{metric}_{method}_{stat}"])


def diff_mean_sem(data, metric_a, metric_b, method):
    mean_diff = g(data, metric_a, method, "mean") - g(data, metric_b, method, "mean")
    sem_diff = np.sqrt(g(data, metric_a, method, "sem") ** 2 + g(data, metric_b, method, "sem") ** 2)
    return mean_diff, sem_diff


def diff_mean_sem_hoc(hoc, metric_a, metric_b, method):
    return g_hoc(hoc, metric_a, method, "mean") - g_hoc(hoc, metric_b, method, "mean")

def add_endpoint_markers(ax, x, y, color):
    """+ at K=1 (additive), x at K=100 (HoC) -- uses the shared style."""
    plot_endpoint(ax, x[0], y[0], color, "+")
    plot_endpoint(ax, x[-1], y[-1], color, "x")


def plot_series(ax, x, y, yerr, color):
    ax.plot(x, y, color=color, lw=LINE_LW, zorder=2)
    ax.errorbar(x, y, yerr=yerr, color=color, marker="o", markersize=MARKER_SIZE,
                markeredgecolor="black", markeredgewidth=MARKER_EDGEWIDTH,
                lw=0, elinewidth=ERR_ELINEWIDTH, capsize=ERR_CAPSIZE, zorder=3)


def plot_endpoint(ax, x, y, color, marker):
    s = ENDPOINT_PLUS_S if marker == "+" else ENDPOINT_X_S
    ax.scatter([x], [y], marker=marker, s=s, color=color, zorder=5)
    

#M_diff, M_diff_sem = {}, {}
#Nij_diff, Nij_diff_sem = {}, {}
#for key, mkey, nkey, bkey in neigh_methods:
#    M_diff[key] = d2[mkey] - d2["M_rand4"]
#    M_diff_sem[key] = np.sqrt(d2[f"{mkey}_sem"] ** 2 + d2["M_rand4_sem"] ** 2)
#    Nij_diff[key] = d2[nkey] - d2["Nij_rand"]
#    Nij_diff_sem[key] = np.sqrt(d2[f"{nkey}_sem"] ** 2 + d2["Nij_rand_sem"] ** 2)
