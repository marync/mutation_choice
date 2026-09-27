import os
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import logomaker

def build_logo(counts, int_to_letter, allele_axis="columns", ax=None,
                xaxis_labels=None, save=False,
               logo_type="bits", outfile="sequence_logo.png", title=None):
    counts = np.asarray(counts, dtype=float)
    if allele_axis == "rows":            # make it (n_sites, n_alleles)
        counts = counts.T

    n_sites, n_alleles = counts.shape
    # Order letters by sorted integer key so column j corresponds to that code.
    letters = [int_to_letter[k] for k in sorted(int_to_letter)]
    if len(letters) != n_alleles:
        raise ValueError(
            f"dict has {len(letters)} alleles but matrix has {n_alleles}"
        )

    df = pd.DataFrame(counts, columns=letters)
    df.index = range(1, n_sites + 1)     # site labels 1..N

    # counts -> probabilities (pseudocount=0 keeps observed frequencies exact)
    prob = logomaker.transform_matrix(
        df, from_type="counts", to_type="probability", pseudocount=0
    )

    if logo_type == "bits":
        mat = logomaker.transform_matrix(
            prob, from_type="probability", to_type="information"
        )
        ylabel, ymax = "Information (bits)", np.log2(n_alleles)
    else:
        mat, ylabel, ymax = prob, "Probability", 1.0

    if ax is None :
        fig, ax = plt.subplots(figsize=(2.5, 2))
    logomaker.Logo(mat, ax=ax, shade_below=0.5, fade_below=0.5, color_scheme='weblogo_protein')
    #ax.set_xlabel("Locus, $m$", fontsize=8)
    #if ylabel :
    #    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(list(mat.index))
    if xaxis_labels is not None :
        ax.set_xticklabels (xaxis_labels, rotation=45)

    ax.set_ylim(0, ymax)
    ax.spines[["right", "top"]].set_visible(False)

    if title is not None :
        ax.set_title (title, fontsize=8, loc='left', pad=1, fontweight='bold')
    
    if save :
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
    return mat


def pool_matrix_sums_numpy (matrix, m):
    """
    From gemini.
    """
    
    matrix = np.array (matrix)
    rows, cols = matrix.shape
    
    # Calculate new dimensions (integer division)
    new_rows = rows // m
    new_cols = cols // m
    
    # Initialize the smaller matrix with zeros
    output_matrix = np.zeros ((new_rows, new_cols), dtype=int)
    
    # Map each m x m block to a single coordinate in the new matrix
    for i in range (new_rows):
        for j in range (new_cols):
            # Calculate the boundary indices in the original matrix
            r_start, r_end = i * m, (i + 1) * m
            c_start, c_end = j * m, (j + 1) * m
            
            # Sum the block and place it in the new smaller matrix
            output_matrix[i, j] = np.sum(matrix[r_start:r_end, c_start:c_end])
            
    return output_matrix


def pvalue_heatmap (mat, statistic, M, scenarios, out_label=None, save=True, outputdir='.') :
    # color map
    colors_pool = ["#008080", "#fcfcfc", "#ff7f0e"]
    my_cmap     = mcolors.LinearSegmentedColormap.from_list("teal_orange", colors_pool)

    n, n = mat.shape
    
    a           = np.nanmax (np.abs (mat[np.tril_indices (n)]))
    my_norm     = mcolors.Normalize (vmin=-a, vmax=a)
    
    bonf = .05 / (math.comb (n, 2)*(M+1))
    print (bonf)
    
    # create the masks
    upper_mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    lower_mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
    
    # 3. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # plot upper
    sns.heatmap(mat, mask=~upper_mask, cmap='Blues_r', fmt=".2f",
                cbar_kws={'label': '$p$-value', 'shrink': .7}, ax=ax,norm=mpl.colors.LogNorm ())
    
    # lower
    sns.heatmap(mat, mask=~lower_mask, cmap=my_cmap, annot=True, fmt=".2f", annot_kws={'size': 8},
                cbar_kws={'label': r'diff. in ' + statistic, 'location': 'top', 'shrink': .7}, ax=ax, norm=my_norm)
    
    for i in range (n) :
        for j in range (i+1, n) :
            if mat[i,j] < bonf :
                ax.scatter (j+.5,i+.5,marker='*', s=100, color='white', edgecolor='black', lw=.5)
    
    # labels
    plt.xticks ( np.arange (0, n, 1) + .5, [x[:4] for x in scenarios], rotation=75)
    plt.yticks ( np.arange (0, n, 1) + .5, [x[:4] for x in scenarios], rotation=360)
    
    if save :
        plt.savefig (os.path.join (outputdir, out_label + '_differences.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()


"""
Draw a heatmap where the upper triangle, lower triangle, and diagonal
each use a different color scheme.

The matrix is plotted three times on the same axes, each layer masked to
its own region:
  - upper triangle (excl. diagonal) -> colormap A
  - lower triangle (excl. diagonal) -> colormap B
  - diagonal                        -> colormap C
Each region gets its own colorbar so the scales stay independent.
"""

def split_triangle_heatmap(
    data,
    upper_cmap="Reds",
    lower_cmap="Blues",
    diagonal_cmap="Greys",
    labels=None,
    show_diagonal=True,
    annotate=True,
    title=None,
    outputdir=None,
):
    """
    Plot a square matrix with a different colormap on each triangle
    and on the diagonal.

    Parameters
    ----------
    data : 2D array-like (square)
        The matrix to display.
    upper_cmap, lower_cmap, diagonal_cmap : str
        Matplotlib colormap names for the three regions.
    labels : list of str, optional
        Tick labels for both axes.
    show_diagonal : bool
        If False, the diagonal cells are hidden (no diagonal layer/colorbar).
    annotate : bool
        Write each cell's value on top of it.
    title : str
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if data.shape[0] != data.shape[1]:
        raise ValueError("data must be a square matrix")

    # Boolean masks: True where a cell BELONGS to that region.
    upper = np.triu(np.ones((n, n), bool), k=1)   # above diagonal
    lower = np.tril(np.ones((n, n), bool), k=-1)  # below diagonal
    diag = np.eye(n, dtype=bool)                  # the diagonal itself

    # Masked arrays: mask=True hides the cell, so hide everything NOT in the region.
    upper_data = np.ma.masked_array(data, mask=~upper)
    lower_data = np.ma.masked_array(data, mask=~lower)
    diag_data = np.ma.masked_array(data, mask=~diag)

    fig, ax = plt.subplots(figsize=(3.2, 2.5))

    im_upper = ax.imshow(upper_data, cmap=upper_cmap)
    im_lower = ax.imshow(lower_data, cmap=lower_cmap)
    if show_diagonal:
        im_diag = ax.imshow(diag_data, cmap=diagonal_cmap)

    # One slim colorbar per region, stacked to the right.
    cb_u = fig.colorbar(im_upper, ax=ax, fraction=0.046, pad=0.25, shrink=.6)
    cb_u.set_label(f"distnace max.")
    cb_l = fig.colorbar(im_lower, ax=ax, fraction=0.046, pad=0.05, shrink=.6)
    cb_l.set_label(f"distance min.")
    if show_diagonal:
        cb_d = fig.colorbar(im_diag, ax=ax, fraction=0.046, pad=0.05, location='top', shrink=.7)
        cb_d.set_label(f"entropy (bits)")

    # Ticks / labels
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    # Cell annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                if i == j and not show_diagonal:
                    continue
                ax.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center", fontsize=6, color="black")

    if title is not None :
        ax.set_title (title, fontsize=8, loc='left', pad=1, fontweight='bold')
    plt.tight_layout()
    
    if outputdir is not None :
        plt.savefig(os.path.join (outputdir, "logo_heatmap.png"), dpi=300)
        plt.close ()
    else :
        plt.show()



    
