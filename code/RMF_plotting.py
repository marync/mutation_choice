import os
import copy as cp
import math
import numpy as np
import matplotlib.pyplot as plt


from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap


def kde_on_grid(x, y, X, Y):
    """Evaluate a KDE on a shared, pre-built grid."""
    kde = gaussian_kde(np.vstack([x, y]))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return Z

def nice_step(rough, allowed=(1, 2, 2.5, 5, 10)):
    """Round a rough step up to the nearest 'nice' number."""
    mag = 10 ** np.floor(np.log10(rough))
    for a in allowed:
        if a * mag >= rough:
            return a * mag
    return 10 * mag



def plot_statistics (R2, nMax, nPaths, A, sigma_a=None, sigma_e=None, hoc=False, outputdir=None, outname=None, 
                     labels = ['adaptive','conditional','random','beneficial'],
                     colors = ['salmon', 'dimgray', 'lightgray', 'gold']) :
    
    """
    """

    plt.rcParams['figure.figsize'] = (9,2)

    ntypes, N, Mt = R2.shape
    M = Mt - 1
    
    fig, axs = plt.subplots (1,4,constrained_layout=True)

    # amplitude spectrum hoc
    binoms = np.array ([math.comb (M, i) for i in range (1,M+1)])
    Am_hoc = cp.deepcopy (binoms) #/ (2**M - 1)
    R2_hoc = np.cumsum (np.append ([0], Am_hoc / (2**M - 1)))
    nM_hoc = np.append ([1], binoms) / (M+1)
    nP_hoc = 1

    # expectations for additive case
    R2add    = np.ones (M+1)
    R2add[0] = 0
    Amadd    = np.zeros (M+1)
    Amadd[1] = 1
    nM_add   = np.append ([1], Am_hoc) * 2**(-M)
    nP_add   = math.factorial (M) * 2**(-M)
    
    xs = np.arange (0, M+1, 1)
    ax = axs[0]
    for i in range (ntypes) :
        ax.errorbar (xs, np.mean (R2[i,:,:], axis=0), 1.96*np.sqrt (np.var (R2[i,:,:], axis=0) / N),
                     capsize=7,  marker='o', color=colors[i])

    if hoc :
        ax.plot (xs, R2_hoc,
                 color='black', linestyle='--', linewidth=1, zorder=10, label='HoC')

    if sigma_a is not None and sigma_e is not None :
        binom_coefficients = np.array ([math.comb (M, k) for k in range (1,M+1)])

        Am = 2**(-M) * sigma_e**2 * binom_coefficients
        Am[0] += (M * sigma_a**2)
        R2_rmf = np.append ([0], (2**(-M) * sigma_e**2 * np.cumsum (binom_coefficients)) + (M * sigma_a**2))
        denom  = M * sigma_a**2 + 2**(-M) * sigma_e**2 * (2**M - 1)
        #R2_rmf[1] -= ((M*sigma_a**2 + M * 2**(-M) * sigma_e**2) / denom)
        ax.plot (xs, R2_rmf / denom,
                 color='black', linestyle='dotted', linewidth=1, zorder=12, label='RMF')

        
   
    ax.legend (frameon=False) 
    ax.set_xticks (xs, xs)
    ax.set_xlabel (r'model order, $m$')
    ax.set_ylabel (r'$\hat R_{m}^2$')
  
    xs = np.arange (0, M+1, 1)
    ax = axs[1]
    for i in range (ntypes) :
        ax.errorbar (xs[1:], np.mean (A[i,:,:], axis=0)[1:],
                     1.96*np.sqrt (np.var ( A[i,:,:], axis=0)[1:] / N),
                     capsize=7, marker='o', color=colors[i], label=labels[i])

    if hoc :
        ax.plot (xs[1:], Am_hoc * np.sum (Am) / (2**M - 1),
                 color='black', linestyle='--', linewidth=1, zorder=10)
    

    if sigma_a is not None and sigma_e is not None :
        ax.plot (xs[1:], Am,
                 color='black', linestyle='dotted', linewidth=1, zorder=10)
        
    
    ax.set_xticks (xs[1:], xs[1:])
    ax.set_xlabel (r'model order, $m$')
    ax.set_ylabel (r'$\hat A_{m}^2$')
   
    
    ax = axs[2]
    for i in range (ntypes) :
        ax.scatter (xs, np.mean (nMax[i,:,:], axis=0), 
                    marker='o', color=colors[i], label=labels[i])
        ax.errorbar (xs, np.mean (nMax[i,:,:], axis=0), 1.96*np.sqrt (np.var (nMax[i,:,:], axis=0) / N),
                     capsize=7, marker=None, color=colors[i])
    ax.set_xticks (xs, xs)
    ax.set_xlabel (r'no. mutations')
    ax.set_ylabel (r'# local maxima')   

    if hoc :
        ax.plot (xs, nM_hoc, color='black', linestyle='--', linewidth=1, zorder=10)
        ax.plot (xs, nM_add, color='black', linestyle='-.', linewidth=1, zorder=10, label='add')
    
    ax = axs[3]
    for i in range (ntypes) :
        ax.scatter (i, np.mean (nPaths[i,:]), marker='o', color=colors[i], label=labels[i][:4])
        ax.errorbar (i, np.mean (nPaths[i,:]), np.sqrt (np.var (nPaths[i,:])/N),
                     capsize=7, marker=None, color=colors[i])

    
    if hoc :
        ax.axhline (1, color='black', linestyle='--', linewidth=1, zorder=0)
        ax.axhline (nP_add, color='black', linestyle='-.', linewidth=1, zorder=10, label='add')
    
    ax.set_xticks (np.arange (0,ntypes,1), [x[:4] for x in labels], rotation=60)
    ax.set_ylim (ymin=-.2)
    ax.set_ylabel ('# access. paths')
    plt.legend (frameon=False, fontsize=8, bbox_to_anchor=(1.05,1))
    
    if outputdir is not None :
        plt.savefig (os.path.join (outputdir, 'rmf_statistics_' + str (outname) + '.pdf'), bbox_inches='tight')
        plt.close ()
    else :
        plt.show ()
