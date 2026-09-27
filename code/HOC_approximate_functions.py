import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import math

from combinatorial_utility import int_to_string, make_combinatorial_library_sequence, fit_regression_fourier, compute_Ak, recode_library
from HOC_amplitude_functions import *


def compute_trajectory_indices (x0, X) :
   
    M = len (x0) 
    Xt         = recode_library ( x0, X )
    match      = np.zeros (M, dtype=int)
    focal_idxs = np.zeros (M, dtype=int)
    for i in range (M) :
        match[i] = 1
        focal_idxs[i] = np.where ( np.sum (Xt == match, axis=1) == M )[0][0]

    return focal_idxs
 

def simulate_approximate_HOC (M, mean, variance, nsims=100, scenarios=['adaptive', 'beneficial', 'deleterious','random', 'conditional'],
                                                            colors=['salmon', 'gold', 'cornflowerblue','lightgray', 'dimgray'],
                                                            rng=None, seed=None) :
     
    if rng is None :
        if seed is not None :
            rng = np.random.default_rng (seed)
        else :
            rng = np.random.default_rng ()

    ntypes = len (scenarios)

    # conduct a test where one directly samples from the normals and compares to theory
    R2 = np.zeros ( (ntypes, nsims, M+1) )
    Am = np.zeros ( (ntypes, nsims, M+1) )

    for i in range (nsims) :
        Ys = dict () 

        x0 = rng.binomial (1, .5, size=M)
        Xt = make_combinatorial_library_sequence (x0)
    
        if i == 0 and 'adaptive' in scenarios :
            focal_idxs = compute_trajectory_indices (x0, Xt)

        # random case
        Ys['random'] = rng.normal (0, 1, size=Xt.shape[0])
        outscenarios = ['random']
       
        if 'beneficial' in scenarios :
            ytb = cp.deepcopy (Ys['random'])
            ytb[1:(M+1)] = rng.normal (mean, np.sqrt (variance), size=M)
            Ys['beneficial'] = cp.deepcopy (ytb)
            outscenarios.append ('beneficial')

        if 'adaptive' in scenarios :
            yta = cp.deepcopy (Ys['random'])
            yta[focal_idxs] = rng.normal (mean, np.sqrt (variance), size=M)
            Ys['adaptive']  = cp.deepcopy (yta)    
            outscenarios.append ('adaptive')

        if 'deleterious' in scenarios :
            ytd = cp.deepcopy (Ys['random'])
            ytd[1:(M+1)] = rng.normal (-mean, np.sqrt (variance), size=M)
            Ys['deleterious'] = cp.deepcopy (ytd)
            outscenarios.append ('deleterious')
        
        if 'conditional' in scenarios :
            ytc = cp.deepcopy (Ys['random'])
            ytc[-1] = Ys['adaptive'][-1] 
            Ys['conditional'] = cp.deepcopy (ytc)
            outscenarios.append ('conditional')

        ct = 0
        for key in outscenarios :
            R2[ct,i,:], intercept, coeffs = fit_regression_fourier ( recode_library (x0, Xt), Ys[key])
            R2_test, intercept_test, coeffs_test = fit_regression_fourier ( Xt, Ys[key])
            #R2[ct,i,:], intercept, coeffs = fit_regression_fourier ( Xt, Ys[key])

            if ct == 0 and i == 0 :
                Coef = np.zeros ((len (scenarios), nsims, len (coeffs)))
                Ints = np.zeros ((len (scenarios), nsims))

            Coef[ct,i,:] = cp.deepcopy (coeffs)
            Ints[ct,i]   = intercept

            Am[ct,i,:] = compute_Ak (intercept, coeffs, M)
            #print (coeffs) 
            #print (coeffs_test)
            #print (Am[ct, i, :])
            #print (compute_Ak (intercept_test, coeffs_test, M))
            #print ()
            ct += 1
       
    return R2, Am, Coef, Ints, outscenarios


def compute_random_spectrum (M) :
    """
    Variance explained for the random mutation model
    """

    var_explained = np.zeros (M)
    for i in range (M) :
        var_explained[i] = math.comb ( M, i+1)

    # normalize by total expected variance
    var_explained /= (2**M-1)
    var_explained = np.concatenate (([0], var_explained))

    return var_explained


def plot_HOC_statistics (R2_approx, Am_approx, mu, variance, R2_sim=None, colors=None) :

    ntypes, nsims, Mone = R2_approx.shape
    if R2_sim is not None :
        ntypes, N, Mone = R2_sim.shape
        nplots = 3
    else :
        nplots = 2

    if colors is None :
        colors = ['black'] * ntypes

    M = Mone - 1
    xs = np.arange (0, Mone, 1)

    plt.rcParams['figure.figsize'] = (2.5*nplots, 2)

    fig, axs = plt.subplots (1, nplots, sharex=False, constrained_layout=True)

    ax = axs[0]
    for i in range (ntypes) :
        ax.scatter (xs, np.mean (R2_approx[i,:,:], axis=0), color=colors[i])
        ax.set_ylabel (r'$R_m^2$')

    ax.set_xticks (xs, xs)
    ax.set_xlabel (r'model order, $m$')
    
    # beneficial amplitude spectrum
    aspectrum = compute_amp_spectrum_beneficial (M, mu, np.sqrt (variance))
    ax.plot (xs, np.cumsum (aspectrum) / np.sum (aspectrum), color='gold', linestyle='--', linewidth=1)
    ax.plot (xs-.05, np.cumsum (aspectrum) / np.sum (aspectrum), color='cornflowerblue', linestyle='--', linewidth=1)

    # trajectory
    aspectrum_adaptive = compute_amp_spectrum_adaptive (M, mu, np.sqrt (variance))
    vary      = compute_variance (M, mu, np.sqrt (variance))
    ax.plot (xs, np.cumsum (aspectrum_adaptive) / np.sum (aspectrum_adaptive), linestyle='--', color='salmon', linewidth=1, zorder=10)

    # random amplitude spectrum
    random_null = compute_random_spectrum (M)
    ax.plot (xs, np.cumsum (random_null) / np.sum (random_null), color='gray', linestyle='--', linewidth=1)

    ax = axs[1]
    for i in range (ntypes) :
        ax.scatter (xs, np.mean (Am_approx[i,:,:], axis=0), color=colors[i])
        ax.plot (xs, np.mean (Am_approx[i,:,:], axis=0), color=colors[i])
        ax.set_ylabel (r'$A_m$')

    ax.set_xticks (xs, xs)
    ax.set_xlabel (r'model order, $m$')
    
    aspectrum = compute_amp_spectrum_beneficial (M, mu, np.sqrt (variance), mean=True)
    aspectrum_adaptive[0] = aspectrum[0]
    ax.plot (xs, aspectrum, color='gold', linestyle='--', linewidth=1)
    ax.plot (xs, aspectrum_adaptive, color='maroon', linestyle='--', linewidth=1)
    ax.plot (xs, random_null, color='gray', linestyle='--', linewidth=1)

    if R2_sim is not None :
        ax = axs[2]
        plot_hoc_simulations (R2_sim, ax=ax, colors=colors)

    
    plt.show ()


def plot_hoc_simulations (R2_sim, colors=None, ax=None) :

    if ax is None :
        fig, ax = plt.subplots ()

    ntypes, nsims, Mone = R2_sim.shape
    M = Mone - 1

    # Positions for boxplots
    space = .5
    positions = np.reshape (np.repeat (np.arange(Mone-2) * (space*(ntypes+.5)), ntypes), (ntypes, Mone-2), order='F')
    for i in range (1,ntypes) :
        positions[i,:] += (space*i)

    ct = 0
    bp_list = list ()
    for i in range (ntypes) :
        bp_list.append (ax.boxplot( R2_sim[i,:,1:-1],
                                    positions=positions[i,:],
                                    widths=space,
                                    patch_artist=True,
                                    boxprops=dict(facecolor=colors[i]),
                                    medianprops=dict(color='black') ))
        ct += 1

    # plot theory
    #plt.scatter (positions[2,:], np.cumsum (var_explained), marker='x', color='black', zorder=10)
    #plt.scatter (positions[3,:], np.cumsum (var_explained), marker='x', color='black', zorder=10)
    #plt.scatter (positions[1,:], np.cumsum (test_amp) / np.sum (test_amp), marker='x', color='black', zorder=10)
    #plt.scatter (positions[0,:], np.cumsum (adap_amp) / np.sum (adap_amp), marker='x', color='black', zorder=10)


    #adapt_maxs = np.nanmax (R2mat[0,:,:], axis=0)
    #bene_maxs  = np.nanmax (R2mat[2,:,:], axis=0)
    #cond_max   = np.nanmax (R2mat[3,:,:], axis=0)

    # Labels and formatting
    ax.set_xticks (positions[0,:]+1)
    ax.set_xticklabels([i for i in np.arange (1,M)])
    ax.set_xlabel (r'model order, $m$', fontsize=12)
    ax.set_ylabel (r'$R^2_m$', fontsize=12)

    leg_list = list ()
    for el in bp_list :
        leg_list.append (el['boxes'][0])

    # Legend
    #ax.legend( leg_list, np.array (scenarios)[myorder], loc='best', frameon=False)

