import math
import copy as cp
import numpy as np
from itertools import combinations, product

from combinatorial_utility import find_partition, find_partition_specific
from moment_approximations import order_statistic_moments

def compute_amp_spectrum_beneficial_better (K, L, mu=0, sigma=1, mean=False) :

    mugs, variances = order_statistic_moments (L, K)
    sigmags = np.sqrt (variances)

    constant = (2**K - K) * sigma**2 + np.sum (sigmags**2) + np.sum (mugs**2)
    total_variance = constant + np.sum (np.outer (mugs, mugs)[np.triu_indices (K, k=1)])    
    spectrum = np.zeros (K+1)

    for i in range (1,K+1) :
        combos_i = list(combinations([int (x) for x in np.arange (0,K)], i))
        
        for mycomb in combos_i :
            # within one partition
            pairs_one = list (combinations (mycomb, 2))
            alt_part  = [int (x) for x in np.arange (0,K,1) if x not in mycomb]
            pairs_alt = list (combinations (alt_part, 2))
            pairs_across = list (product (mycomb, alt_part)) 
            
            # within and across partition terms
            for sub in pairs_one :
                spectrum[i] += 2 * mugs[sub[0]] * mugs[sub[1]]
            for sub in pairs_alt :
                spectrum[i] += 2 * mugs[sub[0]] * mugs[sub[1]]
            for sub in pairs_across :
                spectrum[i] -= 2 * mugs[sub[0]] * mugs[sub[1]]
        
        # add constant factor for each term
        spectrum[i] += (math.comb (K, i) * constant)
    
    if mean :
        outerprod = np.outer (mugs, mugs)
        np.fill_diagonal (outerprod, 0)
        spectrum[0] = constant + np.sum (outerprod)

    return spectrum / (2**(2*K))


def compute_amp_spectrum_adaptive (K, mug, sigmag, mu=0, sigma=1) :
    """
    Compute the amplitude spectrum under assumption of adaptive ascertainment.
    """

    base =  K*(sigmag**2 + mug**2) + (2**K - K) * sigma**2 # every order as this term
    
    spectrum = np.zeros (K+1)
    # compute spectrum for each order
    for i in range (1,K+1) :
        nodds = find_partition (K=K, order=i)
        #print (nodds)
        for j in nodds :
            spectrum[i] += base
            spectrum[i] += 2*(mug**2)*(math.comb (j, 2) + math.comb (K-j, 2) - j*(K-j))
            #print ((math.comb (j, 2) + math.comb (K-j, 2) - j*(K-j)))   
 
    return spectrum 


def compute_amp_spectrum_adaptive_better (K, mugs, sigmags, C, mu=0, sigma=1, mean=False) :

    # constant
    base = (2**K - K) * sigma**2 + np.sum (sigmags**2) + np.sum (mugs**2)
    
    # compute spectrum for each order
    spectrum = np.zeros (K+1)
    for i in range (1,K+1) :
        nodds = find_partition_specific (K=K, order=i)
        ncomb, nmut = nodds.shape
        for j in range (ncomb) :
            indices_odd  = np.where (nodds[j,:] == 1)[0]
            indices_even = np.where (nodds[j,:] == 0)[0]

            pairs_odd    = list (combinations (indices_odd, 2))
            pairs_even   = list (combinations (indices_even, 2))
            pairs_across = list (product (indices_odd, indices_even)) 

            # within and across partition terms
            for sub in pairs_odd :
                spectrum[i] += 2 * (mugs[sub[0]] * mugs[sub[1]] + C[sub[0],sub[1]])
            for sub in pairs_even :
                spectrum[i] += 2 * (mugs[sub[0]] * mugs[sub[1]] + C[sub[0],sub[1]])
            for sub in pairs_across :
                spectrum[i] -= 2 * (mugs[sub[0]] * mugs[sub[1]] + C[sub[0], sub[1]])
        
        # add constant factor for each term
        spectrum[i] += (math.comb (K, i) * base)

    if mean :
        outerprod = np.outer (mugs, mugs)
        np.fill_diagonal (outerprod, 0)
        C0 = cp.deepcopy (C)
        np.fill_diagonal (C0, 0)
        spectrum[0] = base + np.sum (outerprod) + np.sum (C0)

    
    return spectrum / (2**(2*K))

def compute_variance (K, mug, sigmag, sigma=1) :
    """
    Computes the variance of a sample of size 2**K where K values are sampled from a
    distinct distribution with mean mug and variance sigmag^2. The mean and variance
    of the reference distribution are taken to be zero and sigma^2.

    Arguments:
    K: number of mutated positions (and number of atypical samples)
    mug: mean of alternate distribution
    sigmag: standard dev. of alternate distribution
    sigma: s.d. of ref. distribution

    Returns:
    Expected value of sample variance (scalar)
    """
    
    mu = 0 # mean of ref distribution must be zero, could write more general form

    # squared contribution to variance
    vary  = (1. - 2**(-K)) * (K*(sigmag**2 + mug**2) + (2**K - K)*sigma**2)
    # cross terms
    vary -= (2**(-K)) * K * (K-1) * (mug**2)

    return vary / 2**K


def compute_amp_spectrum_beneficial (K, mug, sigmag, mu=0, sigma=1, mean=False) :
    """
    """

    Ctheta = K*(sigmag**2 + mug**2) + (2**K - K) * sigma**2

    spectrum = np.ones (K+1) * Ctheta
    for i in range (1,K+1) :
        spectrum[i] += 2 * ( math.comb (i, 2) + math.comb (K-i, 2) - i * (K-i)) * mug**2
        spectrum[i] *= math.comb (K, i)

    if mean :
        spectrum[0] += 2 * (math.comb (K, 2) * mug**2)
    else :
        spectrum[0] = 0
    
    return spectrum / (2**(2*K))

