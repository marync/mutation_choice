import numpy as np
import copy as cp

from numba import njit

from combinatorial_utility import fit_regression_fourier, recode_library
from maxima_functions import count_local_maxima_step
from path_functions import call_paths

def simulate_from_common_ancestor (ancestor, X, ys, ntries=10, epsilon=0.01, weighted=False, rng=None, seed=None) :


    if rng is None :
        if seed is None :
            seed = np.random.choice (int (10**7))
            print ('Setting seed: ' + str (seed))
        else :
            rng = np.random.default_rng (seed)

    ntypes = 5      # beneficial, adaptive, conditional, random
    n, M   = X.shape

    anc_idx = np.where (np.sum (X == ancestor, axis=1) == M)[0][0]
    #print (anc_idx)

    # R2, no. of maxima, no. of accesible paths
    r2s    = np.zeros ( (ntypes, M+1) )
    nmaxs  = np.zeros ( (ntypes, M+1) )   
    npaths = np.zeros ( ntypes )
    Coefs  = np.zeros ( (ntypes, 2**M) )

    # dictionary to store all of the libraries
    scenario_dict = dict ()
   
    # BENEFICIAL
    ct = 0
    Blib, ylib, ids = try_n_times (1, simulate_library, X=X, ys=ys, ancestor=ancestor, scenario='beneficial')
    r2s[ct,:], nmaxs[ct,:], npaths[ct], Coefs[ct,:] = get_statistics_from_library ( Blib, ylib )
    scenario_dict['beneficial'] = (Blib, ylib, ids)
    
    # DELETERIOUS
    ct = 1
    Blibd, ylibd, ids = try_n_times (1, simulate_library, X=X, ys=ys, ancestor=ancestor, scenario='deleterious')
    r2s[ct,:],  nmaxs[ct,:], npaths[ct], Coefs[ct,:] = get_statistics_from_library ( Blibd, ylibd )
    scenario_dict['deleterious'] = (Blibd, ylibd, ids)
    
    # ADAPTIVE
    ct = 2
    Bliba, yliba, ytraj, ids = try_n_times (ntries, adaptive_given_ancestor, X=X, ys=ys, ancestor=ancestor, weighted=weighted, rng=rng)
    r2s[ct,:], nmaxs[ct,:], npaths[ct], Coefs[ct,:] = get_statistics_from_library ( Bliba, yliba )
    scenario_dict['adaptive'] = (Bliba, yliba, ids, ytraj)
    
    # CONDITIONAL
    ct = 3
    dista       = np.sum (X != ancestor, axis=1) # compute distances from ancestor
    cond_temp   = np.where ( np.logical_and (np.logical_and (ys <= ytraj[-1] + epsilon, ys >= ytraj[-1] - epsilon), dista == M ))[0]
    cond_idxs   = cond_temp[cond_temp != anc_idx]

    
    Blibc, ylibc, ids = try_n_times (ntries, conditional_given_ancestor, X=X, ys=ys, ancestor=ancestor, indices=cond_idxs, rng=rng)
    r2s[ct,:], nmaxs[ct,:], npaths[ct], Coefs[ct,:] = get_statistics_from_library ( Blibc, ylibc )
    scenario_dict['conditional'] = (Blibc, ylibc, ids)
    
    # RANDOM
    ct = 4
    Blibr, ylibr, ids = try_n_times (ntries, simulate_library, X=X, ys=ys, ancestor=ancestor, scenario='random', rng=rng)
    r2s[ct,:], nmaxs[ct,:], npaths[ct], Coefs[ct,:] = get_statistics_from_library ( Blibr, ylibr )
    scenario_dict['random'] = (Blibr, ylibr, ids)

    return r2s, nmaxs, npaths, Coefs, scenario_dict


def simulation_ancestor_wrapper (genos, phenos, nsims=100, indices=None, ntries=10, epsilon=.01, rng=None, seed=None) :
    """
    """

    if rng is None :
        if seed is None :
            seed = np.random.choice (int (10**7))
            print ('Setting seed: ' + str (seed))
        else :
            rng = np.random.default_rng (seed)


    ntypes = 5
    n, M = genos.shape
    
    R2    = np.zeros ((ntypes, nsims, M+1))
    nMax  = np.zeros ((ntypes, nsims, M+1))
    nPaths = np.zeros ( (ntypes, nsims) )
    anclist = np.zeros ( nsims )
    Coefs  = np.zeros ((ntypes, nsims, 2**M))
    
    i = 0
    while i < nsims :
        if indices is None :            
            anc_idx = np.random.choice ( np.arange (0, n, 1) )
        else :
            anc_idx = np.random.choice ( indices )

        # store ancestral genotype
        anc = cp.deepcopy (genos[anc_idx,:])
    
        try :
            R2[:,i,:],  nMax[:,i,:], nPaths[:,i], Coefs[:,i,:] = simulate_from_common_ancestor (ancestor=anc, X=genos, ys=phenos,
                                                                                                ntries=ntries, epsilon=epsilon, rng=rng)

            anclist[i] = anc_idx
            
            # iterate successful simulation count
            i += 1
    
        except Exception as e:
            print('An error occurred at step ' + str (i) + ':', e)
            i += 0

    return R2, nMax, nPaths, anclist, Coefs



def empirical_pvalue (permutations, observed) :

    N = len (permutations)
    p_lower = (np.sum ( permutations <= observed ) + 1) / (N+1)
    p_upper = (np.sum ( permutations >= observed ) + 1) / (N+1)

    #print (p_lower)
    #print (p_upper)

    return 2 * np.min ([p_lower, p_upper])


def try_n_times (n, func, *args, **kwargs):
    """
    Try a function up to n times.
    """

    last_exception = None

    for attempt in range (n):
        try:
            return func(*args, **kwargs)
        
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed: {e}")

    raise last_exception

def adaptive_given_ancestor (X, ys, ancestor,  weighted=False, rng=None) :

    # find the adapative trajectory
    focal, xtraj, ytraj = find_adaptive_trajectory (X=X, ys=ys, ancestor=ancestor, full=True, weighted=weighted, rng=rng)
    Blib, ylib, ids     = simulate_library (X=X, ys=ys, ancestor=ancestor, scenario='given', mutations=focal)

    return Blib, ylib, ytraj, ids


def conditional_given_ancestor (X, ys, ancestor, indices, rng=None) :

    if rng is None :
        rng = np.random.default_rng ()

    derived_idx = rng.choice ( indices )

    return simulate_library (X=X, ys=ys, ancestor=ancestor, scenario='given', mutations=X[derived_idx,:])
    

def get_statistics_from_library ( library, fitness_values ) :

    nlib, M = library.shape
    
    new_idx = np.where ( np.sum (library, axis=1) == 0 )[0][0]
    
    r2_vals, intercept, coefs = fit_regression_fourier (library, fitness_values)
    nmax_vals  = count_local_maxima_step (library, fitness_values, new_idx)
    npath_vals = call_paths (library, fitness_values)

    return r2_vals, nmax_vals, npath_vals, coefs


def simulate_library (X, ys, ancestor, scenario, mutations=None, weighted=False, rng=None) :
    """
    Generates a combinatorially complete library from a given ancestor.
    """

    n, M = X.shape
    
    if scenario == 'random' :
        focal = choose_random_mutations (ancestor, rng=rng)
    elif scenario == 'beneficial' :
        focal = choose_beneficial_mutations (ancestor, X, ys)
        #focal = choose_quantile_mutations (ancestor, X, ys, quantile=.75)
    elif scenario == 'adaptive' :
        focal = find_adaptive_trajectory (ancestor, X, ys, rng=rng, weighted=weighted)
    elif scenario == 'deleterious' :
        focal = choose_functional_mutations (ancestor, X, ys, direction=scenario)
    elif scenario == 'given' :
        focal = cp.deepcopy (mutations)

    # find the right members of the library
    ids = list ()
    for j in range (n) :
        ct = 0
        for k in range (M) :
            ct += (X[j,k] in (int (ancestor[k]), focal[k]))
    
        if ct == M :
            ids.append (j)

    if len (ids) != 2**M :
        raise Exception('The library is too small: ' + str (len (ids)))
    
    else :
        # get library
        Mlib = cp.deepcopy (X[ids,:])
        ylib = ys[ids]
        Blib = recode_library (ancestor, Mlib)

    return Blib, ylib, ids




def estimate_R2 (library, ys) :
    """
    Estimates variance explained at each order.
    """

    n, M = library.shape
    
    return fit_regression_fourier (library, ys)





def choose_random_mutations (ancestor, base=1, nAA=20, rng=None) :
    """
    Choose a set of random mutations starting from an ancestor.
    Assumes a library of mutations from 1,...,20.

    Returns an integer array of the chosen mutations.
    """
    M = len (ancestor)

    if rng is None :
        rng = np.random.default_rng ()
    
    focal = np.zeros (M, dtype=int)
    for m in range (M) :
        values    = np.arange (base, nAA+base, 1, dtype=int)
        newvalues = values[values != ancestor[m]]
        focal[m]  = rng.choice (newvalues)

    return focal


def find_adaptive_trajectory (ancestor, X, ys, full=False, weighted=False, rng=None) :
    """
    """

    if rng is None :
        rng = np.random.default_rng ()

    # number of mutations
    M = len (ancestor)

    # ancestor information
    anc_idx = np.where (np.sum (X == ancestor, axis=1) == M)[0]
    ycur    = ys[anc_idx][0] 
    xcur    = cp.deepcopy (ancestor)
    
    # evolve with no back mutations
    mutated = np.zeros (M, dtype=int)
    xtraj   = np.zeros ( (M+1, M), dtype=int )
    ytraj   = np.zeros (M+1)

    # store initial information
    xtraj[0,:] = cp.deepcopy (ancestor)
    ytraj[0]   = ycur
    
    for i in range (M) :
        # compute distances
        dcur = np.sum (X != xcur, axis=1) 
        nbs  = np.where (dcur == 1)[0]
    
        success = False
        tries   = 0
        while not success and tries < 10 :
            if weighted :
                # compute weights
                weights = (ys[nbs] - ycur)
                weights[weights <= 0] = 0
                weights /= np.sum (weights)
                
                # select mutation
                mut = rng.choice (nbs, p=weights)

            else :
                mut = rng.choice (nbs[ys[nbs] > ycur])

            # which site? make sure the site has not already mutated
            mutloc = np.where (X[mut,:] != xcur)[0]
            if mutated[mutloc] == 0 : 
                xcur = cp.deepcopy (X[mut,:])
                ycur = ys[mut]
                mutated[mutloc] = X[mut,mutloc]
                success = True
                xtraj[i+1,:] = cp.deepcopy (X[mut,:])
                ytraj[i+1]   = ys[mut]
            else :
                tries += 1

    if np.sum (mutated == 0) > 0 :
        raise Exception('Trajectory is too short: ' + str (np.sum (mutated == 0)))

    if full :
        return mutated, xtraj, ytraj
    else :
        return mutated


def choose_quantile_mutations ( ancestor, genos, ys, quantile, upper=True, rng=None ) :


    if rng is None :
        rng = np.random.default_rng ()

    M = len (ancestor)
   
    # find one-step neighbors
    dist      = np.sum (genos != ancestor, axis=1)
    neighbors = np.where (dist == 1)[0]
    
    # subset genos
    genos_nn = cp.deepcopy (genos[neighbors,:])
    ys_nn    = cp.deepcopy (ys[neighbors])

    # cut-off
    threshold = np.quantile (ys_nn, quantile)

    # sample loci from this quantile    
    sites = np.zeros (M, dtype=int)
    yvals = np.zeros (M)
    for i in range (M) :
        if upper :
            pool_i = np.where (np.logical_and (genos_nn[:,i] != ancestor[i], ys_nn >= threshold))[0]
        else :        
            pool_i = np.where (np.logical_and (genos_nn[:,i] != ancestor[i], ys_nn <= threshold))[0]

        # choose among pool
        keep = rng.choice (pool_i)

        sites[i] = genos_nn[keep,i]
        yvals[i] = ys_nn[keep] 
 
    return sites




def choose_beneficial_mutations ( ancestor, genos, ys ) :
    """
    Searches single site neighbors for the best mutation at each site.

    Do I want this to fail if the mutations are not all beneficial?
    """
 
    M = len (ancestor)
   
    # find one-step neighbors
    dist      = np.sum (genos != ancestor, axis=1)
    neighbors = np.where (dist == 1)[0]
    
    # subset genos
    genos_nn = cp.deepcopy (genos[neighbors,:])
    ys_nn    = cp.deepcopy (ys[neighbors])
    
    genos_nn_sorted = cp.deepcopy (genos_nn[np.flip (np.argsort (ys_nn)),:])
    
    sites = np.zeros (M, dtype=int)
    yvals = np.zeros (M)
    for i in range (M) :
        keep = np.min (np.where (genos_nn_sorted[:,i] != ancestor[i])[0])
        sites[i] = genos_nn_sorted[keep,i]
        yvals[i] = np.flip (np.argsort (ys_nn))[keep]
    
    return sites


def choose_functional_mutations ( ancestor, genos, ys, direction='beneficial' ) :
    """
    Need to search only single-site neighbors.
    """

    M = len (ancestor)   
 
    # find one-step neighbors
    dist      = np.sum (genos != ancestor, axis=1)
    neighbors = np.where (dist == 1)[0]
    
    genos_nn = cp.deepcopy (genos[neighbors,:])
    ys_nn    = cp.deepcopy (ys[neighbors])

    if direction == 'beneficial' :
        genos_nn_sorted = cp.deepcopy (genos_nn[np.flip (np.argsort (ys_nn)),:])
    else :
        genos_nn_sorted = cp.deepcopy (genos_nn[np.argsort (ys_nn),:])
    
    sites = np.zeros (M, dtype=int)
    yvals = np.zeros (M)
    for i in range (M) :
        keep = np.min (np.where (genos_nn_sorted[:,i] != ancestor[i])[0])

        sites[i] = genos_nn_sorted[keep,i]
        if direction == 'beneficial' :
            yvals[i] = np.flip (np.argsort (ys_nn))[keep]
        else :
            yvals[i] = np.argsort (ys_nn)[keep]
    
    return sites


def permute_r2 (X, ys, B=100, rng=None) :
    """
    """

    if rng is None :
        rng = np.random.default_rng ()

    # indivs by loci
    n, M = X.shape

    R2_boot = np.zeros ((B, M+1))
    for b in range (B) :
        order_b = np.arange (0,n,1)
        np.random.shuffle (order_b)
        R2_boot[b,:], intercept, coefs = fit_regression_fourier (X[order_b,:], ys)

    R2_obs, intercept, coefs = fit_regression_fourier (X, ys)

    return R2_boot, R2_obs




@njit(parallel=True)
def compute_distance (genos) :
    """
    Computes distance matrix from genotypes
    """

    N, M = genos.shape
    Dmat = np.zeros ( (N, N))
    for i in range (N) :
        Dmat[i,:] = np.sum (np.abs (genos[i,:] - genos), axis=1)

    return Dmat

@njit(parallel=True)
def compute_local_maxima (y, D) :
    """
    Asks whether a given node is a local maximum.
    """
    
    N = len (y)
    indmax = np.zeros (N)

    for i in range (N) :
        neib_i = np.where ( D[i,:] == 1 )[0]
        indmax[i] = np.all (y[i] > y[neib_i] )

    return indmax


@njit(parallel=True)
def compute_distance_multi (genos) :
    """
    Computes distance matrix from genotypes
    """

    N, M = genos.shape
    Dmat = np.zeros ( (N, N) )
    for i in range (N) :
        Dmat[i,:] = np.sum (genos[i,:] != genos, axis=1)

    return Dmat


def aa_to_numeric ( sequence, aa_to_int=None ) :

    if aa_to_int is None :
        aa_to_int = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 
            'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, '*': 21
        }


    return [aa_to_int.get(aa) for aa in sequence]

def default_aa_dict ( alphabetical=False ) :

    if alphabetical :
        aa_to_int = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "*": 21
}
    else :
        aa_to_int = {
           'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 
            'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, '*': 21
            }


    return aa_to_int 



def simulation_scenario_wrapper (genos, phenos, scenario, nsims=100, indices=None) :
    """
    
    """

    n, M = genos.shape
    
    R2      = np.zeros ((nsims, M+1))
    nMax    = np.zeros ((nsims, M+1))
    nPaths  = np.zeros (nsims)
    anclist = np.zeros (nsims)
    
    i = 0
    while i < nsims :
        if indices is None :            
            anc_idx = np.random.choice ( np.arange (0, n, 1) )
        else :
            anc_idx = np.random.choice ( indices )

        # store ancestral genotype
        anc = cp.deepcopy (genos[anc_idx,:])
    
        try :
            # simulate library according to scenario
            Blib, ylib, ids = simulate_library (X=genos, ys=phenos, ancestor=anc, scenario=scenario)

            # index of ancestral sequence
            new_idx = np.where ( np.sum (Blib, axis=1) == 0 )[0][0]

            # count maxima
            nMax[i,:] = count_local_maxima_step (Blib, ylib, new_idx)
            # compute R2            
            R2[i,:]   = fit_regression_fourier (Blib, ylib)
            # number paths
            nPaths[i] = call_paths (Blib, ylib)

            anclist[i] = anc_idx
            # iterate successful simulation count
            i += 1
    
        except Exception as e:
            print('An error occurred at step ' + str (i) + ':', e)
            i += 0

    return R2, nMax, nPaths, anc_idx


def resample_poisson (N0, N1, nreps=100, eta=1, rng=None) :

    n = len (N0)
    
    if rng is None :
        rng = np.random.default_rng ()

    Yboot = np.zeros ( (nreps, n) )
    for i in range (nreps) :
        Yboot[i,:] = rng.poisson (N1+1) / rng.poisson (N0+1)

    return Yboot
    
