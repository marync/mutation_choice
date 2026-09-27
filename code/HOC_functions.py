import os, sys
import copy as cp
import numpy as np
import collections

from combinatorial_utility import int_to_string, make_combinatorial_library, fit_regression_fourier, compute_Ak
from maxima_functions import *
from path_functions import call_paths

def mutate (xanc, ydict, sigma=1, rng=None) :
    """
    xanc: ancestral genotype
    ydict: dictionary with all sequence -> phenotypes sampled thus far
    sigma: standard deviation of bulk distribution
    """

    if rng is None :
        print ('No rng mutate.')
        rng = np.random.default_rng ()

    xanc  = np.array (xanc, dtype=int)
    L     = len (xanc)

    # sample new fitness values for mutants
    ymuts = rng.normal (0, sigma, size=L)

    # generate one-step mutations
    Mseqs = np.zeros ( (L, L), dtype=int ) * np.nan
    for i in range (L) :
        mut_i    = cp.deepcopy (xanc)
        mut_i[i] = int (( 0**xanc[i] ) * ( 1**(1 - xanc[i]))) # mutate at ith position

        # add fitness to dictionary
        str_i = int_to_string (mut_i)
        if str_i not in ydict.keys () :
            ydict[str_i] = ymuts[i]
        else :
            ymuts[i] = ydict[str_i]

        # copy mutant sequence
        Mseqs[i,:] = cp.deepcopy (mut_i)

    return ymuts, np.array (Mseqs, dtype=int)


def evolve_hoc (T, L, x0=None, ydict=None, sigma=1, weighted=False, rng=None, ntries=20) :

    if rng is None :
        print ('No rng evolve.')
        rng = np.random.default_rng ()

    if x0 is None :
        # sample the ancestor
        x0 = rng.binomial (n=1, p=.5, size=L)
 
    # dictionary to store sequences and fitness values
    if ydict is None :
        ydict = dict ()

    if int_to_string (x0) not in ydict.keys () :
        y0 = rng.normal (0, sigma)
        ydict[int_to_string (x0)] = y0

    # try uphill trajectory ntries times from the same ancestor
    steps, trajectory = try_n_times (ntries, go_uphill, T=T, L=L, x0=x0, ydict=ydict, sigma=sigma, weighted=weighted, rng=rng) 

    #print ('final size of landscape:')
    #print (len (ydict))

    return steps, trajectory


def go_uphill (T, L, x0, ydict, sigma, weighted, rng) :

    xcur = cp.deepcopy (x0)
    ycur = ydict[int_to_string (x0)]
    
    #print ('size of landscape:')
    #print (len (ydict))

    # mutations
    steps   = np.ones (T, dtype=int) * (-1)
    fitness = np.zeros (T)
    for t in range (T) :
        #print ('size of landscape:')
        #print ( (t, len (ydict) ))
        # mutate
        ymuts, M = mutate (xcur, ydict, rng=rng)

        if weighted :
            weights = ymuts - ycur
            weights[ymuts < ycur] = 0
            step_i = rng.choice ( np.arange (0, L, 1), p=(weights / np.sum (weights)))
        else :
            step_i = rng.choice (np.where (ymuts >= ycur)[0])
            if step_i in steps :
                raise ValueError("Recurrent mutation.")

        steps[t]   = step_i
        fitness[t] = ymuts[step_i]

        # update current state
        xcur = cp.deepcopy (M[step_i,:])
        ycur = fitness[t]

    fitness_all = np.insert (fitness, 0, ydict[int_to_string(x0)]) # add ancestral fitness
        
    return steps, fitness_all


def get_statistics_from_library ( library, fitness_values ) :

    nlib, M = library.shape
    
    new_idx = np.where ( np.sum (library, axis=1) == 0 )[0][0]
    
    r2_vals, intercept, coeffs  = fit_regression_fourier (library, fitness_values)
    Ak = compute_Ak (intercept, coeffs, M)

    nmax_vals     = count_local_maxima_step (library, fitness_values, new_idx)
    npath_vals    = call_paths (library, fitness_values)

    return r2_vals, Ak, nmax_vals, npath_vals


def find_beneficial_mutations ( ancestor, landscape, M, rng=None ) :
    y_step, Xtmp   = mutate ( ancestor, landscape, rng=rng )      # generate dfe
    bene_mutations = np.argsort (y_step)[-M:]

    return bene_mutations


def find_deleterious_mutations ( ancestor, landscape, M, rng=None ) :
    y_step, Xtmp   = mutate ( ancestor, landscape, rng=rng )      # generate dfe
    dele_mutations = np.argsort (y_step)[:M]

    #print (np.argsort (y_step)[:M])
    #print (y_step[dele_mutations])

    return dele_mutations


def make_conditional_library_hoc ( ancestor, yancestor, yderived, M, sigma=1, rng=None) :
    """
    """

    if rng is None :
        print ('No rng conditional.')
        rng = np.random.default_rng ()

    L = len (ancestor)

    # which mutations
    mutations = rng.choice (np.arange (0, L, 1), size=M, replace=False)

    # evolved sequence
    xevo = cp.deepcopy (ancestor)
    for i in mutations :
        xevo[i] = int (( 0**ancestor[i] ) * ( 1**(1 - ancestor[i])))

        # create new dictionary and add ancestral and evolved sequences
        mydic = dict ()      
        mydic[int_to_string (ancestor)] = yancestor
        mydic[int_to_string (xevo)]     = yderived
        
    return make_combinatorial_library (ancestor, mutations, mydic, sigma=sigma)

    

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


def simulate_from_common_ancestor_hoc ( M, L, sigma=1, ntries=20, weighted=False, rng=None) :

    if rng is None :
        print ('No rng.')
        rng = np.random.default_rng ()

    labels = ['adaptive', 'beneficial', 'deleterious', 'random', 'conditional']
    ntypes = len (labels) 

    # R2, no. of maxima, no. of accesible paths
    r2s    = np.zeros ( (ntypes, M+1) )
    Aks    = np.zeros_like (r2s)
    nmaxs  = np.zeros ( (ntypes, M+1) )   
    npaths = np.zeros ( ntypes )

    # dictionary to store all of the libraries
    scenario_dict = dict ()

    # ADAPTIVE
    ct = 0
    ancestor = rng.binomial (n=1, p=.5, size=L)
    y0 = rng.normal (0, sigma)
    
    # create landscape
    landscape = dict ()
    landscape[int_to_string (ancestor)] = y0
    
    # try adaptive trajectory
    #ancestor, landscape, mutations, ytraj = try_n_times (ntries, evolve_hoc, x0=x0, ydict=landscape, T=M, L=L, weighted=weighted, rng=rng)
    mutations, ytraj = evolve_hoc (T=M, L=L, x0=ancestor, ydict=landscape, rng=rng) 
    
    # make combinatorial library
    Xsuba, Yliba = make_combinatorial_library (x0=ancestor, mutations=mutations, ydict=landscape, sigma=1)
    r2s[ct,:], Aks[ct,:], nmaxs[ct,:], npaths[ct] = get_statistics_from_library ( Xsuba, Yliba )
    scenario_dict['adaptive'] = (Xsuba, Yliba, ytraj)

    # BENEFICIAL
    ct = 1
    
    bene_mutations = find_beneficial_mutations ( ancestor, landscape, M=M, rng=rng )
    Xsubb, Ylibb   = make_combinatorial_library (x0=ancestor, mutations=bene_mutations, ydict=landscape, sigma=1) 
    r2s[ct,:], Aks[ct,:], nmaxs[ct,:], npaths[ct] = get_statistics_from_library ( Xsubb, Ylibb )
    scenario_dict['beneficial'] = (Xsubb, Ylibb)

    # Deleterious 
    ct = 2
    dele_mutations = find_deleterious_mutations ( ancestor, landscape, M=M, rng=rng )
    Xsubd, Ylibd   = make_combinatorial_library (x0=ancestor, mutations=dele_mutations, ydict=landscape, sigma=1)
    r2s[ct,:], Aks[ct,:], nmaxs[ct,:], npaths[ct] = get_statistics_from_library ( Xsubd, Ylibd )
    scenario_dict['deleterious'] = (Xsubd, Ylibd)
    if npaths[ct] > 0 :
        print (ytraj[0])
        print (Ylibd)


    # RANDOM
    ct = 3
    rand_mutations = rng.choice (np.arange (0, L, 1), size=M, replace=False)
    Xsubr, Ylibr   = make_combinatorial_library (x0=ancestor, mutations=rand_mutations, ydict=landscape, sigma=1)
    r2s[ct,:], Aks[ct,:], nmaxs[ct,:], npaths[ct] = get_statistics_from_library ( Xsubr, Ylibr )
    scenario_dict['random'] = (Xsubr, Ylibr)

    # CONDITIONAL
    ct = 4
    Xsubc, Ylibc = make_conditional_library_hoc (ancestor=ancestor, yancestor=ytraj[0], yderived=ytraj[-1], M=M, rng=rng)
    r2s[ct,:], Aks[ct,:], nmaxs[ct,:], npaths[ct] = get_statistics_from_library ( Xsubc, Ylibc )
    scenario_dict['conditional'] = (Xsubc, Ylibc)

    return r2s, Aks, nmaxs, npaths, scenario_dict


def simulate_hoc_wrapper ( M, L, nsims, weighted=False, rng=None, ntries=20) :

    if rng is None :
        print ('No rng sim.')
        rng = np.random.default_rng ()

    labels = ['adaptive','beneficial','deleterious','random','conditional']
    ntypes = len (labels)   
 
    R2    = np.zeros ((ntypes, nsims, M+1))
    Ak    = np.zeros_like (R2)
    Nmax  = np.zeros_like (R2, dtype=int)
    Npath = np.zeros ((ntypes, nsims), dtype=int)
    
    Sim_dict = dict ()
    for el in labels :
        Sim_dict[el] = collections.OrderedDict ()
        
    ct = 0
    while ct < nsims :

        try :
            R2[:,ct,:], Ak[:,ct,:], Nmax[:,ct,:], Npath[:,ct], dict_i = simulate_from_common_ancestor_hoc ( M, L, ntries=ntries, weighted=weighted, rng=rng)
            for key in dict_i.keys () :
                Sim_dict[key][ct] = cp.deepcopy (dict_i[key])

            ct += 1

        except Exception as e:
            print('An error occurred at step ' + str (ct) + ':', e)
            ct += 0

    return R2, Ak, Nmax, Npath, Sim_dict, labels
