import numpy as np
import scipy.stats
import collections
import math
from itertools import chain, permutations

from combinatorial_utility import powerset

def subset_exists (main_list, subset):
    """
    Checks if a subset appears in the main_list.

    Returns a boolean.
    """
    
    return set(subset).issubset(set(main_list))


# from chatgpt
def enumerate_paths (start, end) :
    """
    Enumerate all mutation paths from start to end with no recurrent mutations.

    Parameters
    ----------
    start, end : str or list of int
        Binary sequences of equal length (e.g. "0101" or [0,1,0,1])

    Yields
    ------
    path : list of str
        A sequence of binary states from start to end
    """
    
    if len(start) != len(end):
        raise ValueError("Start and end must have the same length")

    # Convert to lists for mutability
    start = list ( map (int, start))
    end   = list( map (int, end))
    
    # Positions where bits differ
    diff_positions = [i for i in range(len(start)) if start[i] != end[i]]

    for order in permutations (diff_positions) :
        current = start.copy()
        path = [''.join(map(str, current))]

        for pos in order:
            current[pos] = end[pos]
            path.append(''.join(map(str, current)))

        yield path


def create_path_subset_dictionary (K) :
    """
    K: number of mutations in trajectory
    """
    
    # generate path from 00..0 -> 11..1
    adaptive_path = list ()
    for i in range (K+1) :
        seq = ['1'] * i + ['0'] * (K-i)
        adaptive_path.append (('').join (seq))
    
    # create dictionary to store counts of each subset path type
    count_dict = collections.OrderedDict ()
    combos = list (reversed (list (powerset (adaptive_path[1:-1]))))
    for el in combos :
        count_dict[el] = 0
    
    # count instances of subsets among path types
    for path in enumerate_paths ([0]*K, [1]*K) :
        for el in combos :          
            ind = subset_exists (path, el)
            if ind :
                count_dict[el] += 1                
                break
                
    return count_dict, adaptive_path

def compute_path_subset_probabilities (K, y_trajectory) :
    """
    """

    # counts number of times each subset appears in a path 
    count_dict, adaptive_path = create_path_subset_dictionary (K)
    
    # to store probabilities
    probabilities = np.ones (len (count_dict.keys ()))
    
    # for each unique element of the dictionary compute the probability that it is an accessible path
    ct = 0
    for subseq in count_dict.keys () :
        #print (subseq)
        idxs = [0]
        for s in subseq :
            idxs.append ( [int (x) for x in np.arange (K+1) if adaptive_path[x] == s][0])
        idxs.append (K)

        # number of genos separating subsequence
        idiff = np.diff (idxs)
    
        # compute probability
        for i in range (len (idiff)) :
            if idiff[i] > 1 :
                area = scipy.stats.norm.cdf (y_trajectory[idxs[i+1]]) - scipy.stats.norm.cdf (y_trajectory[idxs[i]])
                probabilities[ct] *= ( area )**(idiff[i] - 1) * (1 / math.factorial (idiff[i]-1))
    
        ct += 1

    return count_dict, probabilities

def compute_expected_paths (trajectory) :

    M = len (trajectory) - 1
    count_dict, probs = compute_path_subset_probabilities (M, trajectory)
 
    exp_paths = 0
    ct = 0
    for key in count_dict.keys () :
        exp_paths += count_dict[key] * probs[ct]
        ct += 1

    return exp_paths
