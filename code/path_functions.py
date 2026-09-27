import numpy as np
from itertools import permutations

from combinatorial_utility import int_to_string

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


def find_viable_paths ( paths, ydict ) :

    viable_paths = 0
    for path_i in paths :
        yvals = np.zeros ( len (path_i) ) * np.nan
        ct = 0
        for seq in path_i :
            yvals[ct] = ydict[int_to_string (seq)]
            ct += 1
        if np.all (np.diff (yvals)  > 0 ) :
            viable_paths += 1

    return viable_paths


def path_wrapper ( x0, xK, ydict ) :

    paths = enumerate_paths ( x0, xK )

    return find_viable_paths ( paths, ydict )


def check_path ( path, ydict ) :
    ct = 0
    viable = 0
    yvals = np.zeros ( len (path) ) * np.nan
    for seq in path :
        yvals[ct] = ydict[int_to_string (seq)]
        ct += 1
        if np.all (np.diff (yvals)  > 0 ) :
            viable = 1

    return viable

def create_library_dictionary (library, ys) :
    """
    """
    
    ydict = dict ()
    n, L = library.shape
    
    for i in range (n) :
        key_i = ('').join ([str (x) for x in library[i,:]])
        ydict[key_i] = ys[i]

    return ydict


def call_paths (Blib, ylib) :
    """
    """

    n, L = Blib.shape
    y_dictionary = create_library_dictionary ( Blib, ylib )

    return path_wrapper ( np.zeros (L, dtype=int), np.ones (L, dtype=int), y_dictionary )
