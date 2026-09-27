import numpy as np

def compute_distance_matrix ( Xmat ) :

    n, L = Xmat.shape

    D = np.zeros ((n,n), dtype=int)
    for i in range (n) :
        for j in range (i, n) :
            D[i,j] = D[j,i] = np.sum ( np.abs (Xmat[i,:] - Xmat[j,:]) )


    return D


def count_local_maxima ( Xmat, Ys ) :

    n, L = Xmat.shape
    D = compute_distance_matrix ( Xmat )
    
    nmax = 0
    for i in range (n) :
        neighbors_i = np.where ( D[i,:] == 1 )[0]
        
        if np.sum (Ys[neighbors_i] < Ys[i]) == len (neighbors_i) :
            nmax += 1

    return nmax


def count_local_maxima_step ( Xmat, Ys, anc_idx=0 ) :
    """
    First row of Xmat is the ancestor.

    How many variants at a given mutational distance from the ancestor are local maxima?
    """

    n, L = Xmat.shape
    D    = compute_distance_matrix ( Xmat )

    nmax = np.zeros ( L+1, dtype=int )
    for i in range (n) :
        neighbors_i = np.where ( D[i,:] == 1 )[0]
        
        if np.sum (Ys[neighbors_i] <= Ys[i]) == len (neighbors_i) :
            anc_dist = int (D[anc_idx,i])
            nmax[anc_dist] += 1

    return nmax
