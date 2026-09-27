import copy as cp
import numpy as np
import math
from itertools import chain, combinations, product

# for linear regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, cross_val_predict

def int_to_string (array) :
    return ('').join ([str (x) for x in array])


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) 


def regression_cv ( X, y, fold=10, seed=None, order=None) :

    n, M = X.shape
    if order is None :
        order = int (M+1)

    if seed is None :
        seed = np.random.randint (1,int (10**6))

    # metrics
    kscores = np.zeros (M+1)
    kmse    = np.zeros_like (kscores)
    for i in range (order) :

        #model = make_pipeline (PolynomialFeatures(i, interaction_only=True), LinearRegression())
        model = make_pipeline (PolynomialFeatures(i, interaction_only=True), StandardScaler(), Lasso(alpha=.1))
        kf    = KFold (n_splits=fold, shuffle=True, random_state=seed)
        kscores[i] = np.mean (cross_val_score (model, X, y, cv=kf, scoring='r2'))
        kmse[i]    = np.mean (cross_val_score (model, X, y, cv=kf, scoring='neg_root_mean_squared_error'))

    return kscores, kmse


def regression_loo ( X, y, seed=None, order=None) :

    n, M = X.shape
    if order is None :
        order = int (M+1)

    if seed is None :
        seed = np.random.randint (1,int (10**6))

    # metrics
    kscores  = np.zeros (M+1)
    ypredmat = np.zeros ((n,M+1))
    for i in range (order) :

        model = make_pipeline (PolynomialFeatures(i, interaction_only=True), StandardScaler(), LinearRegression())
        ypred = cross_val_predict (model, X, y, cv=LeaveOneOut())
        kscores[i] =  r2_score (y, ypred)
        ypredmat[:,i] = cp.deepcopy (ypred)

    #return kscores, ypredmat
    return kscores, np.zeros_like (kscores)


def out_of_sample_r2 ( X, y, split=0.9, train_idxs=None, order=None) :
    
    n, M = X.shape

    if order is None :
        order = int (M+1)    
    
    if train_idxs is None :
        train_idxs = np.sort (np.random.choice ( np.arange (0,n,1), size=int (split*n), replace=False))
    
    test_idxs  = np.array ( [x for x in np.arange (0,n,1) if x not in train_idxs], dtype=int )
    #print (test_idxs)

    X_train = cp.deepcopy (X[train_idxs,:])
    y_train = cp.deepcopy (y[train_idxs])

    X_test = cp.deepcopy (X[test_idxs,:])
    y_test = cp.deepcopy (y[test_idxs])

    #X_train = cp.deepcopy (X)
    #y_train = cp.deepcopy (y)
    #X_test  = cp.deepcopy (X)
    #y_test  = cp.deepcopy (y)

    r2  = np.zeros (order)
    mse = np.zeros (order)
    for i in range (order) :
        # define model
        model = make_pipeline (PolynomialFeatures(i, interaction_only=True), LinearRegression())
        # fit
        model.fit (X_train, y_train)
        
        # compute R2
        Ypred = model.predict (X_test) 
        r2[i]  = r2_score (y_test, Ypred)
        mse[i] = np.mean ( (y_test - Ypred)**2 )
    
    return r2, mse 


def compute_Ak (intercept, coefs, M) :

    binoms = np.array ([math.comb (M, k) for k in range (M+1)], dtype=int)
    binom_sum = np.cumsum (binoms)
 
    Aks = np.zeros (M+1)
    for i in range (M+1) :
        if i == 0 :
            Aks[i] = intercept**2
        else :
            idx_1 = binom_sum[i-1]
            idx_2 = binom_sum[i]
            Aks[i] = np.sum ( coefs[idx_1:idx_2]**2 )

    return Aks


def fit_regression (X, Y, order=None) :

    n, M = X.shape

    if order is None :
        order = int (M+1)
        
    r2 = np.zeros (order)
    for i in range (order) :
        # define model
        model = make_pipeline (PolynomialFeatures(i, interaction_only=True), LinearRegression())

        # fit
        model.fit (X, Y)
        
        if i == (order - 1) :
            coeffs = model.named_steps['linearregression'].coef_
            intercept = model.named_steps['linearregression'].intercept_

        # compute R2
        Ypred = model.predict (X)
        r2[i] = r2_score (Y, Ypred)        
    
    return r2, intercept, coeffs


def find_sum_of_squares (nmuts, X, Y, order) :
        
    model = make_pipeline(PolynomialFeatures(order, interaction_only=True), LinearRegression())
        
    # fit
    model.fit (X, Y)
    
    # get coefficients
    linear_regression_model = model.named_steps['linearregression']
    coefficients = linear_regression_model.coef_
    
    # compute sum of squares
    ss = np.zeros (order)
    meanf = np.zeros (order)
    
    start = 0
    for i in range (order) :
        nterms = math.comb (nmuts, i)
        ss[i]    = np.sum ( coefficients[start:(start+nterms)]**2 )
        meanf[i] = np.mean ( coefficients[start:(start+nterms)] )
        #print (coefficients[start:(start+nterms)])
        start += nterms
        
    return ss, meanf


def fit_regression_fourier (X, Y, order=None) :

    n, M = X.shape

    return fit_regression (2*X-1, Y, order=None)


def compute_fourier (nmut, X, Y) :

    coef_list = list ()
    coef_list.append ([np.mean (Y)])

    fis = np.zeros (nmut)
    for i in range (nmut) :
        fis[i] = 2**(-ngen) * ( np.sum (Y[X[:,i] == 1]) - np.sum (Y[X[:,i] == 0]) )

    coef_list.append (fis)
    return coef_list
    

def recode_library (anc, genos) :

    new = genos - anc
    new[new != 0] = 1

    return new

def make_combinatorial_library (x0, mutations, ydict, sigma=1) :

    L = len (x0)
    
    mutant_combos = list (powerset (mutations))
    N = len (mutant_combos)
    
    X = np.reshape (np.repeat (x0, N), (len (mutant_combos), L), order='F')
    Y = np.zeros (N)
    
    ct = 0
    for comb in mutant_combos :
        if len (comb) > 0 :
            for m in comb :
                X[ct,m] = ( 0**x0[m] ) * ( 1**(1 - x0[m]))
            
            if int_to_string (X[ct,:]) in ydict.keys () :
                #print ('in')
                #print (int_to_string (X[ct,:]))
                Y[ct] = ydict[int_to_string (X[ct,:])]
            else :
                #print ('adding')
                #print (int_to_string (X[ct,:]))
                Y[ct] = np.random.normal (0, sigma)
                ydict[int_to_string (X[ct,:])] = Y[ct]

        else :
            Y[ct] = ydict[int_to_string (x0)]
        
        ct += 1

    Xsub = cp.deepcopy (X[:,mutations])

    return recode_library (x0[mutations], Xsub), Y




def make_combinatorial_library_sequence (x0) :

    L = len (x0)
    mutations = np.arange (0,L,1)
    
    mutant_combos = list (powerset ( mutations ))
    N = len (mutant_combos)
    
    X = np.reshape (np.repeat (x0, N), (len (mutant_combos), L), order='F')
    
    ct = 0
    for comb in mutant_combos :
        if len (comb) > 0 :
            for m in comb :
                X[ct,m] = ( 0**x0[m] ) * ( 1**(1 - x0[m]))          
        ct += 1

    return X

def get_subsets_of_k (N, k):
    """
    Generates all unique subsets of size k from a given set of N numbers.
    
    Args:
    n_set: number of elements in set.
    k: The size of the subsets to return.
    
    Returns:
    A list of of lists, where each list is a unique subset of size k.
    """

    n_set = np.arange (0,N,1)
    
    # itertools.combinations returns an iterator of tuples
    combinations_iterator = combinations (n_set, k)
    # Convert the iterator to a list to view all results
    subsets_list = list (combinations_iterator)

    return [list (x) for x in subsets_list]


def get_subsets_of_k_adaptive (N, k):

    """
    Generates all unique subsets of size k from a given set of N numbers.
    
    Args:
    n_set: number of elements in set.
    k: The size of the subsets to return.
    
    Returns:
    A list of of lists, where each list is a unique subset of size k.
    """

    n_set = np.arange (1,N,1)
    
    # itertools.combinations returns an iterator of tuples
    combinations_iterator = combinations (n_set, k)
    # Convert the iterator to a list to view all results
    subsets_list = list (combinations_iterator)

    return [list (x) for x in subsets_list]





def find_partition (K, order) :
    # find the partition
    subsets = get_subsets_of_k (K, order)

    # keep track of number of odd flips
    odd = np.zeros (len (subsets), dtype=int)
    ct  = 0
    for sub in subsets :
        #print (sub)
        odd_sub = np.zeros (K)
        for i in range (K) :
            odd_sub[i] = np.sum (i >= np.array (sub))
       
        #print (odd_sub % 2) 
        odd[ct] = np.sum ( odd_sub % 2 == 1 )
        ct += 1

    return odd


def find_partition_specific (K, order) :
    # find the partition
    subsets = get_subsets_of_k (K, order)

    # keep track of number of odd flips
    odd = np.zeros ( (len (subsets), K), dtype=int)
    ct  = 0
    for sub in subsets :
        odd_sub = np.zeros (K)
        for i in range (K) :
            odd_sub[i] = np.sum (i >= np.array (sub))
       
        odd[ct,:] = odd_sub % 2
        ct += 1

    return odd


