import numpy as np
import math
import scipy
from scipy.stats import norm


def compute_Ma (traj) :
    """
    Computes the (approx) expected number of local maxima at a given distance from the ancestral sequence.

    traj: a vector of length M+1 which contains the expected values of the mutants along the trajectory.
    """
    M = len (traj) - 1
    print (M)

    emas = np.zeros (M+1)
    for i in range (2, M-1) :
        print (i)
        emas[i]  = (1. - (scipy.stats.norm.cdf (traj[i-1]))**(M) ) * ((M-i-1) / M)
        emas[i] += (1. - (scipy.stats.norm.cdf (traj[i+1]))**(M) ) * ((i-1) / M)
        emas[i] += (1. - (scipy.stats.norm.cdf (traj[i+1]))**(M-1) ) / (M-1)
        emas[i] += (math.comb (M, i) - M) / (M+1)

    # for the one step
    emas[1]  =  (1. - (scipy.stats.norm.cdf (traj[2]))**(M-1)) / (M-1)
    emas[1] += (M-2)/(M+1)
    emas[M-1]  = (1. - (scipy.stats.norm.cdf (traj[M]))**(M-1)) / (M-1)
    emas[M-1] += (1. - (scipy.stats.norm.cdf (traj[M]))**(M)) * (M-2) / M
    emas[M] = (scipy.stats.norm.cdf (traj[M]))**(M-1)
    
    return emas


def blom_expected_values(L, k_top=10, alpha=0.375):
    """
    From Gemini.
    Computes the Blom approximation for the expected values of the 
    top K order statistics from L draws of N(0,1).
    
    Parameters:
    - L: Sample size.
    - k_top: Number of top statistics to return (1 = maximum).
    - alpha: The Blom constant (default 0.375 for normal distribution).
    """
    # k is the rank from largest (1) to smallest (L)
    k = np.arange(1, k_top + 1)
    
    # i is the rank from smallest (1) to largest (L)
    i = L - k + 1
    
    # Calculate Blom plotting positions (P_i)
    # P_i = (i - alpha) / (L - 2*alpha + 1)
    p = (i - alpha) / (L - 2 * alpha + 1)
    
    # Inverse normal CDF (probit) to get expected values
    expected_values = scipy.stats.norm.ppf(p)
    
    return expected_values


def order_statistic_moments(L, k_top=1, alpha=0.375):
    """
    From Gemini.
    Approximates the Mean and Variance of top K order statistics 
    using David & Nagaraja's second-order Taylor expansion.
    """
    # Ranks: k=1 is the maximum
    k = np.arange(1, k_top + 1)
    i = L - k + 1
    
    # Blom Plotting Position
    p = (i - alpha) / (L - 2 * alpha + 1)
    q = 1 - p
    
    # First-order location (Expected Value)
    z = norm.ppf(p)
    
    # Density and its derivative at z
    phi = norm.pdf(z)
    #phi_prime = -z * phi  # Derivative of standard normal PDF
    
    # Standard First-Order Variance (Delta Method)
    # Var ≈ (p*q) / ((L+2) * phi^2)
    var_1st = (p * q) / ((L + 2) * (phi**2))
    
    # Second-Order Correction Term
    # This accounts for the curvature of the normal distribution
    # derived from David & Nagaraja Eq. 9.2.4
    #correction = 1 + (var_1st * ( (phi_prime / phi)**2 - (phi_prime / phi) ))
    
    #var_2nd = var_1st * correction
    
    return z, var_1st

