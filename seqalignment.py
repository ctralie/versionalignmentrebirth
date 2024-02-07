"""
Tools for local alignment of sequences
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def qmax(csm, match=1, gap_opening=0.5, gap_extension=0.5):
    """
    Perform Serra's qmax alignment on a binary cross-similarity matrix

    Parameters
    ----------
    csm: ndarray(M, N)
        Binary cross-similarity matrix
    match: float
        Score for match
    gap_opening: float
        Score for opening a gap
    gap_extension: float
        Score for extending a gap
    
    Returns
    -------
    max_score: float
        Max Smith Waterman score over all diagonals
    D: ndarray(M, N)
        Smith Waterman dynamic programming table
    """

    M = csm.shape[0]
    N = csm.shape[1]
    D = np.zeros((M, N))
    max_score = 0
    for i in range(2, M):
        for j in range(2, N):
            # Measure the diagonal when a similarity is found in the input matrix
            if csm[i, j] == 1:
                c1 = D[i-1, j-1]
                c2 = D[i-2, j-1]
                c3 = D[i-1, j-2]
                D[i, j] = np.max(np.array([c1, c2, c3])) + match
            else:
                # Apply gap penalty gap_opening for disruption and gap_extension when 
                # similarity is not found in the input matrix
                gamma = gap_extension
                if csm[i-1, j-1] == 1:
                    gamma = gap_opening
                c1 = D[i-1, j-1] - gamma
                
                gamma = gap_extension
                if csm[i-2, j-1] == 1:
                    gamma = gap_opening
                c2 = D[i-2, j-1] - gamma
                
                gamma = gap_extension
                if csm[i-1, j-2] == 1:
                    gamma = gap_opening
                c3 = D[i-1, j-2] - gap_opening
                D[i, j] = np.max(np.array([c1, c2, c3, 0.0]))
            if D[i, j] > max_score:
                max_score = D[i, j]
    return max_score, D


@jit(nopython=True)
def swalignimpconstrained(csm, mismatch=-1, match=1, gap_opening=-0.5, gap_extension=-0.7):
    """
    Perform a constrianed Smith-Waterman alignment on a binary matrix
    
    Parameters
    ----------
    csm: ndarray(M, N)
        Binary cross-similarity matrix
    mismatch: float
        Score for mismatch
    match: float
        Score for match
    gap_opening: float
        Score for opening a gap
    gap_extension: float
        Score for extending a gap
    
    Returns
    -------
    max_score: float
        Max Smith Waterman score over all diagonals
    D: ndarray(M, N)
        Smith Waterman dynamic programming table
    """
    
    N = csm.shape[0]+1
    M = csm.shape[1]+1
    D = np.zeros((N, M))
    max_score = 0
    for i in range(3, N):
        for j in range(3, M):
            MS = mismatch
            if csm[i-1, j-1] == 1:
                MS = match
            #H_(i-1, j-1) + S_(i-1, j-1) + delta(S_(i-2,j-2), S_(i-1, j-1))
            delta = 0
            if csm[i-1, j-1] == 0:
                if csm[i-2, j-2] > 0:
                    delta = gap_opening
                else:
                    delta = gap_extension
            d1 = D[i-1, j-1] + MS + delta
            
            #H_(i-2, j-1) + S_(i-1, j-1) + delta(S_(i-3, j-2), S_(i-1, j-1))
            delta = 0
            if csm[i-1, j-1] == 0:
                if csm[i-3, j-2] > 0:
                    delta = gap_opening
                else:
                    delta = gap_extension
            d2 = D[i-2, j-1] + MS + delta
            
            #H_(i-1, j-2) + S_(i-1, j-1) + delta(S_(i-2, j-3), S_(i-1, j-1))
            delta = 0
            if csm[i-1, j-1] == 0:
                if csm[i-2, j-3] > 0:
                    delta = gap_opening
                else:
                    delta = gap_extension
            
            d3 = D[i-1, j-2] + MS + delta
            D[i, j] = np.max(np.array([d1, d2, d3, 0.0]))
            if (D[i, j] > max_score):
                max_score = D[i, j]
    return max_score, D

