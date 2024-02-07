"""
Algorithms for finer scale alignment that are more precise but slower
"""
from csmssmtools import get_csm, get_csm_cosine, sliding_csm, csm_binary_mutual
from audioutils import get_oti
import numpy as np

def get_alignment_score_serra(chroma1, chroma2, wins_per_block, kappa=0.095):
    """
    Perform Serra's qmax alignment technique
    https://iopscience.iop.org/article/10.1088/1367-2630/11/9/093017/pdf

    Parameters
    ----------
    chroma1: ndarray(12, n_frames)
        Chroma features for first tune (should be aggregated)
    chroma2: list of features
        Chroma features for second tune (should be aggregated)
    wins_per_block: int
        Number of windows to use in each diagonally averaged block
    kappa: float
        Proportion of mutual nearest neighbors to use in similarity network
        fusion and in binary cross-similarity matrix

    Returns
    -------
    float: Maximum similarity between tunes
    """
    from seqalignment import qmax
    from csmssmtools import get_csm_cosine
    ## Step 1: Compute the optimal transposition index between the
    ## two chroma vectors, and transpose the first one accordingly
    oti = get_oti(np.mean(chroma1, axis=1), np.mean(chroma2, axis=1))
    chroma1 = np.roll(chroma1, oti, axis=0)

    ## Step 2: Compute the binary similarity between lagged versions
    ## of the chromas, and perform qmax to score the best alignment
    csm = get_csm_cosine(chroma1.T, chroma2.T)
    csm = sliding_csm(csm, wins_per_block)
    B = csm_binary_mutual(csm, kappa)
    return qmax(B)[0]
