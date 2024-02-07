import numpy as np

def csm_binary(D, kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If kappa = 0, take all neighbors
    If kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise kappa is the number of mutual neighbors to consider
    """
    from scipy.sparse import coo_matrix
    N = D.shape[0]
    M = D.shape[1]
    if kappa == 0:
        return np.ones((N, M))
    elif kappa < 1:
        NNeighbs = int(np.round(kappa*M))
    else:
        NNeighbs = kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()

def csm_binary_mutual(D, kappa):
    """
    Take the binary AND between the nearest neighbors in one direction
    and the other
    """
    B1 = csm_binary(D, kappa)
    B2 = csm_binary(D.T, kappa).T
    return B1*B2

def get_csm(X, Y, do_sqrt=True):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    if do_sqrt:
        C = np.sqrt(C)
    return C

def get_csm_cosine(X, Y):
    """
    Cosine distance between two pairs of vectors
    """
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y**2, 1))
    YNorm[YNorm == 0] = 1
    D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
    D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
    return D

def sliding_csm(D, win):
    """
    Perform the effect of a sliding window on an CSM by averaging
    along diagonals
    Parameters
    ----------
    D: ndarray(M, N)
        A cross-similarity matrix
    win: int
        Window length
    """
    M = D.shape[0] - win + 1
    N = D.shape[1] - win + 1
    S = np.zeros((M, N))
    J, I = np.meshgrid(np.arange(N), np.arange(M))
    for i in range(-M+1, N):
        x = np.diag(D, i)**2
        x = np.array([0] + x.tolist())
        x = np.cumsum(x)
        x = x[win::] - x[0:-win]
        S[np.diag(I, i), np.diag(J, i)] = np.sqrt(x)
    return S
