"""


mweigert@mpi-cbg.de

"""
import numpy as np
from gputools import noise


def orthogonal_vecs(rs, weigths=None):
    """
    finds a triad that is maximal orthogonal to a set of points rs by performing a pca on rs
    and returns the eigensystem m (i.e. the matrix whose columns are the orthogonal pca vectors)
    and eigenvalues s (sorted by magnitude)


    :param rs: ndarray of shape (N,ndim)
        the coordinates of the points
    :param weigths: ndarray/list/tuple of length N, optional
        if given, assigns different weights to individual points

    :return:
        m,s
        eigensystem matrix m
        eigenvalues s

    m[:,i] are the eigenvectors
    m[:,0] is the dominant direction (most orthogonal to rs)
    """

    if weigths is None:
        weigths = np.ones(len(rs))

    rs = rs-np.mean(rs, 0)
    T = np.dot(rs.T*weigths, rs)
    u, s, v = np.linalg.svd(T)
    inds = np.argsort(s)
    n = u[:, inds]

    return n, s[inds]


def perlin_ellipse2(shape=(21, 21), rs=(5, 5), transform_m=np.identity(2),
                    offset = 0, shift = (0,0), scale = 6):
    Xs = np.stack(np.meshgrid(*[np.arange(s)-s/2. for s in shape], indexing="ij"))

    # transform points
    Xs = np.dot(Xs.T, transform_m.T).T

    mask = (np.sum([X**2/r**2 for X, r in zip(Xs, rs)], axis=0)<1)

    p = noise.perlin2(shape, scale=scale, shift = shift)

    p = (p-np.amin(p))/(np.amax(p)-np.amin(p))

    res = np.zeros(shape, np.float32)

    res[mask] = p[mask]+offset

    return res, mask


def perlin_ellipse3(shape=(21, 21, 21), rs=(5, 5, 5), transform_m=np.identity(3),
                    offset = 0, shift = (0,0,0), scale = 6):

    Xs = np.stack(np.meshgrid(*[np.arange(s)-s/2. for s in shape], indexing="ij"))

    # transform points
    Xs = np.dot(Xs.T, transform_m.T).T

    mask = (np.sum([X**2/r**2 for X, r in zip(Xs, rs)], axis=0)<1)

    p = noise.perlin3(shape, scale=scale, shift = shift)

    p = (p-np.amin(p))/(np.amax(p)-np.amin(p))

    res = np.zeros(shape, np.float32)

    res[mask] = p[mask]+offset

    return res, mask

if __name__=='__main__':
    from spimagine.utils.transform_matrices import *

    u2, _ = perlin_ellipse2(shape=(65, 65), rs=(10, 20),
                         transform_m=mat4_rotation(.3,0,0,1)[:2,:2],
                         offset = .4)


    u3, _ = perlin_ellipse3(shape=(65, 65, 65), rs=(10, 20, 10),
                         transform_m=mat4_rotation(.3,1,0,0)[:3,:3],
                            shift = (.4,)*3,
                         offset = .4)
