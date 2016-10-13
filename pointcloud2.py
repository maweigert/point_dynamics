"""
Version for 2d

mweigert@mpi-cbg.de

"""


import numpy as np
import matplotlib.pyplot as plt

from pointcloud import PointCloud
from utils import orthogonal_vecs, perlin_ellipse2


def best_orthogonal(rs, weigths=None):
    """
    finds the most orthogonal vector n to the set of N vectors rs of dimension ndim
    rs.shape = (N,ndim)

    returns n, aniso

    where aniso is the degree of anisotropy
    """

    if weigths is None:
        weigths = np.ones(len(rs))

    # rs = rs - np.mean(rs,0)
    T = np.dot(rs.T*weigths, rs)
    u, s, v = np.linalg.svd(T)
    ind_min = np.argmin(s)
    n = u[:, ind_min]
    n *= 1./(np.linalg.norm(n)+1.e-10)

    return n, np.sqrt((np.amax(s)+1.e-8)/(np.amin(s)+1.e-8)) if len(rs)>1 else 1000.




class PointCloud2(PointCloud):
    """
    2d version of the PointCloud

    example:

    def bound_grad_circle(rs, t):
        u = np.linalg.norm(rs,axis = -1) - 1.
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n,axis = 0)
        return -5*(n*u).T

    rs = np.random.uniform(-.7, .7, (100, 2))
    p = PointCloud2(rs, boundary_gradient, r_repell = 0.1, t_divide = 1.)

    p.step(.3,10)

    p.draw_ellipses(with_force = True)

    p._lineage.show()

    p.save("positions.dat")
    


    """


    def draw(self, with_force=True, **kwargs):
        plt.plot(self._rs[:, 0], self._rs[:, 1], "o", **kwargs)
        if with_force:
            plt.quiver(self._rs[:, 0], self._rs[:, 1],
                         self._Fs[:, 0], self._Fs[:, 1])
        plt.axis([-1.2, 1.2, -1.2, 1.2])

    def _ellipse_w_h_angle(self, dists):
        r0 = self._r_repell
        # closer ones should matter more
        weigths = 1./(1.+np.sum(dists**2, -1)**2/(2.*r0)**4)

        # dists *= 1./(1.e-3+np.sum(dists**2,-1)[:,np.newaxis])
        n, aniso = best_orthogonal(dists, weigths=weigths)

        aniso = aniso**(np.amax(weigths))
        aniso = aniso**.5
        aniso = np.clip(aniso, 0.2, 5.)

        angle = np.arctan2(n[1], n[0])/np.pi*180
        w, h = r0*np.sqrt(aniso), r0/np.sqrt(aniso)

        return w, h, angle




    def draw_ellipses(self, with_force = False, cmap = None, **kwargs):

        from matplotlib.patches import Ellipse

        ax = plt.gca()
        # draw ellipses according to packing
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))




        for i, (r, inds, id) in enumerate(zip(self._rs, indss, self._ids)):
            # ignore r, i.e. inds[0]
            neighs = self._rs[inds[1:]]
            w, h, angle = self._ellipse_w_h_angle(r-neighs)

            ell = Ellipse(xy=r, width=w, height=h,
                          angle=angle)

            if not cmap is None:
                ell.set_facecolor(cmap(id%cmap.N))

            ax.add_artist(ell)

        if with_force:
            plt.quiver(self._rs[:, 0], self._rs[:, 1],
                         self._Fs[:, 0], self._Fs[:, 1])


    def create_signal_label(self, shape, extent=((-1,1),(-1,1)),
                            scale = 10, intens = 100,
                            poisson_noise = False,
                            gaussian_noise = 0,
                            blur_sigma= 0):

        from spimagine.utils.transform_matrices import mat4_rotation
        from gputools import convolve_sep2

        signal = np.zeros(shape)
        label = np.zeros(shape, dtype='int32') # default int is 64, but that's too deep for Fiji. 2^32 is big enough.

        units = tuple([1.*(ext[1]-ext[0])/(s-1.) for s,ext in zip(shape, extent)])

        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        for i, (r, inds, id) in enumerate(zip(self._rs, indss, self._ids)):


            neighs = self._rs[inds[1:]]
            w, h, angle = self._ellipse_w_h_angle(r-neighs)


            m = mat4_rotation(-angle/180*np.pi,0,0,1)[:2,:2].T

            max_w = int(np.ceil(max([_b/_a for _a,_b in zip(units,(h,w))])))

            ind_arr = [int(1.*(s-1)*(_x-ext[0])/(ext[1]-ext[0])) for ext, s, _x in zip(extent, shape, r[::-1])]
            slice_mask = tuple([slice(j-int(max_w/2), j+int(max_w/2)) for j in ind_arr])

            sig_part = signal[slice_mask]
            label_part = label[slice_mask]

            density, mask = perlin_ellipse2(sig_part.shape,(.5*h/units[0],.5*w/units[1]),
                                            offset = .4,
                                            shift = (id%np.pi+0.05*np.random.normal(0,1),id%np.pi+0.05*np.random.normal(0,1)),
                                            transform_m=m,
                                            scale = scale)


            sig_part[mask] = intens*density[mask]

            label_part[mask] = id+1

        if blur_sigma>0:
            hx = np.exp(-np.arange(-2*blur_sigma,2*blur_sigma)/blur_sigma**2)
            hx *= 1./np.sum(hx)
            signal = convolve_sep2(signal,hx,hx)


        if poisson_noise:
            signal = np.random.poisson(signal.astype(int))

        if gaussian_noise>0:
            signal = np.maximum(0,signal+gaussian_noise*np.random.normal(0,1,signal.shape))


        return signal, label


if __name__=='__main__':
    from time import time

    np.random.seed(0)
    rs = np.random.uniform(-.7, .7, (40, 2))


    def bound_grad_circle(rs, t):
        r0 = 1.
        u = np.linalg.norm(rs,axis = -1) -r0

        # the gradient
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n,axis = 0)
        return -5*(n*u).T

    p = PointCloud2(rs, bound_grad_circle, r_repell = .1, t_divide=100000)

    for _ in xrange(200):
        p.step(.3,10)
        plt.clf()
        p.draw_ellipses(with_force=True)
        plt.axis("equal")
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.pause(.1)