"""
Version for 3d


mweigert@mpi-cbg.de


"""


import numpy as np
# import pylab
# import matplotlib.pyplot as plt

from pointcloud import PointCloud



def orthogonal_vecs(rs, weigths=None):
    """
    do a pca on the point cloud, returning the main vectors

    rs.shape = (N,ndim)

    returns n, with n.shape = (ndim,ndim) and n[:,0] is the vector
    along the smallest variation  and n[:,-1] along the biggest
    """

    if weigths is None:
        weigths = np.ones(len(rs))

    # rs = rs - np.mean(rs,0)
    T = np.dot(rs.T*weigths, rs)
    u, s, v = np.linalg.svd(T)

    inds = np.argsort(s)

    n = u[:, inds]
    #n *= 1./(np.linalg.norm(n, axis = )+1.e-10)

    return n, s[inds]



class PointCloud3(PointCloud):
    """
    3d version of the PointCloud

    example:

    def bound_grad_sphere(rs, t):
        u = np.linalg.norm(rs,axis = -1) - 1.
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n,axis = 0)
        return -5*(n*u).T

    rs = np.random.uniform(-.7, .7, (100, 3))
    p = PointCloud3(rs, bound_grad_sphere, r_repell = 0.1, t_divide = 100.)

    p.step(.3,10)

    p.draw_spimagine()

    """
    def draw(self, with_force=True, **kwargs):
        ax = plt.gca()
        ax.scatter(self._rs[:, 0], self._rs[:, 1], self._rs[:, 2], "o", **kwargs)
        if with_force:
            ax.quiver(self._rs[:, 0], self._rs[:, 1], self._rs[:, 2],
                      self._Fs[:, 0], self._Fs[:, 1], self._Fs[:, 2])

    def _divide_times(self):
        _x = self._rs[:, -1]

        divide_param = (_x-np.amin(_x))/(np.amax(_x)-np.amin(_x))
        return self._t_divide*(1.1*divide_param+.9*(1-divide_param))*(1+np.sqrt(self._generations))

    def _ellipsoid_rs_mat(self, dists):
        from spimagine.utils.transform_matrices import mat4_rotation_euler
        r0 = self._r_repell

        # closer ones should matter more
        weigths = 1./(1.+np.sum(dists**2, -1)**2/(2.*r0)**4)


        ns, ss = orthogonal_vecs(dists, weigths=weigths)

        ss = ss**.2
        ss = np.clip(ss,0.1,10)
        rs = 1./(ss+1.e-10)

        rs *= (r0**3/np.prod(rs))**(1./3)
        #rs = r0*rs/np.amax(rs)

        return rs, ns


    def draw_spimagine(self, cmap = None, **kwargs):
        from PyQt4 import QtGui
        from spimagine import volfig, EllipsoidMesh

        w = volfig(1, raise_window = False)
        w.resize(1400,1400)
        w.glWidget.meshes = []
        # draw ellipses according to packing
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        t_divides = self._divide_times()
        for i, (r, t,t_divide, inds, id) in enumerate(zip(self._rs, self._ts, t_divides,indss, self._ids)):
            # ignore r, i.e. inds[0]
            neighs = self._rs[inds[1:]]

            rs, m = self._ellipsoid_rs_mat(r-neighs)

            #let young cells be smaller
            fac = 1-.3*np.exp(-10.*t/t_divide)
            rs *= fac



            if not cmap is None:
                col = cmap(id%cmap.N)
            else:
                col = (.9,1.,.2)

            ell = EllipsoidMesh(rs = rs,pos = r,facecolor = col, transform_mat = m,
                                n_theta = 20, n_phi = 30)

            w.glWidget.add_mesh(ell)

        w.glWidget.render()
        w.glWidget.refresh()
        QtGui.QApplication.instance().processEvents()



if __name__=='__main__':


    def bound_grad_sphere(rs, t):
        u = np.linalg.norm(rs,axis = -1) - .8
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n,axis = 0)
        return -5*(n*u).T


    np.random.seed(0)
    rs = np.random.uniform(-.7, .7, (200, 3))
    p = PointCloud3(rs, bound_grad_sphere, r_repell = 0.1, t_divide = 100.)


    from spimagine import volfig

    for _ in xrange(100):
        p.step(0.1, 20, random_v=0.01)
        p.draw_spimagine()



