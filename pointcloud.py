import numpy as np
from scipy.spatial import KDTree, cKDTree
from pykdtree.kdtree import KDTree as pyKDTree
import pylab


def best_orthogonal(rs, weigths=None):
    """
    finds the most orthogonal vector n to the set of vectors rs
    rs.shape = (N,2)

    returns n, lam

    where lam is the degree of anisotropy
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


class PointCloud(object):
    def __init__(self, rs,
                 bound_func_gradient=None,
                 R_repell=0.1,
                 n_neighbors=6,
                 t_divide=1.e20,
                 KDTreeType=pyKDTree):

        self._rs = np.array(rs)
        self._ts = np.zeros(self._rs.shape[0])
        self._generations = np.zeros(self._rs.shape[0])

        self._Fs = 0*self._rs
        self._vs = 0*self._rs
        self._t = 0

        self._t_divide = t_divide

        self._R_repell = R_repell

        self._KDTreeType = KDTreeType
        self._n_neighbors = n_neighbors
        if bound_func_gradient is None:
            bound_func_gradient = lambda x, t: 0*x

        self._bound_func_gradient = bound_func_gradient
        self.ktree = self._KDTreeType(self._rs)

    def _force_repell2(self):
        Fs = 0*self._rs

        # compute the indeces f the nearest neighbors
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        for i, (r, inds) in enumerate(zip(self._rs, indss)):
            # _, inds = self.ktree.query(r, min(len(self._rs),self._n_neighbors+1))
            # ignore r, i.e. inds[0]
            neighs = self._rs[inds[1:]]
            dists = r-neighs

            dabs_all = np.sqrt(np.sum(dists**2, -1))+1.e-10
            for d, dabs in zip(dists, dabs_all):
                # for d in dists:
                #     dabs = np.sqrt(np.sum(d**2))+1.e-10
                if dabs<2*self._R_repell:
                    Fs[i] += d/dabs*((2.*self._R_repell)-dabs)
                elif dabs>3*self._R_repell:
                    Fs[i] += .2*self._R_repell*d/dabs
                    # else:
                    #     Fs[i] -= .05*self._R_repell*d/dabs

        return Fs

    def _force_repell(self):
        rs = self._rs
        ndim = rs.shape[-1]

        # the indices of the nearest neighbors
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        # the distances to the nearest neighbors
        dists = rs-rs[indss[:, 1:]].transpose((1, 0, 2))
        dabs = np.linalg.norm(dists, axis=-1)+1.e-10

        indss_inner = dabs<2*self._R_repell
        indss_outer = dabs>3*self._R_repell

        # Force update
        # repelling
        Fs_inner_all = dists/dabs[..., np.newaxis]*((2.*self._R_repell)-dabs[..., np.newaxis])
        Fs_inner_all *= indss_inner[..., np.newaxis]
        Fs_inner = np.sum(Fs_inner_all, 0)

        # contracting
        Fs_outer_all = dists/dabs[..., np.newaxis]*0.2*self._R_repell
        Fs_outer_all *= indss_outer[..., np.newaxis]
        Fs_outer = np.sum(Fs_outer_all, 0)

        Fs = Fs_inner+Fs_outer

        return Fs

    def _force_bound(self):
        Fs = self._bound_func_gradient(self._rs, self._t)

        fnorm = np.sqrt(np.sum(Fs**2, -1))[:, np.newaxis]
        Fs *= 1./(1.e-6+fnorm)
        Fs *= np.minimum(fnorm, 4.*self._R_repell)
        return Fs

    def single_step(self, dt):
        """
        a single velocity corrected verlet step
        """
        self._rs += self._vs*dt+.5*self._Fs*dt**2
        v_half = self._vs+.5*self._Fs*dt
        self._Fs = self._force_repell()
        self._Fs += self._force_bound()
        self._vs = .97*v_half+.5*self._Fs*dt
        self._t += dt
        self._ts += dt

        self._divide()
        self.ktree = self._KDTreeType(self._rs)

    def step(self, dt0, n_divide=1, random_v=0):

        """
        a velocity corrected verlet step of size dt
        subdivided in n_divide substeps
        """
        dt = 1.*dt0/n_divide

        for _ in xrange(n_divide):
            self._rs += self._vs*dt+.5*self._Fs*dt**2
            v_half = self._vs+.5*self._Fs*dt
            self._Fs = self._force_repell()
            self._Fs += self._force_bound()
            self._vs = .97*v_half+.5*self._Fs*dt
            self._t += dt
            self._ts += dt
            self._divide()

        self._rs += 1./np.sqrt(3)*random_v*dt0*np.random.uniform(-1, 1, self._rs.shape)

        self.ktree = self._KDTreeType(self._rs)

    def _divide_times(self):
        _x = self._rs[:, -1]

        _x = self._rs[:, 0]

        divide_param = (_x-np.amin(_x))/(np.amax(_x)-np.amin(_x))
        return self._t_divide*(1.2*divide_param+.8*(1-divide_param))*(1+np.sqrt(self._generations))

    def _divide(self):
        # t_divide = self._t_divide

        t_divide = self._divide_times()

        inds = np.where(self._ts>t_divide*np.random.uniform(.9, 1.1, len(self._ts)))[0]

        r_new = []
        f_new = []
        v_new = []
        g_new = []

        if len(inds)>0:
            for ind in inds:
                # divide in random direction
                dx = .1*self._R_repell*np.random.uniform(-1, 1, self._rs.shape[-1])
                r_new.append(self._rs[ind]+dx)
                f_new.append(1.*self._Fs[ind])
                v_new.append(1.*self._vs[ind])
                g_new.append(self._generations[ind]+1)

                self._generations[ind] += 1
                self._rs[ind] -= dx
                self._ts[ind] = 0.

            self._rs = np.concatenate([self._rs, np.stack(r_new)])
            self._Fs = np.concatenate([self._Fs, np.stack(f_new)])
            self._vs = np.concatenate([self._vs, np.stack(v_new)])
            self._ts = np.concatenate([self._ts, np.zeros(len(inds))])
            self._generations = np.concatenate([self._generations, g_new])

    def draw(self, with_force=True, **kwargs):
        raise NotImplementedError()


class PointCloud2D(PointCloud):
    def draw(self, with_force=True, **kwargs):
        pylab.plot(self._rs[:, 0], self._rs[:, 1], "o", **kwargs)
        if with_force:
            pylab.quiver(self._rs[:, 0], self._rs[:, 1],
                         self._Fs[:, 0], self._Fs[:, 1])
        pylab.axis([-1.2, 1.2, -1.2, 1.2])

    def _ellipse_w_h_angle(self, dists):
        r0 = self._R_repell
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

    def draw_ellipses(self, **kwargs):
        from matplotlib.patches import Ellipse

        ax = pylab.gca()
        # draw ellipses according to packing
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        for i, (r, inds) in enumerate(zip(self._rs, indss)):

            # ignore r, i.e. inds[0]
            neighs = self._rs[inds[1:]]
            w, h, angle = self._ellipse_w_h_angle(r-neighs)

            ell = Ellipse(xy=r, width=w, height=h,
                          angle=angle)

            ax.add_artist(ell)

        pylab.axis([-1.2, 1.2, -1.2, 1.2])


class PointCloud3D(PointCloud):
    def draw(self, with_force=True, **kwargs):
        ax = plt.gca()
        ax.scatter(self._rs[:, 0], self._rs[:, 1], self._rs[:, 2], "o", **kwargs)
        if with_force:
            ax.quiver(self._rs[:, 0], self._rs[:, 1], self._rs[:, 2],
                      self._Fs[:, 0], self._Fs[:, 1], self._Fs[:, 2])


if __name__=='__main__':
    from time import time

    np.random.seed(0)
    rs = np.random.uniform(-.7, .7, (100, 2))

    p = PointCloud2D(rs, None, .1)

    t = time()
    p._Fs = p._force_repell()
    print "time for (n_points = %s):  %.1f ms"%(len(rs), 1000.*(time()-t))



    t = time()
    F2 = p._force_repell2()
    print "time for (n_points = %s):  %.1f ms"%(len(rs), 1000.*(time()-t))


    for i in xrange(200):
        print i, p._t
        p.step(0.2, 20, random_v=0.01)

        pylab.clf()
        p.draw()
        pylab.gca().set_aspect('equal')
        pylab.axis([-1., 1., -1., 1.])
        pylab.axis("off")
        pylab.pause(.1)


