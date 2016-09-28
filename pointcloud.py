"""

Defines the PointCloud base class that simulates the dynamics of repelling, dividing points on a manifold

mweigert@mpi-cbg.de

Todo:
- include lineage in class
- make points saveable each timestep
"""


import numpy as np
# from scipy.spatial import KDTree, cKDTree
from pykdtree.kdtree import KDTree as pyKDTree
from treelib import Tree


class PointCloud(object):
    """
    Simulates the dynamics of a collection of interacting repelling points that are
    constrained to a manifold and can divide

    This is an abstract base class, for instantiations in 2d and 3d use

    PointCloud2 / PointCloud3



    """
    def __init__(self,
                 rs,
                 bound_func_gradient=None,
                 r_repell=0.1,
                 n_neighbors=6,
                 t_divide=1.e20,
                 KDTreeType=pyKDTree):

        self._rs = np.array(rs)

        N = self._rs.shape[0]
        self._ts = np.zeros(N)
        self._generations = np.zeros(N)

        self._Fs = 0*self._rs
        self._vs = 0*self._rs
        self._t = 0



        self._t_divide = t_divide

        self._ids = np.arange(N)
        self._id_max = N-1
        self._parent_ids = -np.ones(N)

        self._lineage = Tree()
        self._lineage.create_node(-1,-1)  #root node
        for id in self._ids:
            self._lineage.create_node(id,id, parent = -1)

        self._r_repell = r_repell

        self._KDTreeType = KDTreeType
        self._n_neighbors = n_neighbors

        if bound_func_gradient is None:
            bound_func_gradient = lambda x, t: 0*x

        self._bound_func_gradient = bound_func_gradient
        self.ktree = self._KDTreeType(self._rs)


    def _force_repell(self):
        rs = self._rs
        ndim = rs.shape[-1]

        # the indices of the nearest neighbors
        _, indss = self.ktree.query(self._rs, min(len(self._rs), self._n_neighbors+1))

        # the distances to the nearest neighbors
        dists = rs-rs[indss[:, 1:]].transpose((1, 0, 2))
        dabs = np.linalg.norm(dists, axis=-1)+1.e-10

        indss_inner = dabs<2*self._r_repell
        indss_outer = dabs>3*self._r_repell

        # Force update
        # repelling
        Fs_inner_all = dists/dabs[..., np.newaxis]*((2.*self._r_repell)-dabs[..., np.newaxis])
        Fs_inner_all *= indss_inner[..., np.newaxis]
        Fs_inner = np.sum(Fs_inner_all, 0)

        # contracting
        Fs_outer_all = dists/dabs[..., np.newaxis]*0.2*self._r_repell
        Fs_outer_all *= indss_outer[..., np.newaxis]
        Fs_outer = np.sum(Fs_outer_all, 0)

        Fs = Fs_inner+Fs_outer

        return Fs

    def _force_bound(self):
        Fs = self._bound_func_gradient(self._rs, self._t)

        fnorm = np.sqrt(np.sum(Fs**2, -1))[:, np.newaxis]
        Fs *= 1./(1.e-6+fnorm)
        Fs *= np.minimum(fnorm, 4.*self._r_repell)
        return Fs

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
        _x = self._rs[:, 0]
        divide_param = (_x-np.amin(_x))/(np.amax(_x)-np.amin(_x))
        return self._t_divide*(1.2*divide_param+.8*(1-divide_param))*(1+np.sqrt(self._generations))

    def _divide(self):
        t_divide = self._divide_times()

        inds = np.where(self._ts>t_divide*np.random.uniform(.9, 1.1, len(self._ts)))[0]

        r_new = []
        f_new = []
        v_new = []
        g_new = []
        id_new = []
        parents_new = []

        if len(inds)>0:
            for ind in inds:
                # divide in random direction
                dx = .1*self._r_repell*np.random.uniform(-1, 1, self._rs.shape[-1])
                r_new.append(self._rs[ind]+dx)
                f_new.append(1.*self._Fs[ind])
                v_new.append(1.*self._vs[ind])
                g_new.append(self._generations[ind]+1)

                parent = self._ids[ind]
                child1 , child2 = self._id_max+1, self._id_max+2
                id_new.append(child1)
                parents_new.append(parent)



                self._generations[ind] += 1
                self._rs[ind] -= dx
                self._ts[ind] = 0.
                self._ids[ind] = child2
                self._id_max += 2
                self._parent_ids[ind] = parent
                self._lineage.create_node(child1,child1, parent = parent)
                self._lineage.create_node(child2,child2, parent = parent)



            self._rs = np.concatenate([self._rs, np.stack(r_new)])
            self._Fs = np.concatenate([self._Fs, np.stack(f_new)])
            self._vs = np.concatenate([self._vs, np.stack(v_new)])
            self._ts = np.concatenate([self._ts, np.zeros(len(inds))])
            self._ids = np.concatenate([self._ids, np.array(id_new)])
            self._generations = np.concatenate([self._generations, g_new])
            self._parent_ids = np.concatenate([self._parent_ids, parents_new])


    def save(self, fname):
        """
        save the positions to file fname

        format

        id  parent_id x y z
        """

        data = np.hstack([self._ids[:,np.newaxis],self._parent_ids[:,np.newaxis], self._rs])
        with open(fname,"w") as f:
            np.savetxt(f, data, header = "#id  parent_id  x y z")


    def load(self, fname):
        """
        load the positions from file fname
        """
        try:
            data = np.genfromtxt(fname)
            self._id = data[:,0]
            self._parent_ids = data[:,1]
            self._rs = data[:,2:]
            self._id_max = np.amax(self._id)

        except Exception as e:
            print "could not load %s"%fname
            print e



    def draw(self, with_force=True, **kwargs):
        raise NotImplementedError()


if __name__=='__main__':
    pass