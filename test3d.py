import numpy as np
from scipy.spatial import KDTree, cKDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pointcloud import PointCloud3D


if __name__=='__main__':
    # first on a circle

    np.random.seed(0)

    R_circle = 1.

    R_cell = .07
    n_points = 100

    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.arccos(np.random.uniform(-1, 1, n_points))

    rs = (R_circle*np.stack([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])).T


    # rs = np.random.uniform(-2,2,(n_points,3))

    def bound_grad_limacon3(rs, t):
        #a = 2.*np.arctan(t/2000.)/np.pi
        a = 8.*np.arctan(t/100.)/np.pi

        x, y, z = rs.T

        # the function
        u = (z**2+x**2+y**2+a*z)**2-(z**2+x**2+y**2)

        # the gradient
        dz = 2*(z**2+x**2+y**2+a*z)*(2.*z+a)-2.*z
        dy = 2*(z**2+x**2+y**2+a*z)*2.*y-2.*y
        dx = 2*(z**2+x**2+y**2+a*z)*2.*x-2.*x

        dr = np.stack([dx, dy, dz])
        normed = np.sum(dr**2, 0)
        dr *= 1./(1.e-8+normed)

        return -(dr*u).T


    p = PointCloud3D(rs, bound_grad_limacon3, R_cell, t_divide=40.)

    p.single_step(0.1)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    plt.show()

    for i in xrange(500):
        print i, p._t
        # p.step(0.2, 10, random_v=0.01)
        p.step(0.5, 10, random_v=0.01)

        # for j in xrange(10):
        #     p.single_step(0.02)

        ax.cla()
        p.draw(with_force=False, s=10.)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5,2.5)
        ax.set_zlim(-4, 1.)
        ax._axis3don = False
        ax.set_aspect('equal')
        plt.pause(.1)
        plt.savefig("output3/fig_%s.png"%str(i).zfill(4))
