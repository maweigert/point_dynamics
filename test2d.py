import numpy as np
import pylab
from pointcloud2 import PointCloud2
from rand_cmap import rand_cmap

if __name__=='__main__':
    # first on a circle

    np.random.seed(0)

    R_circle = 1.
    R_repell = .07
    n_points = 20

    phi = np.pi+np.random.uniform(-1.2, 1.2, n_points)
    rs = (R_circle*np.stack([np.cos(phi), np.sin(phi)])).T


    def bound_grad_limacon(rs, t):
        a = 2.*np.arctan(t/100.)/np.pi
        x = rs[:, 0]
        y = rs[:, 1]
        # the function
        u = (x**2+y**2+a*x)**2-(x**2+y**2)

        u_abs = np.sign(u)*np.abs(u)**.5
        # the gradient
        dx = 2*(x**2+y**2+a*x)*(2.*x+a)-2.*x
        dy = 2*(x**2+y**2+a*x)*2.*y-2.*y
        dr = np.stack([dx, dy])
        normed = np.sum(dr**2, 0)
        dr *= 1./(1.e-8+normed)

        return -(dr*u).T


    p = PointCloud2(rs, bound_grad_limacon, R_repell, t_divide=50)

    import matplotlib.pyplot as plt

    plt.figure(1, facecolor = "k")
    plt.clf()
    plt.show()
    plt.axis("equal")

    p.draw_ellipses()

    cmap  = rand_cmap(300, type  ="soft")

    for i in xrange(3000):
        print i, p._t
        p.step(0.2, 20, random_v=0.01)

        if i%5==0:
            plt.clf()
            p.draw_ellipses(cmap = cmap)


            p.save("output/positions_%s.dat"%str(i).zfill(4))

            plt.gca().set_aspect('equal')
            plt.gca().set_axis_bgcolor('red')
            plt.axis([-2.5, 1.1, -1.8, 1.8])

            plt.axis("off")
            plt.pause(.1)

            #pylab.savefig("output/fig_%s.png"%str(i).zfill(4))


