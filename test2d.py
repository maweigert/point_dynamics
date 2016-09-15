import numpy as np
import pylab
from pointcloud import PointCloud2D


if __name__=='__main__':
    # first on a circle

    np.random.seed(0)

    R_circle = 1.

    R_cell = .07
    n_points = 20

    # n_points = 1000


    phi = np.random.uniform(0, 1.5*np.pi, n_points)
    # rs = (R_circle*np.stack([np.cos(phi), np.sin(phi)])*np.random.uniform(.9,1.1,n_points)).T


    rs = (R_circle*np.stack([np.cos(phi), np.sin(phi)])).T

    phi = np.pi+np.random.uniform(-1.2, 1.2, n_points)
    #phi = np.random.uniform(0,2*np.pi, n_points)

    rs = (R_circle*np.stack([np.cos(phi), np.sin(phi)])).T


    def bound_grad(rs, t):
        d = np.sqrt(np.sum(rs**2, -1))-R_circle
        r_norm = 1.*rs/(np.sqrt(np.sum(rs**2, -1))[:, np.newaxis]+1.e-10)
        return -r_norm*d[:, np.newaxis]


    def bound_grad_ellipse(rs, t):
        eps = 1+.1*np.arctan(.01*t)/np.pi
        dist = np.sqrt(rs[:, 0]**2+eps**2*rs[:, 1]**2)[:, np.newaxis]
        d = dist-1.
        r_norm = 1.*rs/(dist+1.e-10)
        return -r_norm*d


    def bound_grad_limacon(rs, t):
        # = 2.*np.arctan(t/2000.)/np.pi
        a = 3.*np.arctan(t/200.)/np.pi
        x = rs[:, 0]
        y = rs[:, 1]
        # the function
        u = (x**2+y**2+a*x)**2-(x**2+y**2)

        u = np.sign(u)*np.abs(u)**.5
        # the gradient
        dx = 2*(x**2+y**2+a*x)*(2.*x+a)-2.*x
        dy = 2*(x**2+y**2+a*x)*2.*y-2.*y
        dr = np.stack([dx, dy])
        normed = np.sum(dr**2, 0)
        dr *= 1./(1.e-8+normed)

        return -(dr*u).T


    p = PointCloud2D(rs, bound_grad_limacon, R_cell, t_divide=50)

    import pylab

    pylab.figure(1)
    pylab.clf()
    pylab.show()
    pylab.axis("equal")

    p.draw_ellipses()


    for i in xrange(3000):
        print i, p._t
        p.step(0.2, 20, random_v=0.01)

        # for j in xrange(10):
        #     p.single_step(0.02)
        if i%5==0:
            pylab.clf()
            #p.draw(with_force=False, ms=2.)
            p.draw_ellipses()
            pylab.gca().set_aspect('equal')
            pylab.axis([-2.5, 1.1, -1.8, 1.8])

            pylab.axis("off")
            pylab.pause(.1)
            #pylab.savefig("output/fig_%s.png"%str(i).zfill(4))


