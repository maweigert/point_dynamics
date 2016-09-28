import numpy as np
from pointcloud3 import PointCloud3

if __name__=='__main__':

    np.random.seed(0)

    N = 20

    phi = np.random.uniform(0, 2.*np.pi, N)
    t = np.arccos(np.random.uniform(-1, 1, N))
    rs = 1.*np.stack([np.cos(phi)*np.sin(t), np.sin(phi)*np.sin(t), np.cos(t)]).T


    def a_func(t):
        return 2*np.arctan(t/30.)/np.pi


    def bound_grad_limacon3(rs, t):
        a = a_func(t)
        x, y, z = rs.T

        # the function
        u = (z**2+x**2+y**2+a*z)**2-(z**2+x**2+y**2)
        u_norm = np.sign(u)*np.abs(u)**.5

        # the gradient
        dz = 2*(z**2+x**2+y**2+a*z)*(2.*z+a)-2.*z
        dy = 2*(z**2+x**2+y**2+a*z)*2.*y-2.*y
        dx = 2*(z**2+x**2+y**2+a*z)*2.*x-2.*x

        dr = np.stack([dx, dy, dz])
        normed = np.linalg.norm(dr, axis=0)
        dr *= 1./(1.e-8+normed)

        return -(dr*u_norm).T


    p = PointCloud3(rs, bound_grad_limacon3, .1, t_divide=15)

    from spimagine import volfig
    from rand_cmap import rand_cmap

    cmap = rand_cmap(200, type="soft")

    for i in xrange(1000):
        print i, p._t
        p.step(0.3, 20, random_v=0.01)

        p.draw_spimagine(cmap=cmap)

        w = volfig(1, raise_window = False)
        w.transform.setRotation(.006*i,0,1,0)
        w.saveFrame("tmp/fig_%s.png"%str(i).zfill(4))
