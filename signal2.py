"""


mweigert@mpi-cbg.de

"""

import numpy as np
from pointcloud2 import PointCloud2
import os

def safemkdirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if __name__=='__main__':
    from time import time

    np.random.seed(0)
    rs = np.random.uniform(-1., 1., (40, 2))

    phi = np.linspace(0,2*np.pi,40+1)[:40]
    rs = np.stack([.8*np.cos(phi), .8*np.sin(phi)]).T

    def bound_grad_circle(rs, t):
        r0 = .5
        u = np.linalg.norm(rs, axis=-1)-r0

        # the gradient
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n, axis=0)
        return -5*(n*u).T

    def bound_grad_basin(rs, t):
        # the gradient

        n = 1.*rs.T
        u = np.linalg.norm(n, axis=0)
        n /= 1.e-10+u
        return -.01*(n*np.sqrt(u)).T

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

    def bound_grad_box(rs,t):
        x = rs[:,0]
        y = rs[:,1]
        dx = np.zeros(x.shape)
        dx[np.where(x>0.8)] = -1.0
        dx[np.where(x<-0.8)] = 1.0
        dy = np.zeros(y.shape)
        dy[np.where(y > 0.8)] = -1.0
        dy[np.where(y < -0.8)] = 1.0
        dr = np.stack([dx,dy])
        return dr.T

    # p = PointCloud2(rs, bound_grad_limacon, r_repell=.1, t_divide=10)
    p = PointCloud2(rs, bound_grad_box, r_repell=.1, t_divide=10)

    # save using skimage to write grayscale (not RGB). Suppress low contrast UserWarning.
    # use separate directories for different image types, allows drag n' drop stack opening in Fiji.
    import skimage.io as io
    import warnings

    sigdir = "data2d/box/signal/"
    labdir = "data2d/box/label/"
    safemkdirs(sigdir)
    safemkdirs(labdir)

    with warnings.catch_warnings():
        warnings.simplefilter("once")

        for i in xrange(30):
            print i, p._t
            p.step(dt0=.8, n_divide=20, random_v=0.02)
            sig, label = p.create_signal_label(
                (1024, 1024),
                extent = ((-2,2),(-2,2)),
                intens=100,
                poisson_noise=True,
                gaussian_noise=10,
                blur_sigma=2)

            # do simulation in Floats, bin into Int before saving (just like microscopes do!)
            sig = np.array(sig, dtype='int16')
            # label = np.array(label, dtype='int32')

            io.imsave(sigdir + "signal_%s.tif"%str(i).zfill(4), sig)
            io.imsave(labdir + "label_%s.tif"%str(i).zfill(4), label)
