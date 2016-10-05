"""


mweigert@mpi-cbg.de

"""

import numpy as np
from pointcloud2 import PointCloud2
import matplotlib.pyplot as plt
from rand_cmap import rand_cmap

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

    p = PointCloud2(rs, bound_grad_limacon, r_repell=.1, t_divide=10)

    cmap  = rand_cmap(300, type = "soft", first_color_black=True)

    # save using skimage to write grayscale (not RGB). Suppress low contrast UserWarning.
    # use separate directories for different image types, allows drag n' drop stack opening in Fiji
    import skimage.io as io
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("once")

        for i in xrange(10):
            print i, p._t
            p.step(.2, 10, random_v=0.02)
            sig, label = p.create_signal_label(
                (1024, 1024),
                #(128,128),
                extent = ((-2,2),(-2,2)),
                intens=100,
                poisson_noise=True,
                gaussian_noise=10,
                blur_sigma=3)

            # plt.subplot(1, 2, 1)
            # plt.imshow(sig)
            # plt.axis("off")
            # plt.subplot(1, 2, 2)
            # plt.imshow(label%cmap.N, cmap = cmap, vmin = 0, vmax = cmap.N-1)
            # plt.axis("off")

            # do simulation in Floats, bin into Int before saving (just like microscopes do!)
            sig = np.array(sig, dtype='int16')
            # label = np.array(label, dtype='int32')
            io.imsave("segm2_sig/signal_%s.tiff"%str(i).zfill(4), sig)
            io.imsave("segm2_lab/label_%s.tiff" % str(i).zfill(4), label)

            # io.imsave("segm2_lab/label_%s.tiff"%str(i).zfill(4), label%cmap.N, cmap=cmap, vmin = 0, vmax = cmap.N-1)
            #plt.pause(.1)
