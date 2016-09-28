"""


mweigert@mpi-cbg.de

"""

import numpy as np
from pointcloud3 import PointCloud3
import matplotlib.pyplot as plt
from rand_cmap import rand_cmap
from spimagine import volfig, volshow

if __name__=='__main__':
    from time import time

    np.random.seed(0)
    rs = np.random.uniform(-1., 1., (40, 3))

    N = 5
    phi = np.linspace(0, 2*np.pi, N+1)[:N]
    t = np.arccos(np.linspace(-1, 1, N+2)[1:-1])
    P, T = np.meshgrid(phi, t, indexing="ij")
    phi, t = P.flatten(), T.flatten()

    rs = .8*np.stack([np.cos(phi)*np.sin(t), np.sin(phi)*np.sin(t), np.cos(t)]).T
    rs += .2*np.random.uniform(-1,1,rs.shape)


    # rs = [[0,0,0],[0,0,.4],[0,0,-.4],[.4,0,0],[-.4,0,0]]

    def bound_grad_sphere(rs, t):
        r0 = .8
        u = np.linalg.norm(rs, axis=-1)-r0

        # the gradient
        n = 1.*rs.T
        n /= 1.e-10+np.linalg.norm(n, axis=0)
        return -1*(n*u).T


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


    p = PointCloud3(rs, bound_grad_sphere, r_repell=.15, t_divide=18)

    #cmap = rand_cmap(300, type = "soft" , first_color_black=True)
    cmap = rand_cmap(200, type = "soft", first_color_black=True)

    # from time import time
    # t = time()
    # sig, label = p.create_signal_label(
    #         (256,)*3,
    #         extent=((-1.3, 1.3),)*3,
    #         intens=100,
    #         poisson_noise=True,
    #         gaussian_noise=10,
    #         blur_sigma=0)
    #
    # print time()-t

    w1, w2 = None, None

    for i in xrange(2):
        print i, p._t
        p.step(.4, 20, random_v=0.02)

        sig, label = p.create_signal_label(
            (256,)*3,
            extent=((-1.3, 1.3),)*3,
            intens=100,
            poisson_noise=True,
            gaussian_noise=0,
            blur_sigma=1)


        if w1 is None:
            w1 = volfig(1,raise_window = False)
            w1 = volshow(sig,autoscale = False,raise_window = False)
            w1.set_colormap("grays")
            w1.transform.setBox(False)
            w1.transform.setZoom(1.3)
        else:
            w1.glWidget.renderer.update_data(sig.astype(np.float32))
            w1.glWidget.refresh()

        w1.transform.setRotation(0.01*i,0,1,0)
        w1.saveFrame("segm3/signal_%s.png"%str(i).zfill(4))

        if w2 is None:
            w2 = volfig(2,raise_window = False)
            w2 = volshow((label%cmap.N).astype(np.float32), autoscale = False, raise_window = False)
            w2.transform.setMax(cmap.N-.5)
            w2.transform.setMin(0)
            w2.glWidget._set_colormap_array(cmap(np.arange(cmap.N))[:,:3])
            w2.transform.setBox(False)
            w2.transform.setZoom(1.3)
        else:
            w2.glWidget.renderer.update_data((label%cmap.N).astype(np.float32))
            #w2.glWidget._set_colormap_array(cmap(np.arange(cmap.N))[:,:3])
            w2.glWidget.refresh()


        w2.transform.setRotation(0.01*i,0,1,0)
        w2.saveFrame("segm3/label_%s.png"%str(i).zfill(4))