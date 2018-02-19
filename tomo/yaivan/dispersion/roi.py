import numpy as np 
import pylab as plt 

import scipy

import astra 
import tomopy


import alg 
import sirt 
import sirt_noise

@profile
def main():
    size = 128
    phantom = tomopy.misc.phantom.shepp2d(size=size).astype('float32')
    phantom = np.squeeze(phantom)/np.max(phantom)

    # make sinogram
    n_angles = 180
    angles = np.arange(0.0, 180.0,  180.0 / n_angles)
    angles = angles.astype('float32') / 180 * np.pi

    pg = astra.create_proj_geom('parallel', 1.0, size, angles)
    vg = astra.create_vol_geom((size, size))
    sino = alg.gpu_fp(pg, vg, phantom)
    sino = sino.astype('float32')
    
    D = np.ones_like(sino)
    
    
    mask = np.zeros_like(sino)
    mask[:,30:-30] = 1
    sino[mask == 0] = 0
    D[mask==0] = 0
    
    proj_id = astra.create_projector('cuda', pg, vg)
    W = astra.OpTomo(proj_id)
    x0 = np.zeros_like(phantom)
    eps = 1e-30

    x0 = np.zeros_like(phantom)
    #x0 = rec_1.copy()
    rec = sirt_noise.run_gpu(W, sino, D, x0, eps, 2, 'steepest')
    en_0 = rec['energy'] 
    alpha_0 = rec['alpha']
    rec_0 = rec['rec']

    astra.projector.delete(proj_id)
    

if __name__ == "__main__":
    main()