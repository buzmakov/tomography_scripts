import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astra

import alg 
import VarSIRT

def err_l2(img, rec):
  return np.sum((img - rec) ** 2) / (rec.shape[0] * rec.shape[1])

def mean_value(lam, num = 100):
  m_v = 0.0
  factor = 1.0
  for x in np.arange(1, num + 1, 1):
    factor = factor * lam / x
    m_v -= np.log(x) * factor
  m_v = m_v * np.exp(-lam)
  return m_v
  m_v = 0.0

def var_value(lam, M, num = 100):
  d_v = 0.0
  factor = 1.0
  for x in np.arange(1, num + 1, 1):
    factor = factor * lam / x
    d_v += (- np.log(x) - M) ** 2 * factor
  d_v = d_v * np.exp(-lam)
  return d_v

def made_data():
  # make phantom
  size = 50
  mu1 = 0.006
  mu2 = 0.005
  mu3 = 0.004
  phantom = np.zeros((size, size))
  half_s = size / 2

  y, x = np.meshgrid(range(size), range(size))
  xx = (x - half_s).astype('float32')
  yy = (y - half_s).astype('float32')
    
  mask_ell1 = pow(xx + 0.1 * size, 2) / np.power(0.35 * size, 2) + pow(yy, 2) / np.power(0.15 * size, 2) <= 1
  mask_ell2 = pow(xx - 0.15 * size, 2) / np.power(0.3 * size, 2) + pow(yy - 0.15 * size, 2) / np.power(0.15 * size, 2) <= 1 
  phantom[mask_ell1] = mu1
  phantom[mask_ell2] = mu2
  phantom[np.logical_and(mask_ell1, mask_ell2)] = mu3
  phantom[int(0.15 * size):int(0.35 * size), int(0.2 * size):int(0.5 * size)] = mu3 
  phantom = 1e+1 * phantom

  # make sinogram
  n_angles = 90.0
  angles = np.arange(0.0, 180.0,  180.0 / n_angles)
  angles = angles.astype('float32') / 180 * np.pi

  pg = astra.create_proj_geom('parallel', 1.0, size, angles)
  vg = astra.create_vol_geom((size, size))
  sino = alg.gpu_fp(pg, vg, phantom)

  i0 = 2e+2
  sino = i0 * np.exp(-sino)
  sino = sino.astype('float64')

  return phantom, pg, vg, sino, i0

def calkculate_STD(sino):
  M = np.zeros_like(sino)
  D = np.zeros_like(sino)
  for i in np.arange(0, sino.shape[0]):
    for j in np.arange(0, sino.shape[1]):
      M[i,j] = mean_value(sino[i,j], num = 1000)
      D[i,j] = var_value(sino[i,j], M[i,j], num = 1000)
  
  Div = D.copy()
  Div = np.sqrt(D)
  Div = 1.0 / Div
  return Div, D

def prepera_data_to_ceronctruct(sino, Div, i0):
  sino_noise = np.random.poisson(lam=(sino)).astype('float32')
  sino_noise[sino_noise > i0] = i0
  sino_noise = np.log(i0) - np.log(sino_noise)
  sino_new = sino_noise * Div
  return sino_new, sino_noise

def reconstruct(proj_geom, vol_geom, sinogram):
  rec_sirt = alg.gpu_sirt(proj_geom, vol_geom, sinogram, 300)
  plt.imsave('SIRT.png', rec_sirt, cmap="gray")
  return rec_sirt

def rec_VSIRT(phantom, pg, vg, sino_new, Div):
  proj_id = astra.create_projector('cuda', pg, vg)
  W = astra.OpTomo(proj_id)
  eps = 1e-30
  x0 = np.zeros_like(phantom)
  rec_1 = VarSIRT.run(W, sino_new, Div, x0, eps, 100, 'steepest')
  rec_1 = rec_1['rec']

  astra.projector.delete(proj_id)
  plt.imsave('VarSIRT.png', rec_1, cmap="gray")
  return rec_1

def plot_result(phantom, sino_new, sino_noise, D, rec_sirt, rec_vsirt):
  er_1 = err_l2(phantom, rec_sirt)
  er_2 = err_l2(phantom, rec_vsirt)

  fig = plt.figure(figsize=(10,10))
  a=fig.add_subplot(2,3,1)
  imgplot = plt.imshow(sino_noise, interpolation=None, cmap="gray")
  a.set_title('Noisy sinogram')
  plt.colorbar(orientation='horizontal');
  a=fig.add_subplot(2,3,2)
  imgplot = plt.imshow(D, interpolation=None, cmap="gray")
  a.set_title('Variance')
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(2,3,3)
  imgplot = plt.imshow(sino_new, interpolation=None, cmap="gray")
  a.set_title('Noisy sinogram / standard deviation')
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(2,3,4)
  imgplot = plt.imshow(phantom, interpolation=None, cmap="gray")
  a.set_title('Phantom')
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(2,3,5)
  imgplot = plt.imshow(rec_sirt, interpolation=None, cmap="gray")
  a.set_title('SIRT, Err=' + str('{:0.2e}'.format(er_1)))
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(2,3,6)
  imgplot = plt.imshow(rec_vsirt, interpolation=None, cmap="gray")
  a.set_title('VarSIRT, Err=' + str('{:0.2e}'.format(er_2)))
  plt.colorbar(orientation ='horizontal');
  plt.savefig("ex_4.png")

def main():
  print "Program is started"
  phantom, pg, vg, sino, i0 = made_data()
  V, D = calkculate_STD(sino)
  sino_new, sino_noise = prepera_data_to_ceronctruct(sino, V, i0)
  rec_sirt = reconstruct(pg, vg, sino_noise)
  rec_vsirt = rec_VSIRT(phantom, pg, vg, sino_new, V)
  plot_result(phantom, sino_new, sino_noise, D, rec_sirt, rec_vsirt)
  print "Completed successfully"

main()
