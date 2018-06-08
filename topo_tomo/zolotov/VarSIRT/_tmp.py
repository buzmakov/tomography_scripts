import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import astra
import minimg

import alg 
import RegVarSIRT
import vrecon_vsirt

def load_data(path_sino, path_std, path_x_0):
  img = minimg.MinImg.load(path_sino)
  sinogram = img.asarray()[0]
  sinogram = sinogram.astype('float32')
  print 'sino: ', sinogram.min(), sinogram.max()
  print 'sino: ', sinogram.shape[0], sinogram.shape[1]

  img = minimg.MinImg.load(path_std)
  std = img.asarray()[0]
  std = std.astype('float32')
  print 'std: ', std.min(), std.max()

  img = minimg.MinImg.load(path_x_0)
  x_0 = img.asarray()[0]
  x_0 = np.flipud(x_0)
  x_0 = x_0.astype('float32')
  plt.imsave('2.png', x_0, cmap='gray')
  print 'vol: ', x_0.min(), x_0.max()

  f = plt.figure()
  a=f.add_subplot(1,1,1)
  imgplot = plt.imshow(x_0, interpolation=None, cmap="gray")
  plt.colorbar(orientation='horizontal');
  plt.savefig("vol.png")
  plt.close(f)

  fig = plt.figure()
  a=fig.add_subplot(1,1,1)
  imgplot = plt.imshow(sinogram, interpolation=None, cmap="gray")
  plt.colorbar(orientation='horizontal');
  plt.savefig("sinogram.png")
  plt.close(fig)

  detector_cell = sinogram.shape[1]
  n_angles = sinogram.shape[0] 

  '''
  Image Pixel Size (um)=11.000435
  Object to Source (mm)=54.350
  Camera to Source (mm)=225.315
  '''
  pixel_size = 11.000435e-3
  os_distance = 54.350 / pixel_size
  ds_distance = 225.315 / pixel_size

  angles = np.arange(n_angles) * 0.4
  angles = angles.astype('float32') / 180.0 * np.pi
  angles = angles - (angles.max() + angles.min()) / 2
  angles = angles + np.pi / 2

  vol_geom = astra.create_vol_geom(detector_cell, detector_cell)
  proj_geom = astra.create_proj_geom('fanflat', ds_distance / os_distance, detector_cell, angles,
                                    os_distance, (ds_distance - os_distance))
  return proj_geom, vol_geom, sinogram, std, x_0

def rec_VSIRT(proj_geom, vol_geom, sinogram, std, vol):
  proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
  W = astra.OpTomo(proj_id)
  eps = 1e-30

  x0 = np.copy(vol)
  reg_parameters =  {'total': 0.001, 'exp': 0.2, 'l1_grad': 0.5, 'l2': 0.3}
  rec_vsirt = vrecon_vsirt.iterate(W, sinogram, x0, std, None, 15)
  #rec_vsirt = RegVarSIRT.run(W, sinogram, x0, std, 0.0, eps, 2, 'steepest')
  en = rec_vsirt['energy']
  al = rec_vsirt['alpha']
  rec_vsirt = rec_vsirt['rec']

  en = np.asarray(en)
  print en
  x = np.arange(0, len(en))
  fig = plt.figure() 
  plt.semilogy(x, en, label="energy", linewidth=2.0)
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Number of iterations, a.u.')
  plt.legend(loc='best');
  plt.savefig("convergence.png")
  plt.close(fig)

  al = np.asarray(al)
  print al
  x = np.arange(0, len(al))
  fig = plt.figure() 
  plt.plot(x, al, label="alpha", linewidth=2.0)
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Number of iterations, a.u.')
  plt.legend(loc='best');
  plt.savefig("alpha.png")

  return rec_vsirt

def plot_result(rec_vsirt):
  print 'VarSIRT: ', rec_vsirt.min(), rec_vsirt.max()
  plt.imsave('VarSIRT.png', rec_vsirt, cmap="gray")

  f1 = plt.figure()
  a=f1.add_subplot(1,1,1)
  cax = plt.imshow(rec_vsirt, interpolation=None, cmap="gray")
  plt.colorbar(cax, orientation ='horizontal')
  plt.savefig("vsirt.png")
  plt.close(f1)

def main():
  print "Program is started"
  path_sino = 'D:/_svn_tomo/trunk/build/prj.core/vrecon/sino.tif'
  path_std = 'D:/_svn_tomo/trunk/build/prj.core/vrecon/std.tif'
  path_vol = 'D:/_svn_tomo/trunk/build/prj.core/vrecon/vol.tif'
  proj_geom, vol_geom, sinogram, std, vol = load_data(path_sino, path_std, path_vol)
  rec_vsirt = rec_VSIRT(proj_geom, vol_geom, sinogram, std, vol)
  plot_result(rec_vsirt)
  print "Completed successfully"

main()