import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import astra

import alg 
import VarSIRT

def STD(x):
  return 1.1e-7 * x * x + 165

def load_data(path_sino):
  sinogram = plt.imread(path_sino)
  if len(sinogram.shape) == 3:
    sinogram = sinogram[...,0]
  sinogram = np.flipud(sinogram)
  sinogram = sinogram.astype('float32')
  sinogram = sinogram[0:449, :]
  
  fig = plt.figure(figsize=(20,20))
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
  os_distance = 54. / pixel_size
  ds_distance = 225.315 / pixel_size

  angles = np.arange(n_angles) * 0.4
  angles = angles.astype('float32') / 180.0 * np.pi
  angles = angles - (angles.max() + angles.min()) / 2
  angles = angles + np.pi / 2

  vol_geom = astra.create_vol_geom(detector_cell, detector_cell)
  proj_geom = astra.create_proj_geom('fanflat', ds_distance / os_distance, detector_cell, angles,
                                    os_distance, (ds_distance - os_distance))
  return proj_geom, vol_geom, sinogram

def recontructr(proj_geom, vol_geom, sinogram):
  rec_fbp = alg.gpu_fbp(proj_geom, vol_geom, sinogram)
  rec_sirt = alg.gpu_sirt(proj_geom, vol_geom, sinogram, 500)
  return rec_fbp, rec_sirt

def rec_VSIRT(proj_geom, vol_geom, sinogram, rec_fbp):
  V = sinogram.copy()
  V = 1.0 / STD(V)

  fig = plt.figure(figsize=(20,20))
  a=fig.add_subplot(1,1,1)
  imgplot = plt.imshow(V, interpolation=None, cmap="gray")
  plt.colorbar(orientation='horizontal');
  plt.savefig("STD.png")
  plt.close(fig)

  sino_new = sinogram * V
  proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
  W = astra.OpTomo(proj_id)
  eps = 1e-30

  x0 = np.copy(rec_fbp)
  rec_vsirt = VarSIRT.run(W, sino_new, V, x0, eps, 100, 'steepest')
  en_1 = rec_vsirt['energy']
  rec_vsirt = rec_vsirt['rec']

  en_1 = np.asarray(en_1)
  x = np.arange(0, len(en_1))
  fig = plt.figure() 
  plt.semilogy(x, en_1, label="energy", linewidth=3.0)
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Number of iterations, a.u.')
  plt.legend(loc='best');
  plt.savefig("conv.png")
  return rec_vsirt

def plot_result(rec_fbp, rec_sirt, rec_vsirt):
  rec_fbp = np.flipud(rec_fbp)
  rec_sirt = np.flipud(rec_sirt)
  rec_vsirt = np.flipud(rec_vsirt)
  
  plt.imsave('FBP.png', rec_fbp, cmap="gray")
  plt.imsave('SIRT.png', rec_sirt, cmap="gray")
  plt.imsave('VarSIRT.png', rec_vsirt, cmap="gray")
  
  s_x = 500
  s_y = 500
  e_x = 650
  e_y = 650
  fig = plt.figure(figsize=(10,4))
  a=fig.add_subplot(1,3,1)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VarSIRT')
  plt.colorbar(orientation='horizontal');
  a=fig.add_subplot(1,3,2)
  imgplot = plt.imshow(rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('SIRT')
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(1,3,3)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y] - rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VarSIRT - SIRT')
  plt.colorbar(orientation ='horizontal');
  plt.savefig("diff_1.png")
  plt.close()


  s_x = 620
  s_y = 560
  e_x = 650
  e_y = 600
  f1 = plt.figure(figsize=(10,4))
  a=f1.add_subplot(1,3,1)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VarSIRT')
  plt.colorbar(orientation='horizontal');
  a=f1.add_subplot(1,3,2)
  imgplot = plt.imshow(rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('SIRT')
  plt.colorbar(orientation ='horizontal');
  a=f1.add_subplot(1,3,3)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y] - rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VarSIRT - SIRT')
  plt.colorbar(orientation ='horizontal');
  plt.savefig("diff_2.png")
  plt.close(f1)

  num_str = 700
  s = 400
  e = 600
  f2 = plt.figure(figsize=(10,7))    
  plt.plot(rec_sirt[num_str, s:e] / np.mean(rec_sirt[num_str, :]), label="SIRT", linewidth=1.0)
  plt.plot(rec_vsirt[num_str, s:e] / np.mean(rec_vsirt[num_str, :]), label="VarSIRT", linewidth=1.0)
  plt.xticks(np.arange(0, 225, 25), np.arange(s, e+25, 25))
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Pixel number, a.u.')
  plt.legend(loc='best');
  plt.savefig("sr_1.png")
  plt.close(f2)

  f3 = plt.figure(figsize=(10,7))    
  plt.plot(rec_fbp[num_str, s:e] / np.mean(rec_fbp[num_str, :]), label="FBP", linewidth=1.0)
  plt.plot(rec_vsirt[num_str, s:e] / np.mean(rec_vsirt[num_str, :]), label="VarSIRT", linewidth=1.0)
  plt.grid(True)
  plt.xticks(np.arange(0, 225, 25), np.arange(s, e+25, 25))
  plt.ylabel('Value, a.u.')
  plt.xlabel('Pixel number, a.u.')
  plt.legend(loc='best');
  plt.savefig("sr_2.png")
  plt.close(f3)

def main():
  print "Program is started"
  path_sino = '/home/ingacheva/_Schlumberger/S1S2S3_2.74um__sino0245.tif'
  proj_geom, vol_geom, sinogram = load_data(path_sino)
  rec_fbp, rec_sirt = recontructr(proj_geom, vol_geom, sinogram)
  rec_vsirt = rec_VSIRT(proj_geom, vol_geom, sinogram, rec_fbp)
  plot_result(rec_fbp, rec_sirt, rec_vsirt)
  print "Completed successfully"

main()