import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import astra

import alg 
import RegVarSIRT
import vrecon

def STD(x):
  return 1.1e-7 * x * x + 165

def load_data(path_sino):
  sinogram = plt.imread(path_sino)
  if len(sinogram.shape) == 3:
    sinogram = sinogram[...,0]
  sinogram = np.flipud(sinogram)
  sinogram = sinogram.astype('float32')
  
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

def recontruct(proj_geom, vol_geom, sinogram):
  rec_fbp = alg.gpu_fbp(proj_geom, vol_geom, sinogram)
  k = sinogram.shape[0] / (proj_geom['DetectorWidth']**2) / (np.pi / 2.0)
  rec_fbp *= k
  rec_sirt = alg.gpu_sirt(proj_geom, vol_geom, sinogram, 10, np.copy(rec_fbp))
  return rec_fbp, rec_sirt

def rec_VSIRT(proj_geom, vol_geom, sinogram, rec_fbp):
  V = sinogram.copy()
  V = STD(V)

  fig = plt.figure()
  a=fig.add_subplot(1,1,1)
  imgplot = plt.imshow(V, interpolation=None, cmap="gray")
  plt.colorbar(orientation='horizontal');
  plt.savefig("STD.png")
  plt.close(fig)

  V = 1.0 / V

  proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
  W = astra.OpTomo(proj_id)
  eps = 1e-30

  x0 = np.copy(rec_fbp)
  reg_parameters =  {'total': 0.001, 'exp': 0.2, 'l1_grad': 0.5, 'l2': 0.3}
  rec_vsirt = vrecon.VCG(W, sinogram, x0, V, reg_parameters, 100)
  en_1 = rec_vsirt['energy']
  al_1 = rec_vsirt['alpha']
  rec_vsirt = rec_vsirt['rec']

  en_1 = np.asarray(en_1)
  x = np.arange(0, len(en_1))
  fig = plt.figure()
  plt.plot(x, en_1, label="energy", linewidth=3.0)
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Number of iterations, a.u.')
  plt.legend(loc='best');
  plt.savefig("conv.png")

  al_1 = np.asarray(al_1)
  x = np.arange(0, len(al_1))
  fig = plt.figure() 
  plt.plot(x, al_1, label="energy", linewidth=3.0)
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Number of iterations, a.u.')
  plt.legend(loc='best');
  plt.savefig("alpha.png")
  return rec_vsirt

def plot_result(rec_fbp, rec_sirt, rec_vsirt):
  rec_fbp = np.flipud(rec_fbp)
  rec_sirt = np.flipud(rec_sirt)
  rec_vsirt = np.flipud(rec_vsirt)

  miv = rec_fbp.min()
  if (miv > rec_sirt.min()):
    miv = rec_sirt.min()
  if (miv > rec_vsirt.min()):
    miv = rec_vsirt.min()

  mav = rec_fbp.max()
  if (mav < rec_sirt.max()):
    mav = rec_sirt.max()
  if (mav < rec_vsirt.max()):
    mav = rec_vsirt.max()

  print 'FBP: ', rec_fbp.min(), rec_fbp.max()
  print 'SIRT: ', rec_sirt.min(), rec_sirt.max()
  print 'VCG: ', rec_vsirt.min(), rec_vsirt.max()

  print 'value: ', miv, mav 
  
  plt.imsave('FBP.png', rec_fbp, cmap="gray", vmin=miv, vmax=mav)
  plt.imsave('SIRT.png', rec_sirt, cmap="gray", vmin=miv, vmax=mav)
  plt.imsave('VCG.png', rec_vsirt, cmap="gray", vmin=miv, vmax=mav)

  f1 = plt.figure()
  a=f1.add_subplot(1,1,1)
  cax = plt.imshow(rec_fbp, interpolation=None, cmap="gray", vmin=miv, vmax=mav)
  plt.colorbar(cax, orientation ='horizontal', ticks=[miv, (mav + miv) / 2 , mav])
  plt.savefig("FBP_f.png")
  plt.close(f1)

  f2 = plt.figure()
  a=f2.add_subplot(1,1,1)
  cax = plt.imshow(rec_sirt, cmap=plt.get_cmap('gray'), interpolation=None, vmin=miv, vmax=mav)
  plt.colorbar(cax, ticks=[miv, (mav + miv) / 2 , mav], orientation ='horizontal')
  plt.savefig("SIRT_f.png")
  plt.close(f2)

  f3 = plt.figure()
  a=f3.add_subplot(1,1,1)
  cax = plt.imshow(rec_vsirt, cmap=plt.get_cmap('gray'), interpolation=None, vmin=miv, vmax=mav)
  plt.colorbar(cax, ticks=[miv, (mav + miv) / 2 , mav], orientation ='horizontal');
  plt.savefig("VCG_f.png")
  plt.close(f3)
  
  s_x = 500
  s_y = 500
  e_x = 650
  e_y = 650
  fig = plt.figure(figsize=(10,4))
  a=fig.add_subplot(1,3,1)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VCG')
  plt.colorbar(orientation='horizontal');
  a=fig.add_subplot(1,3,2)
  imgplot = plt.imshow(rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('SIRT')
  plt.colorbar(orientation ='horizontal');
  a=fig.add_subplot(1,3,3)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y] - rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VCG - SIRT')
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
  a.set_title('VCG')
  plt.colorbar(orientation='horizontal');
  a=f1.add_subplot(1,3,2)
  imgplot = plt.imshow(rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('SIRT')
  plt.colorbar(orientation ='horizontal');
  a=f1.add_subplot(1,3,3)
  imgplot = plt.imshow(rec_vsirt[s_x:e_x, s_y:e_y] - rec_sirt[s_x:e_x, s_y:e_y], interpolation=None, cmap="gray")
  a.set_title('VCG - SIRT')
  plt.colorbar(orientation ='horizontal');
  plt.savefig("diff_2.png")
  plt.close(f1)

  num_str = 700
  s = 400
  e = 600
  f2 = plt.figure(figsize=(10,7))    
  plt.plot(rec_sirt[num_str, s:e] / np.mean(rec_sirt[num_str, :]), label="SIRT", linewidth=1.0)
  plt.plot(rec_vsirt[num_str, s:e] / np.mean(rec_vsirt[num_str, :]), label="VCG", linewidth=1.0)
  plt.xticks(np.arange(0, 225, 25), np.arange(s, e+25, 25))
  plt.grid(True)
  plt.ylabel('Value, a.u.')
  plt.xlabel('Pixel number, a.u.')
  plt.legend(loc='best');
  plt.savefig("sr_1.png")
  plt.close(f2)

  f3 = plt.figure(figsize=(10,7))    
  plt.plot(rec_fbp[num_str, s:e] / np.mean(rec_fbp[num_str, :]), label="FBP", linewidth=1.0)
  plt.plot(rec_vsirt[num_str, s:e] / np.mean(rec_vsirt[num_str, :]), label="VCG", linewidth=1.0)
  plt.grid(True)
  plt.xticks(np.arange(0, 225, 25), np.arange(s, e+25, 25))
  plt.ylabel('Value, a.u.')
  plt.xlabel('Pixel number, a.u.')
  plt.legend(loc='best');
  plt.savefig("sr_2.png")
  plt.close(f3)

def main():
  print "Program is started"
  path_sino = 'D:/_svn_tomo/trunk/prj.scripts/VarSIRT/S1S2S3_2.74um__sino0245.tif'
  proj_geom, vol_geom, sinogram = load_data(path_sino)
  rec_fbp, rec_sirt = recontruct(proj_geom, vol_geom, sinogram)
  rec_vsirt = rec_VSIRT(proj_geom, vol_geom, sinogram, rec_fbp)
  plot_result(rec_fbp, rec_sirt, rec_vsirt)
  print "Completed successfully"

main()