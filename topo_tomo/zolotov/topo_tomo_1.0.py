# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Измерения кристалла алмаза для Ширяева А.А. (Sample_4)
# 1. Трубка - Mo,режим 40 Кв, 40 мА                     
# 2. Геометрия эксперимента.
#  * Угол Брэгга - 9.9165 (по факту 8.416)
#  * Угол детектора - 19.833 
#  * Углы поворота (для томографии) 0-185 с шагом 1 градус
# 3. Данные *.fit (папка exper3_1x60s_186) 1 кадр на поворот, экспозиция - 60 сек.
# 4. Темновой ток *.fit (папка exper3_1x60s_186)
#  * экспозиция - 60 сек. (снимались 5 кадров до и после эксперимента)  

# %%
import os
import glob
import numpy as np
import pylab as plt
import h5py
import astra
import cv2
from astropy.io import fits
from pprint import pprint
import warnings
import ipywidgets
from tqdm import tqdm_notebook
from ipywidgets import interact, interactive, fixed, interact_manual


# %%
plt.gray()

# %%
import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


# %%
def read_fit(fit_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(fit_path) as fit:
            tmp_data = fit[0].data.astype('uint16')
    return tmp_data


# %%
#lodad files and sort it ina natural way
data_dir = os.path.expanduser('/diskmnt/a/makov/topo_tomo/Sample_4/exper3_1x60s_186/')
all_files = glob.glob(os.path.join(data_dir,'*.FITS'))
all_files = sorted(all_files, key=natural_key)
print(len(all_files))
all_files[0:10]
# print len(data_files)
# empty_files = glob.glob(os.path.join(data_dir,'FREE*.fit'))
# empty_files = sorted(empty_files, key=natural_key)
# print len(empty_files)


# %%
data_range = np.index_exp[214:500, 832:1150]
zero_file = read_fit(all_files[190])[data_range]
plt.figure(figsize=(10,10))
plt.imshow(cv2.medianBlur(zero_file,3))
plt.colorbar(orientation='horizontal')

# %%
zero_file = read_fit(all_files[0])[data_range]
all_data = np.zeros(shape=(len(all_files), zero_file.shape[0], zero_file.shape[1]), dtype='float32')
for i in tqdm_notebook(np.arange(len(all_files))):
    all_data[i]=read_fit(all_files[i])[data_range]

# %%
free_numbers = np.hstack((range(0,5), range(190,195)))
# free_numbers = [x[0] for x in free_numbers]
data_numbers = np.arange(5,190)
print(len(free_numbers))

# %%
free_numbers

# %%
free_frame = np.median(all_data[free_numbers], axis=0)
plt.imshow(free_frame, vmin=0, vmax=100)
plt.colorbar()

# %%
clear_frames = all_data[data_numbers]-free_frame

# %%
plt.figure(figsize=(10,10))
plt.plot(clear_frames[:].mean(axis=-1).mean(axis=-1))
plt.grid()
plt.show()


# %%
def show_array(data3d, axis_numb=0):
    vmax = np.percentile(data3d, 99)
    def show_slice(i):
        plt.figure(figsize=(10,10))
        plt.imshow(local_data.take(i,axis=axis_numb),vmin=0, vmax=vmax)
        plt.colorbar(orientation='horizontal')
        plt.show()
    
    local_data = data3d
    ipywidgets.interact(show_slice, i=(0,data3d.shape[axis_numb]-1))


# %%
show_array(clear_frames)

# %%
shift = 0  # add -shift*2 to data_range [:, +:]
plt.figure(figsize=(10,10))
plt.plot(np.roll(clear_frames[0],shift,1).sum(axis=0))
plt.plot(np.roll(np.fliplr(clear_frames[180]),-shift,1).sum(axis=0))
plt.grid()

# %%
plt.figure(figsize=(10,10))
plt.imshow(np.roll(clear_frames[0],shift,1)-np.roll(np.fliplr(clear_frames[180]),-shift,1))

# %%
angles = np.arange(len(clear_frames))*1.0  # Angles staep = 2 deg   

# %%
import astra
def astra_tomo2d_parallel(sinogram, angles):
    angles = angles.astype('float64')
    detector_size = sinogram.shape[1]
    

    rec_size = detector_size
    vol_geom = astra.create_vol_geom(rec_size, rec_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_size, angles)


    sinogram_id = astra.data2d.create('-sino', proj_geom, data=sinogram)
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
#     proj_id = astra.create_projector('strip', proj_geom, vol_geom) # for CPU reconstruction only

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
#     cfg['ProjectorId'] = proj_id # for CPU reconstruction only
    cfg['option'] = {}
    cfg['option']['MinConstraint'] = 0
    # cfg['option']['MaxConstraint'] = 5

    # Available algorithms:
    # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    astra.algorithm.run(alg_id, 10)

    # Get the result
    rec = astra.data2d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.clear()
    return rec

def astra_tomo3d_parallel(sinogram, angles):
    angles = angles.astype('float64')
    detector_size = sinogram.shape[-1]
    slices_number = sinogram.shape[0]

    rec_size = detector_size
    vol_geom = astra.create_vol_geom(rec_size, rec_size, slices_number)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0,  slices_number, detector_size, angles)

    sinogram_id = astra.data3d.create('-sino', proj_geom, data=sinogram)
    
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id


    # Available algorithms:
    # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    astra.algorithm.run(alg_id, 10)

    # Get the result
    rec = astra.data3d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    astra.clear()
    return rec

def astra_topotomo3d(sinogram, angles):
    astra_angles = angles.astype('float64')
    detector_size = sinogram.shape[-1]
    slices_number = sinogram.shape[0]

    rec_size = detector_size
    vol_geom = astra.create_vol_geom(rec_size, rec_size, slices_number)
    
    # We generate the same geometry as the circular one above.
    vectors = np.zeros((len(astra_angles), 12))
    alpha = -19.833/2*np.pi/180  #  define bragg angle
    for i in range(len(astra_angles)):
        # ray direction
        vectors[i,0] = np.sin(astra_angles[i])*np.cos(alpha)
        vectors[i,1] = -np.cos(astra_angles[i])*np.cos(alpha)
        vectors[i,2] = np.sin(alpha)

        # center of detector
        vectors[i,3:6] = 0

        # vector from detector pixel (0,0) to (0,1)
        vectors[i,6] = np.cos(astra_angles[i])
        vectors[i,7] = np.sin(astra_angles[i])
        vectors[i,8] = 0;

        # vector from detector pixel (0,0) to (1,0)
        vectors[i,9] = 0
        vectors[i,10] = 0
        vectors[i,11] = 1

    # Parameters: #rows, #columns, vectors
    proj_geom = astra.create_proj_geom('parallel3d_vec', slices_number, detector_size, vectors)

    sinogram_id = astra.data3d.create('-sino', proj_geom, data=sinogram)
    
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)


    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('CGLS3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id

    # Available algorithms:
    # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    astra.algorithm.run(alg_id, 5)

    # Get the result
    rec = astra.data3d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    astra.clear()
    return rec

# %%
data = clear_frames/np.expand_dims(np.expand_dims(clear_frames.mean(axis=-1).mean(axis=-1),-1),-1)
data[data<0] = 0

# %%
show_array(data,1)

# %%
s1 = np.require(data[:,:,:].swapaxes(0,1),dtype=np.float32, requirements=['C'])

# %%
# rec_3d = astra_topotomo3d(np.log(1.+ s1) , angles*np.pi/180)
rec_3d = astra_topotomo3d(s1, angles*np.pi/180)

# %%
show_array(rec_3d,2)

# %%
import ipyvolume as ipv

# %%
ipv.figure()
ipv.volshow(rec_3d)
ipv.save('s4.html')
ipv.show()

# %%
# sl = rec_3d[rec_3d.shape[0]/3]
# mask = np.zeros_like(sl)
# X,Y = meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
# X = X-mask.shape[0]/2
# Y = Y-mask.shape[1]/2
# mask = (X*X+Y*Y)**0.5<(mask.shape[0]/2)-3
# rec_3d_masked = rec_3d*mask 

# %%
# show_array(rec_3d_masked)

# %%
data_file_parts = os.path.join(data_dir,'result.h5')
out_file = '{}_reconstruction.h5'.format(''.join(data_file_parts[:-1]))

out_file = os.path.join(data_dir,'result.h5')
print(out_file)
with h5py.File(out_file,'w') as h5f:
    h5f.create_dataset('Result', data=rec_3d, dtype=np.float32)


# %%
def save_amira(result_file):
    """
    Функция сохраняет реконструированные слои в формате Amira raw file

    Inputs:
        data_path - путь к директории, где находиться файл res_tomo.hdf5 в формате HDF5
            в этом файде должен быть раздел (node) /Results в котором в виде 3D массива
            записаны реконструированный объём
    Outputs:
        Файлы amira.raw и tomo.hx. Если файлы уже существуют, то они перезаписываются.
        Тип данных: float32 little endian
    """
    data_path = os.path.dirname(result_file)
    with open(os.path.join(data_path, 'amira.raw'), 'wb') as amira_file:
        with h5py.File(result_file, 'r') as h5f:
            x = h5f['Result']
            for i in range(x.shape[0]):
                np.array(x[i, :, :]).tofile(amira_file)

            file_shape = x.shape

            with open(os.path.join(data_path, 'tomo.hx'), 'w') as af:
                af.write('# Amira Script\n')
                af.write('remove -all\n')
                af.write(r'[ load -raw ${SCRIPTDIR}/amira.raw little xfastest float 1 '+
                         str(file_shape[1])+' '+str(file_shape[2])+' '+str(file_shape[0])+
                         ' 0 '+str(file_shape[1]-1)+' 0 '+str(file_shape[2]-1)+' 0 '+str(file_shape[0]-1)+
                         ' ] setLabel tomo.raw\n')


# %%
save_amira(out_file)

# %%
