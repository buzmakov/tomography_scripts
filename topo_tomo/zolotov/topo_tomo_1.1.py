# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%
import re
from pathlib import Path
import warnings
from astropy.io import fits
import numpy as np
import pylab as plt
import cv2
from tqdm import tqdm_notebook
from tomo.recon.astra_utils import astra_recon_3d_parallel_vec, astra_fp_3d_parallel_vec
# import os
# import glob

# 
# import h5py
# import astra
# 
# 
# from pprint import pprint
# 
import ipywidgets
from tqdm import tqdm_notebook
from ipywidgets import interact, interactive, fixed, interact_manual

# from pprint import pprint

from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper
from bokeh.layouts import gridplot
from bokeh.palettes import brewer
output_notebook()


# %%
import dask
from dask.distributed import Client, progress
client = Client(threads_per_worker=4, n_workers=16)
client


# %%
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


# %%
#data reader functions
def read_fit(fit_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(fit_path) as fit:
            tmp_data = fit[0].data.astype('uint16')
    return tmp_data

def read_tif(tif_path):
    return plt.imread(tif_path)

def read_txt(txt_path):
    return np.loadtxt(txt_path)


# %%
config = {'data_dir': Path('/home/zolotovden/topo-tomo/Si/2019_10_31/pre-process'),
          'data_type': 'tif',
          'readme_file': '../experiment_2019-10-31.txt',
          'data_template': 'f_Si7_poly*.tif',
          'need_rot_90': 1,
          'roi':{
              'x': [0, -50],
              'y': [0, -1]
          },
          'shift': -10,
          'ang_step': 1,
          'bragg_ang': 10.644
         }

# config = {'data_dir': Path('/nfs/synology-tomodata/external_data/topo-tomo/ESRF/2018/'+
#                            '__TOPODATA_ESRF_data_filtered/filtered_data_window_3pixels/'),
#           'data_type': 'txt',
#           'readme_file': '../../Описание эксперимента.txt',
#           'data_template': 'Imported Stack*_3.ham',
#           'need_rot_90': 0,
#           'roi':{
#               'x': [800, 1850],
#               'y': [780, 1350]
#           },
#           'shift': 20,
#           'da': 3.6
#          }

# %%
# !ls {config['data_dir']}

# %%
data_dir  = config['data_dir']
readme_txt = list(data_dir.glob(config['readme_file']))[0] if config['readme_file'] is not None else None
data_type = config['data_type']
data_template = config['data_template']
need_rot_90 = config['need_rot_90']
xmin, xmax = config['roi']['x']
ymin, ymax = config['roi']['y']
shift = config['shift']
ang_step = config['ang_step']
bragg_ang = config['bragg_ang']


if config['data_type'] == 'tif':
    read_data_ptr = read_tif
elif config['data_type'] == 'txt':
    read_data_ptr = read_txt
else:
    raise ValueError('Unknown data type')

def read_data(f_name):
    t = read_data_ptr(f_name)
    return np.rot90(t, need_rot_90).astype('float32')


# %%
#try find and print text files
if readme_txt is not None:
    with open(readme_txt,'r', encoding="windows-1251") as tff:
        for l in tff.readlines():
            print(l.replace('\n',''))
    print('\n')

#load files and sort it ina natural way     
data_files = [str(f) for f in data_dir.glob(data_template)]
data_files = sorted(data_files, key=natural_key)
print(f'Found {len(data_files)} DATA files')


# %%
# xmin, xmax = 800, 1850
# ymin, ymax = 780, 1350

# xmin, xmax = 0, -1
# ymin, ymax = 0, -1
#rotation axis should be vertical
data_range = np.index_exp[ymin:ymax, xmin:xmax]

plt.figure(figsize=(15,15))
for i in range(4):
    plt.subplot(221+i)
    tmp_file = read_data(data_files[i*len(data_files)//4])[data_range]
    filt_img = tmp_file #cv2.medianBlur(tmp_file, 3)
    plt.imshow(filt_img, vmin=np.percentile(filt_img[::10],1), vmax=np.percentile(filt_img[::10],99))
    plt.colorbar()


# %%
def safe_median(data):
    m_data = cv2.medianBlur(data,3)
    mask = np.abs(m_data-data) > 0.05*np.abs(data)
    res = data.copy()
    res[mask] = m_data[mask]
    return res


# %%
lazy_data=[]
for d in data_files:
    lazy_data.append(dask.delayed(read_data)(d)[data_range])
futures = dask.persist(*lazy_data)

# client.cluster.scale(4)
data_data = dask.compute(*futures)
tomo_data = np.asarray(data_data, dtype='float32')


# %%
def show_array(data3d, axis_numb=0):
    vmax = np.percentile(data3d, 99.99)
    vmin = np.percentile(data3d, 5)
    
    if axis_numb==0:
        im_shape_x = data3d.shape[1]
        im_shape_y = data3d.shape[2]
    elif axis_numb==1:
        im_shape_x = data3d.shape[0]
        im_shape_y = data3d.shape[2]
    else:
        im_shape_x = data3d.shape[0]
        im_shape_y = data3d.shape[1]
        
    color_mapper = LinearColorMapper(palette="Viridis256", low=vmin, high=vmax)
    p_ph = figure(title='phantom', plot_height=im_shape_x, plot_width=im_shape_y)
    ph_image = p_ph.image(image=[np.zeros((im_shape_x,im_shape_y))], color_mapper=color_mapper,
                          x=0, y=0, dw=im_shape_y, dh=im_shape_x)
    
    target = show(p_ph, notebook_handle=True)
    def show_slice(i):
        ph_image.data_source.data = {'image': [local_data.take(i,axis=axis_numb)[::1,::1]]}
        push_notebook(handle=target)
    
    local_data = data3d
    ipywidgets.interact(show_slice, i=(0, data3d.shape[axis_numb]-1))


# %%
show_array(tomo_data,0)

# %%
show_array(tomo_data,1)


# %%
def cv_rotate(x, angle):
    """
    Rotate square array using OpenCV2 around center of the array
    :param x: 2d numpy array
    :param angle: angle in degrees
    :return: rotated array
    """
    x_center = tuple(
        np.array((x.shape[1], x.shape[0]), dtype='float32') / 2.0 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(x_center, angle, 1.0)
    xro = cv2.warpAffine(
        x, rot_mat, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR)
    return xro


# %%
def diff_array(data3d, axis_numb=0):
    vmax = np.percentile(data3d, 99.99)
    vmin = np.percentile(data3d, 5)
    
    if axis_numb==0:
        im_shape_x = data3d.shape[1]
        im_shape_y = data3d.shape[2]
    elif axis_numb==1:
        im_shape_x = data3d.shape[0]
        im_shape_y = data3d.shape[2]
    else:
        im_shape_x = data3d.shape[0]
        im_shape_y = data3d.shape[1]
        
    color_mapper = LinearColorMapper(palette="Viridis256", low=vmin, high=vmax)
    p_ph = figure(title='phantom', plot_height=im_shape_x, plot_width=im_shape_y)
    ph_image = p_ph.image(image=[np.zeros((im_shape_x,im_shape_y))], color_mapper=color_mapper,
                          x=0, y=0, dw=im_shape_y, dh=im_shape_x)
    
    target = show(p_ph, notebook_handle=True)
    def show_slice(i, di, da, shift):
        im0 = local_data.take(i, axis=axis_numb)
        im0 = np.roll(im0, shift, 1)
        im0 = cv_rotate(im0, da)
        im1 = local_data.take(i+di,axis=axis_numb)
        im1 = np.roll(im1, shift, 1)
        im1 = cv_rotate(im1, da)
        ti = im0+np.fliplr(im1) 
        ph_image.data_source.data = {'image': [ti]}
        push_notebook(handle=target)
    
    local_data = data3d
    ipywidgets.interact_manual(show_slice,
                        i=(0, data3d.shape[axis_numb]-1),
                        di=(0, data3d.shape[axis_numb]-5),
                        da=(-10,10,1),
                        shift = (-100,100,1)
                       )


# %%
diff_array(tomo_data, 0)

# %%
shift = -10# add -shift*2 to data_range [:, +:]
plt.figure(figsize=(10,10))
plt.plot(np.roll(tomo_data[0],shift,1).max(axis=0))
plt.plot(np.fliplr(np.roll(tomo_data[179],shift,1)).max(axis=0))
plt.grid()

# %%
tomo_data_fixed = np.roll(tomo_data, shift,2)

# %%
plt.figure(figsize=(10,10))
plt.imshow(tomo_data_fixed[0]-np.fliplr(tomo_data_fixed[179]), cmap=plt.cm.seismic)

# %%
angles = np.arange(len(tomo_data_fixed))*ang_step # Angles staep = 2 deg   

# %%
import scipy.ndimage
def my_rc(sino0, level):
    def get_my_b(level):
        t= np.mean(sino0, axis=0)
        gt = scipy.ndimage.filters.gaussian_filter1d(t,level/2.)
        return gt-t
    
    def get_my_a(level):
        my_b = get_my_b(level)
        return np.mean(my_b)/my_b.shape[0]
    
    my_a = get_my_a(level)
    my_b = get_my_b(level)
    
    res = sino0.copy()
    if not level==0:
        res+= sino0*my_a+my_b
    
    return res


# %%
ring_coor = 5
tomo_data_small = tomo_data_fixed[:,::1,::1]
tomo_data_small = tomo_data_small
for i in range(tomo_data_small.shape[1]):
    tomo_data_small[:,i,:] = my_rc(tomo_data_small[:,i,:], ring_coor)
tomo_data_small[tomo_data_small<100]=100
tomo_data_small = np.log(1.+tomo_data_fixed)
# tomo_data_small =(tomo_data_small.T/tomo_data_small.sum(axis=-1).sum(axis=-1)).T
angles_small = angles[:]

# %%
plt.figure(figsize=(10,10))
plt.imshow(tomo_data_small[:,200,:])
# plt.colorbar()

# %%
# tomo_data_small = tomo_data_fixed[:tomo_data_fixed.shape[0]//2,::2,::2]
# angles_small = angles[:angles.shape[0]//2]

# %%
s1 = np.require(tomo_data_small[:,::1,::1].swapaxes(0,1),dtype=np.float32, requirements=['C'])

# %%
rec_3d = astra_recon_3d_parallel_vec(s1, angles_small, bragg_ang, n_iters=20)

# %%
show_array(rec_3d,0)

# %%
back_sino = astra_fp_3d_parallel_vec(rec_3d, angles_small, bragg_ang)

# %%
show_array(back_sino-s1,1)

# %%
# import ipyvolume as ipv

# %%
import os
def reshape_volume(volume, reshape):
    res = np.zeros([s//reshape for s in volume.shape], dtype='float32')
    xs,ys,zs = [s*reshape for s in res.shape]
    for x,y,z in np.ndindex(reshape, reshape, reshape):
        res += volume[x:xs:reshape, y:ys:reshape, z:zs:reshape]
    return res/reshape**3

def save_amira(in_array, out_path, reshape=3):
    data_path = str(out_path)
    with open(os.path.join(data_path, 'amira.raw'), 'wb') as amira_file:
        reshaped_vol = reshape_volume(in_array, reshape)
        reshaped_vol.tofile(amira_file)
        file_shape = reshaped_vol.shape
        with open(os.path.join(data_path, 'tomo.hx'), 'w') as af:
                af.write('# Amira Script\n')
                af.write('remove -all\n')
                af.write(r'[ load -raw ${SCRIPTDIR}/amira.raw little xfastest float 1 '+
                         str(file_shape[2])+' '+str(file_shape[1])+' '+str(file_shape[0])+
                         ' 0 '+str(file_shape[2]-1)+' 0 '+str(file_shape[1]-1)+' 0 '+str(file_shape[0]-1)+
                         ' ] setLabel tomo.raw\n')


# %%
save_amira(rec_3d, '.', 1)
