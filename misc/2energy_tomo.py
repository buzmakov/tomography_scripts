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

# %%
import numpy as np
import pylab as plt
from glob import glob
import pathlib
import cv2
from skimage import restoration
from PIL import Image
from tqdm import tqdm_notebook
import imageio

# %%
data_folders = ["/home/zolotovden/topo-tomo/Laue-analyzer/experiment/2019_06_24/Kalpha1+Kbeta1_exp",
               ]


# %%
def safe_median(data):
    m_data = cv2.medianBlur(data,3)
    mask = np.abs(m_data-data) > 0.1*np.abs(data)
    res = data.copy()
    res[mask] = m_data[mask]
    return res


# %%
def remove_bg(data, dark):
#     d1 = cv2.medianBlur(data,3)
#     d2 = cv2.medianBlur(dark,3)
    
    d1 = safe_median(data)
    d2 = safe_median(dark)
    k1 = np.percentile(d1, 50, axis=-1)
    k2 = np.percentile(d2, 50, axis=-1)
    
#     print(k1, k2)
    res = (d1.T*k2/k1).T - d2
    return res


# %%
for i in range(len(data_folders)):
    df = pathlib.Path(data_folders[i])
    data = []
    empty = []
    dark = []
    for f in df.glob('dark2*.tif'):
        d = plt.imread(f).astype('float32')[1400:1700, 1200:2600]
        dark.append(d)
    
    dark = np.asanyarray(dark)
    dark_f = np.percentile(dark, 90, axis=0).astype('float32')
    
    for f in df.glob('empty*.tif'):
        d = plt.imread(f).astype('float32')[1400:1700, 1200:2600]
        empty.append(d)
        
    empty = np.asanyarray(empty)
    empty_f = np.percentile(empty, 90, axis=0).astype('float32')
    
    for f in tqdm_notebook(df.glob('sample*.tif')):
        d = plt.imread(f).astype('float32')[1400:1700, 1200:2600]
#         im = Image.fromarray(remove_bg(d, dark_f).astype('float32'))
#         im.save(f'{df}/post/{f.name}f')
        imageio.imwrite(f'{df}/post/{f.name}f', remove_bg(d, dark_f).astype('float32'))
#         data.append(d)
#     d = np.flipud(remove_bg(data_f, dark))
#     d = restoration.denoise_bilateral(d-np.min(d), multichannel=False)
    
#     im = Image.fromarray(d-d.min())
#     im.save(f'{i}.tiff')
    
#     plt.imsave(f'{i}.tiff', d-d.min(), cmap=plt.cm.gray_r)
    
    plt.figure(figsize=(8,5))
    plt.imshow(safe_median(dark_f), cmap=plt.cm.gray_r)
    plt.colorbar(orientation='horizontal')
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.imshow(safe_median(empty_f), cmap=plt.cm.gray_r)
    plt.colorbar(orientation='horizontal')
    plt.show()

# %%
plt.figure(figsize=(8,5))
plt.imshow(remove_bg(data[10], dark_f), cmap=plt.cm.gray_r)
plt.colorbar(orientation='horizontal')
plt.show()

# %%
# !mkdir {df}/post

# %%
d = imageio.imread(f'{df}/post/{f.name}f')
d.shape

# %%
d.shape

# %%
plt.figure(figsize=(8,5))
plt.imshow(d, cmap=plt.cm.gray_r)
plt.colorbar(orientation='horizontal')
plt.show()

# %%
