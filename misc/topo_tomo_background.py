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

# %%
data_folders = ["/home/zolotovden/topo-tomo/Laue-analyzer/111_no_grid/20x30s",
                "/home/zolotovden/topo-tomo/Laue-analyzer/111-220_no_grid/20x30s",
                "/home/zolotovden/topo-tomo/Laue-analyzer/220_no_grid/20x30s"]


# %%
def remove_bg(data, dark):
    d1 = cv2.medianBlur(data,3)
    d2 = cv2.medianBlur(dark,3)
    
    k1 = np.percentile(d1, 50, axis=-1)
    k2 = np.percentile(d2, 50, axis=-1)
    
#     print(k1, k2)
    res = (d1.T*k2/k1).T - d2
    return res


# %%
for i in range(len(data_folders)):
    df = pathlib.Path(data_folders[i])
    data = []
    dark = []
    for f in df.glob('*'):
        d = plt.imread(f).astype('float32')
        if f.name == 'dark.tif':
            dark = d
        else:
            data.append(d)
            
    data = np.asanyarray(data)
    data_f = np.percentile(data, 90, axis=0).astype('float32')
    
    d = np.flipud(remove_bg(data_f, dark))
    d = restoration.denoise_bilateral(d-np.min(d), multichannel=False)
    
    im = Image.fromarray(d-d.min())
    im.save(f'{i}.tiff')
    
#     plt.imsave(f'{i}.tiff', d-d.min(), cmap=plt.cm.gray_r)
    
    plt.figure(figsize=(13,13))
    plt.imshow(d-d.min(), vmax=60, cmap=plt.cm.gray_r)
    plt.colorbar(orientation='horizontal')
    plt.show()

# %%
