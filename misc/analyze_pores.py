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
import h5py
from tqdm import tqdm_notebook

from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, binary_opening, binary_dilation, binary_erosion
# from skimage.morphology import watershe
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed, random_walker

import cv2

# %%
data = h5py.File('tomo_rec.h5')['Reconstruction'][150:,300:1000, 200:900]


# %%
def create_mask(im):
    t = im>0.01
    t = cv2.medianBlur(t.astype('float32'), 7)
    t = binary_fill_holes(binary_closing(t))
    t = binary_erosion(t)
    return t               


# %%
plt.figure(figsize=(10,10))
plt.imshow(data[340], vmin=0.01, vmax=0.1, cmap=plt.cm.gray_r)
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(create_mask(data[340]), cmap=plt.cm.gray_r)
# plt.colorbar()
plt.show()

# %%
# # !rm -rf images

# %%
# !mkdir images

# %%
for i in tqdm_notebook(range(data.shape[0])):
    plt.imsave(f'images/0_{i}.png',data[i], vmin=0.01, vmax=0.1, cmap=plt.cm.gray_r)

for i in tqdm_notebook(range(data.shape[1])):
    plt.imsave(f'images/1_{i}.png',data[:,i,:], vmin=0.01, vmax=0.1, cmap=plt.cm.gray_r)

for i in tqdm_notebook(range(data.shape[2])):
    plt.imsave(f'images/2_{i}.png',data[:,:,i], vmin=0.01, vmax=0.1, cmap=plt.cm.gray_r)

# %%
# !ffmpeg -y -r 10 -i "images/0_%d.png" -b:v 2000k poly_0.avi
# !ffmpeg -y -r 10 -i "images/1_%d.png" -b:v 2000k poly_1.avi
# !ffmpeg -y -r 10 -i "images/2_%d.png" -b:v 2000k poly_2.avi

# %%
for i in tqdm_notebook(range(data.shape[0])):
    data[i] = cv2.medianBlur(np.asarray(data[i]>0.01, dtype='float32'), 7)


# %%
# x = data[200]
def find_pores(x, debug=False):
    x = x.copy()
    x[x<0.01] = 0.01
    x_m = cv2.medianBlur(np.asarray(x, dtype='float32'), 7)-cv2.medianBlur(np.asarray(x, dtype='float32'), 3)
    data_dtf = distance_transform_edt(x>0)
    data_dtf_r = distance_transform_edt(x<1)
    pores = binary_opening(binary_closing((np.abs(x_m)<0.004)*(x<0.08)))
    mask = create_mask(x)
    pores = pores*mask
    
    if debug:
        plt.figure(figsize=(15,15))
        plt.imshow(x)
        plt.contour(pores)
#         plt.colorbar(orientation='horizontal')
        plt.show()

#         plt.figure(figsize=(15,15))
#         plt.imshow(pores)
#         # plt.colorbar(orientation='horizontal')
#         plt.show()

#         plt.figure(figsize=(15,15))
#         plt.imshow(mask)
#         # plt.colorbar(orientation='horizontal')
#         plt.show()
    return pores


# %%
for i in range(70,350, 50):
     find_pores(data[i], True)   

# %%
pores = data[70:].copy()
for i in tqdm_notebook(range(pores.shape[0])):
    pores[i] = find_pores(pores[i])

# %%
pores_t = pores #[200:300, 200:500, 200:500]
# mask_t = mask[200:300, 200:500, 200:500]
pores_dtf = distance_transform_edt(pores_t)
pores_dtf_r = distance_transform_edt(1-pores_t)

# %%
plt.figure(figsize=(15,15))
plt.imshow(pores_dtf[50])
plt.colorbar(orientation='horizontal')
plt.show()

# plt.figure(figsize=(15,15))
# plt.imshow(pores_dtf_r[50]*binary_erosion(mask, iterations=20), vmax=5)
# plt.colorbar(orientation='horizontal')
# plt.show()

plt.figure(figsize=(15,15))
plt.imshow(pores_dtf_r[50], vmax=5)
plt.colorbar(orientation='horizontal')
plt.show()

# %%
# #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
local_maxi = peak_local_max(pores_dtf, indices=False, 
                            threshold_abs=2, min_distance=10,# footprint=np.ones((3, 3, 3)),
                           labels=pores_t)# 
markers, num_features = ndi.label(local_maxi)#, np.ones((3, 3, 3)))
labels = watershed(-pores_dtf, markers, mask=pores_t)

# %%
#https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
# pores_t = pores[200:300, 200:500, 200:500]
# local_maxi = peak_local_max(pores_dtf, indices=False, min_distance=3)#, footprint=np.ones((3, 3, 3)))
# markers, num_features = ndi.label(pores_t)
# labels = watershed(pores_t, markers)

# %%
num_features

# %%
regions=regionprops(labels)

# %%
print(regions[0].equivalent_diameter)

# %%
plt.figure(figsize=(15,15))
plt.imshow(pores_dtf[51])
# plt.colorbar(orientation='horizontal')
plt.contour(markers[51])

plt.show()

# %%
plt.figure(figsize=(15,15))
plt.imshow(pores_t[50])
plt.contour(labels[50], colors='r')#, vmin = np.percentile(labels[200].flat, 77))
# plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(labels[50])
# plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(labels[:,200,:])
# plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(labels[:,:,200])
# plt.colorbar(orientation='horizontal')
plt.show()

# %%
vol = [r.area for r in regions]
# #volume of each pore
# vol = np.zeros((num_features+1), dtype=int)
# for x in tqdm_notebook(labels.flat):
#     vol[x] += 1

# %%
xv, yv = np.histogram(vol[1:], bins=100)
plt.figure(figsize=(15,15))
plt.semilogy(yv[1:],xv,'o')
plt.grid()
plt.show()

# %%
#Raduis of each pore
tt = local_maxi*pores_dtf  #todo.fixit
xr, yr = np.histogram(tt.flat, bins=100)
xr0, yr0 = np.histogram(np.power(vol,1./3), bins=1000)

# %%
plt.figure(figsize=(15,15))
plt.semilogy(yr[1:],xr[:],'o')
plt.semilogy(yr0[2:],xr0[1:],'o')
plt.xlim([0,20])
plt.grid()
plt.show()

# %%

# %%
