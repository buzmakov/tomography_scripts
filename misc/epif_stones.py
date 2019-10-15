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
import os
import numpy as np
import h5py
import pylab as plt
from scipy import ndimage as ndi

from scipy.ndimage.morphology import distance_transform_edt

from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.measure import regionprops

# %%
data_files = [
    '/home/krivonosov/reconstruction/e89e7874-3178-4807-8678-df4e9695f4ae/e89e7874-3178-4807-8678-df4e9695f4ae.h5',
    '/home/krivonosov/reconstruction/bc9b34a9-144d-4c2e-bd0d-3ba2f6358eff/bc9b34a9-144d-4c2e-bd0d-3ba2f6358eff.h5']

# %%
df = data_files[0]

# %%
data = h5py.File(df, 'r')['Reconstruction'].value
data.shape


# %%
def test_segmentation(image, mask=None):
    plt.figure(figsize=(13,13))
    plt.imshow(image)
    plt.colorbar()
    if not mask is None:
        plt.contour(mask, levels=range(np.max(mask)), colors=['r',])
    plt.show()

test_segmentation(data[350], data[350]>0.9)

# %%
mask = data>0.9

# %%
data_dtf = distance_transform_edt(mask)

# %%
test_segmentation(data_dtf[350])

# %%
local_maxi = peak_local_max(data_dtf, indices=False, 
                            threshold_abs=2, min_distance=20,# footprint=np.ones((3, 3, 3)),
                           labels=mask)
markers, num_features = ndi.label(local_maxi)#, np.ones((3, 3, 3)))
labels = watershed(-data_dtf, markers, mask=mask)

# %%
print(num_features)

# %%
test_segmentation(labels[:,400], mask[:,400])
test_segmentation(data[:,400], labels[:,400])

test_segmentation(labels[400], mask[400])
test_segmentation(data[400], labels[400])

# %%
regions = regionprops(labels)
print(len(regions))

# %%
areas = [r.area for r in regions]

# %%
plt.figure(figsize=(10,10))
plt.hist(areas, bins=1000)
plt.xlim(0,np.percentile(areas, 90))
plt.grid()
plt.show()

# %%
markers_m, num_features_m = ndi.label(mask, np.ones((3,3,3)))

# %%
num_features_m

# %%
plt.figure(figsize=(13,13))
plt.imshow(markers_m[500], vmin=0, vmax=markers_m.max())
plt.show()

plt.figure(figsize=(13,13))
plt.imshow(markers_m[:, 500], vmin=0, vmax=markers_m.max())
plt.show()

test_segmentation(markers_m[:,400], mask[:,400])
test_segmentation(data[:,400], markers_m[:,400])

test_segmentation(markers_m[400], mask[400])
test_segmentation(data[400], markers_m[400])

# %%
regions_m = regionprops(markers_m)
print(len(regions_m))

# %%
areas_m = [r.area for r in regions_m]
x,y = np.histogram(areas_m, bins=10000)
plt.figure(figsize=(10,10))
plt.semilogy(y[:-1],x,'o')
plt.xlim(0,np.percentile(areas_m, 99))
plt.grid()
plt.show()


# %%
def reshape_volume(volume, reshape):
    res = np.zeros([s//reshape for s in volume.shape], dtype='float32')
    xs,ys,zs = [s*reshape for s in res.shape]
    for x,y,z in np.ndindex(reshape, reshape, reshape):
        res += volume[x:xs:reshape, y:ys:reshape, z:zs:reshape]
    return res/reshape**3

def save_amira(in_array, out_path, reshape=3):
    data_path = out_path
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
save_amira(markers_m, '.', 1)

# %%
