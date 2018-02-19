__author__ = 'makov'
import numpy
import scipy.ndimage

def SSIM(x,y):
    def cov(x,y):
        return scipy.ndimage.measurements.mean(x*y)-\
               scipy.ndimage.measurements.mean(x)*scipy.ndimage.measurements.mean(y)
    import scipy.ndimage
    sigma_x=scipy.ndimage.measurements.variance(x)
    sigma_y=scipy.ndimage.measurements.variance(y)
    mean_x=scipy.ndimage.measurements.mean(x)
    mean_y=scipy.ndimage.measurements.mean(y)
    L=1.0
    c1=(0.01*L)**2
    c2=(0.03*L)**2
    ssim=(2*mean_x*mean_y+c1)*(2*cov(x,y)+c2)/((mean_x**2+mean_y**2+c1)*(sigma_x+sigma_y+c2))
    return ssim

def MSSIM(x,y,block_size=8):
    if x.shape!=y.shape:
        raise Exception("Input arrays must be equal shape")
    if x.shape[0]!=x.shape[1]:
        raise Exception("Input arrays must be square")
    grid_size=x.shape[0]/block_size if x.shape[0]%block_size==0 else x.shape[0]/block_size+1
    mssim_matrix=numpy.zeros(shape=(grid_size,grid_size))
    for ix in range(grid_size):
        range_x=range(ix*block_size,min((ix+1)*block_size,x.shape[0]))
        for iy in range(grid_size):
            range_y=range(iy*block_size,min((iy+1)*block_size,x.shape[0]))
            tx=x[numpy.ix_(range_x,range_y)]
            ty=y[numpy.ix_(range_x,range_y)]
            mssim_matrix[ix,iy]=SSIM(tx,ty)
    return mssim_matrix