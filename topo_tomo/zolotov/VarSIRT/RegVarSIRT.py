import numpy as np
import os
import pylab as plt

def norm2sq(mat):
  return np.dot(mat.ravel(), mat.ravel())

def dot_product(mat1, mat2):
  return np.dot(mat1.ravel(), mat2.ravel())

def tv(img):
  res = np.zeros(img.shape)
  res[:-1, :] = np.abs(np.diff(img, axis=0))
  res[:, :-1] += np.abs(np.diff(img, axis=1))
  return np.sum(res)

def grad_tv(img):
  res = np.zeros(img.shape)
  g_0 = np.diff(img, axis=0)
  g_1 = np.diff(img, axis=1)
  
  res[:-1, :] = np.sign(g_0)
  res[1:, :] += np.sign(-g_0)
  res[:, :-1] += np.sign(g_1)
  res[:, 1:] += np.sign(-g_1)
  return res

def grad_fun(A, sino, D, x, Lambda):
  k = 1. / sino.size
  return k * (2.0 * (A.T * ((D * (A * x).reshape(sino.shape) - sino) * D)).reshape(x.shape) - Lambda * grad_tv(x))

def fun(A, sino, D, x, Lambda):
  k = 1. / sino.size
  return k * (np.sum((D * (A * x).reshape(sino.shape) - sino) ** 2) + Lambda * tv(x))

def ternary_search(A, sino, D, x_0, grad, Lambda, left, right, eps=1e-3):
  while right - left > eps:
    a = (left * 2 + right) / 3
    b = (left + right * 2) / 3
    f1 = fun(A, sino, D, x_0 + a * grad, Lambda)
    f2 = fun(A, sino, D, x_0 + b * grad, Lambda)
    
    #print 'a: ', a, 'f1: ', f1
    #print 'b:', b, 'f2: ', f2

    if f1 < f2:
      right = b
    else:
      left = a
  return (left + right) / 2

def Fletcher_Reeves(grad_f, grad_f_old):
  return -norm2sq(grad_f)/norm2sq(grad_f_old)

def Polak_Ribier(grad_f, grad_f_old):
  return dot_product(grad_f, grad_f - grad_f_old) / norm2sq(grad_f_old)

def run(A, sino, x0, D=None, Lambda=0, eps=1e-10, n_it=100, step='CG'):
  if D is None:
    prefix = 's'
    D = np.ones_like(sino, dtype='float32')
  else:
    prefix = 'v'
  
  if not np.isclose(Lambda,0):
    prefix+='_tv'
      
  r = A * np.ones(x0.shape, dtype='float32')
  r[r == 0.0] = 1.0
  r = 1.0 / r
  max_v = sino.size
  
  sino_d = sino.copy()
  sino_d *= D

  en_ar = []
  x_k_ar = []
  al_ar = []

  x_k = x0.copy()
  grad_f = grad_fun(A, sino_d, D, x_k, Lambda)
  p_k = -np.copy(grad_f)
  grad_f_old = np.copy(grad_f)
  eng = 1.0
  eng_old = 2.0
  k = 0

  while (np.abs(eng_old - eng) > eps) and (k < n_it):
    if step == 'const':
      alpha = 1.0
    elif step == 'steepest':
      alpha = ternary_search(A, sino_d, D, x_k, -grad_f, Lambda, 0.0, 10.0, eps=1e-3)
    elif step == 'CG':
      alpha = ternary_search(A, sino_d, D, x_k, p_k, Lambda, 0.0, 10.0, eps=1e-3)
    
    x_k = x_k + alpha * p_k

    plt.imsave('x_' + str(k) + '.png', np.flipud(x_k), cmap='gray')
    plt.imsave('g_' + str(k) + '.png', np.flipud(grad_f), cmap='gray')
  
    grad_f_old = np.copy(grad_f)
    grad_f = grad_fun(A, sino_d, D, x_k, Lambda)
    
    beta = 0.0
    if step == 'CG':
      beta = Polak_Ribier(grad_f, grad_f_old)
      #beta = Fletcher_Reeves(grad_f, grad_f_old)
      if beta < 0.0:
          beta = 0.0
    
    p_k = -grad_f + beta * p_k

    eng_old = np.copy(eng)
    eng = np.sum(((A * x_k).reshape(sino.shape) - sino).ravel()**2) / max_v
    en_ar.append(eng)
    x_k_ar.append(x_k)
    al_ar.append(alpha)
    k += 1
    
    #if k % 10 == 0 or k < 20:
    #  plt.imsave('{}_{:004}.png'.format(prefix, k), x_k, vmin=0, vmax=0.1, cmap='gray')


  return {'rec': x_k,
          'iter': k,
          'energy': en_ar,
          'x_k': x_k_ar,
          'alpha': al_ar
          }



