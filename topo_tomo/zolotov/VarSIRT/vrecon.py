import numpy as np
import os
import pylab as plt

def calculateNonNegative(x):
  res = np.zeros_like(x)
  res = -1.0 * x[x < 0]
  res = np.sign(res)
  return np.sum(res)

def compute_regularization(x, reg_params):
  res_linear = 0.0
  res_exp = 0.0
  res_l1 = 0.0
  res_l2 = 0.0
  res_l1_grad = 0.0
  res_l2_grad = 0.0
  if reg_params[1] >= 0.000001:
    res_linear = np.zeros_like(x)
    res_linear[x < 0] = x[x < 0]
    res_linear = -1. * np.sum(res_linear)
  if reg_params[2] >= 0.000001:
    res_exp = np.zeros_like(x)
    res_exp[x < 0] = x[x < 0] 
    res_exp = np.sum(res_exp ** 2)
  if reg_params[3] >= 0.000001:
    res_l1 = np.sum(np.abs(x))
  if reg_params[4] >= 0.000001:
    res_l2 = np.sum(x ** 2)
  if reg_params[5] >= 0.000001:
    res_l1_grad = np.zeros_like(x)
    res_l1_grad[:-1, :] = np.abs(np.diff(x, axis=0))
    res_l1_grad[:, :-1] += np.abs(np.diff(x, axis=1))
    res_l1_grad = np.sum(res_l1_grad)
  if reg_params[6] >= 0.000001:
    res_l2_grad = np.zeros_like(x)
    res_l2_grad[:-1, :] = np.diff(x, axis=0) ** 2
    res_l2_grad[:, :-1] += np.diff(x, axis=1) ** 2
    res_l2_grad = np.sum(res_l2_grad)
  
  sx = x.shape[0]
  k = sx / calculateNonNegative(x)
  res_linear *= k
  res_exp *= k
  
  res_l1 /= sx
  res_l1_grad /= sx
  res_l2 /= sx
  res_l2_grad /= sx

  res_l2 = sx * np.sqrt(res_l2 / sx);
  res_l2_grad = sx * np.sqrt(res_l2_grad / sx);
  res_exp = sx * np.sqrt(res_exp / sx);

  res = res_linear * reg_params[1] + res_exp * reg_params[2] + res_l1 * reg_params[3] + res_l1_grad * reg_params[5] + res_l2 * reg_params[4] + res_l2_grad * reg_params[6];
  return res

def compute_functional(D_r, D_g, D_r_grad, x_k, alpha, reg_parameters):
  k = 1.0 / (D_r.size)
  f = k * np.sum((D_r + alpha * D_r_grad) ** 2)
  f = np.sqrt(f) * (1.0 - reg_parameters[0]) + reg_parameters[0] * compute_regularization(x_k + alpha * D_g, reg_parameters);
  return f

def normilize_gr(g, grad_buffer, reg_params_1, reg_params_2):
  norm1 = 1.0 / np.sum(np.abs(grad_buffer))
  if norm1 > 1e+6:
    norm = 1.0
  g += grad_buffer * (reg_params_1 * reg_params_2 * norm1)
  return g

def regularize_gradient(x, g, reg_params):
  norm = 1.0 / np.sum(np.abs(g))
  if norm < 1e+6:
    g *= norm

  if reg_params[0] < 0.000001:
    return g

  g = g * (1. - reg_params[0])
  
  if np.abs(reg_params[1]) > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer[x < 0] = -1. * [x < 0]
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[1])
  if np.abs(reg_params[2]) > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer[x < 0] = x[x < 0]
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[2])
  if np.abs(reg_params[3]) > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer = np.sign(x)
    grad_buffer[grad_buffer == 0.0] = 1.0
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[3])
  if np.abs(reg_params[4]) > 0.000001:
    grad_buffer = np.copy(x)
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[4])
  if np.abs(reg_params[5]) > 0.000001:
    grad_buffer = np.zeros_like(g)
    g_0 = np.diff(x, axis=0)
    g_1 = np.diff(x, axis=1)
    mask_0 = g_0 == 0.0
    mask_1 = g_1 == 0.0
    g_0 [mask_0] = -1.0
    grad_buffer[:-1, :] = np.sign(-g_0)
    g_0 [mask_0] = 1.0
    grad_buffer[1:, :] += np.sign(g_0)
    g_1 [mask_1] = -1.0
    grad_buffer[:, :-1] += np.sign(-g_1)
    g_1 [mask_1] = 1.0
    grad_buffer[:, 1:] += np.sign(g_1)
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[5])
  if np.abs(reg_params[6]) > 0.000001:
    grad_buffer = np.zeros_like(g)
    g_0 = np.diff(x, axis=0)
    g_1 = np.diff(x, axis=1)
    grad_buffer[:-1, :] = -g_0
    grad_buffer[1:, :] += g_0
    grad_buffer[:, :-1] -= g_1
    grad_buffer[:, 1:] += g_1
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[6])

  return g

def compute_gradient(A, sino, D, x, reg_parameters):
  D_r = D * (A * x).reshape(sino.shape) - sino
  D_g = (A.T * (D_r * D)).reshape(x.shape)
  D_g = regularize_gradient(x, D_g, reg_parameters)
  D_r_grad = D * (A * D_g).reshape(sino.shape)
  return D_r, D_g, D_r_grad

def ternary_search(D_r, D_g, D_r_grad, x_k, left, right, reg_parameters):
  while right - left > 1e-2:
    a = (left * 2 + right) / 3
    b = (left + right * 2) / 3
    
    f1 = compute_functional(D_r, D_g, D_r_grad, x_k, a, reg_parameters)
    f2 = compute_functional(D_r, D_g, D_r_grad, x_k, b, reg_parameters)

    if f1 < f2:
      right = b
    else:
      left = a

  return (left + right) / 2

def iterate(A, sino, x0, D=None, reg_parameters=None, n_it=100):
  if D is None:
    prefix = 'cg'
    D = np.ones_like(sino, dtype='float32')
  else:
    prefix = 'vcg'
  reg_param = np.zeros(7)
  if reg_parameters is not None:
    if 'total' in reg_parameters:
      reg_param[0] = reg_parameters['total']
    if 'lin' in reg_parameters:
      reg_param[1] = reg_parameters['lin']
    if 'exp' in reg_parameters:
      reg_param[2] = reg_parameters['exp']
    if 'l1' in reg_parameters:
      reg_param[3] = reg_parameters['l1']
    if 'l2' in reg_parameters:
      reg_param[4] = reg_parameters['l2']
    if 'l1_grad' in reg_parameters:
      reg_param[5] = reg_parameters['l1_grad']
    if 'l2_grad' in reg_parameters:
      reg_param[6] = reg_parameters['l2_grad']
    prefix+='_reg'
   
  sino_d = sino.copy()
  sino_d *= D

  x_k = x0.copy()
  en_ar = []
  x_k_ar = []
  al_ar = []
  it_count = n_it
  for it in range(0, n_it):
    D_r, D_g, D_r_grad = compute_gradient(A, sino_d, D, x_k, reg_param) 

    if compute_functional(D_r, D_g, D_r_grad, x_k, 0.0, reg_param) < 1e-8:
      it_count = it - 1
      break

    alpha = ternary_search(D_r, D_g, D_r_grad, x_k, -500.0, 0.0, reg_param)
    x_k = x_k + alpha * D_g
    
    eng = np.sum(((A * x_k).reshape(sino.shape) - sino) ** 2)
    en_ar.append(np.sqrt(eng))
    x_k_ar.append(x_k)
    al_ar.append(alpha)
    
    #if it % 10 == 0 or it < 20:
    #  plt.imsave('{}_{:004}.png'.format(prefix, it), x_k, vmin=0, vmax=0.1, cmap='gray')

  return {'rec': x_k,
          'iter': it_count,
          'energy': en_ar,
          'x_k': x_k_ar,
          'alpha': al_ar
          }

def CG(A, sino, x0, reg_parameters=None, n_it=100):
     return iterate(A, sino, x0, None, reg_parameters, n_it)

def VCG(A, sino, x0, D, reg_parameters=None, n_it=100):
  return iterate(A, sino, x0, D, reg_parameters, n_it)
          