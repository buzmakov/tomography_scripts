import numpy as np
import os
import pylab as plt

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

  res_linear /= (x.size)
  res_exp /= (x.size)
  res_l1 /= (x.size)
  res_l1_grad /= (x.size)
  res_l2 /= (x.size)
  res_l2_grad /= (x.size)

  res_l2 = np.sqrt(res_l2);
  res_l2_grad = np.sqrt(res_l2_grad);
  res_exp = np.sqrt(res_exp);

  res = res_linear * reg_params[1] + res_exp * reg_params[2] + res_l1 * reg_params[3] + res_l1_grad * reg_params[5] + res_l2 * reg_params[4] + res_l2_grad * reg_params[6];
  return res

def compute_functional(D_r, D_g, D_r_grad, x_k, alpha, reg_parameters):
  k = 1. / (4. * D_r.size)
  f = k * np.sum((D_r + alpha * D_r_grad) ** 2)
  f = np.sqrt(f) * (1. - reg_parameters[0])
  f = f + reg_parameters[0] * compute_regularization(x_k + alpha * D_g, reg_parameters);
  return f

def normilize_gr(g, grad_buffer, reg_params_1, reg_params_2):
  norm1 = np.sum(np.abs(grad_buffer))
  if norm1 > 1e-6:
    g += grad_buffer * (reg_params_1 * reg_params_2 / norm1)
  return g

def regularize_gradient(x, g, reg_params):
  norm = np.sum(np.abs(g))
  if norm > 1e-6:
    g = g / norm

  if reg_params[0] < 0.000001:
    return g

  g = g * (1. - reg_params[0])
  
  if reg_params[1] > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer[x < 0] = -1.* [x < 0]
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[1])
  if reg_params[2] > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer[x < 0] = x[x < 0]
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[2])
  if reg_params[3] > 0.000001:
    grad_buffer = np.zeros_like(g)
    grad_buffer = np.sign(x)
    grad_buffer[grad_buffer == 0.0] = 1.0
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[3])
  if reg_params[4] > 0.000001:
    grad_buffer = np.copy(x)
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[4])
  if reg_params[5] > 0.000001:
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
  if reg_params[6] > 0.000001:
    grad_buffer = np.zeros_like(g)
    g_0 = np.diff(x, axis=0)
    g_1 = np.diff(x, axis=1)
    grad_buffer[:-1, :] = -g_0
    grad_buffer[1:, :] += g_0
    grad_buffer[:, :-1] -= g_1
    grad_buffer[:, 1:] += g_1
    g = normilize_gr(g, grad_buffer, reg_params[0], reg_params[6])
  
  norm_final = np.sum(np.abs(g))
  if norm_final > 1e-6:
    g = g / norm_final
  return g

def compute_gradient(A, sino, D, x, reg_parameters):
  D_r = D * (A * x).reshape(sino.shape) - sino
  D_g = (A.T * (D_r * D)).reshape(x.shape)
  D_g = regularize_gradient(x, D_g, reg_parameters)
  D_r_grad = D * (A * D_g).reshape(sino.shape)
  return D_r, D_g, D_r_grad

def ternary_search(D_r, D_g, D_r_grad, x_k, left, right, reg_parameters, eps=1e-3):
  while right - left > 1e-10:
    a = (left * 2 + right) / 3
    b = (left + right * 2) / 3
    
    f1 = compute_functional(D_r, D_g, D_r_grad, x_k, a, reg_parameters)
    f2 = compute_functional(D_r, D_g, D_r_grad, x_k, b, reg_parameters)

    if f1 < f2:
      right = b
    else:
      left = a

    if f1 + eps >= f2 and f2 + eps >= f1:
      break
  return (left + right) / 2

def iterate(A, sino, x0, D=None, reg_parameters=None, n_it=100):
  if D is None:
    prefix = 's'
    D = np.ones_like(sino, dtype='float32')
  else:
    prefix = 'v'
  reg_param = np.zeros(7)
  print reg_param
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
  D_r, D_g, D_r_grad = compute_gradient(A, sino_d, D, x_k, reg_param)
  
  en_ar = []
  x_k_ar = []
  al_ar = []
  it_count = n_it
  for it in range(0, n_it):
    k = 1. / (4. * D_r.size)
    norm_g = k * np.sum(np.abs(D_g))

    if norm_g < 1e-9:
      it_count = it - 1
      break

    functional_current = compute_functional(D_r, D_g, D_r_grad, x_k, 0.0, reg_param);
    lower_border_of_search = -(np.abs(functional_current) / norm_g)

    if lower_border_of_search < -1000.0:
      lower_border_of_search = -1000.0

    alpha = ternary_search(D_r, D_g, D_r_grad, x_k, lower_border_of_search, 0.0, reg_param, 0.001 * functional_current)
    x_k = x_k + alpha * D_g
  
    D_r, D_g, D_r_grad = compute_gradient(A, sino_d, D, x_k, reg_param)

    
    vol = (A * x_k).reshape(sino.shape) - sino
    eng = np.sum(vol ** 2)
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
          