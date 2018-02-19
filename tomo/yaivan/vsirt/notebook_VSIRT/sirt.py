import numpy as np
import os

import pylab as plt

def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())

def dot_product(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())

def grad_fun(A, sino, x):
    return (A.T * ((A*x).reshape(sino.shape) - sino)).reshape(x.shape)

def fun(A, sino, x):
    return 0.5 * np.sum(((A*x).reshape(sino.shape) - sino)**2)

def ternary_search(A, sino, x_0, grad, left, right, eps=1e-10):
    while right - left > eps:
        a = (left * 2 + right) / 3
        b = (left + right * 2) / 3
        f1 = fun(A, sino, x_0 + a * grad)
        f2 = fun(A, sino, x_0 + b * grad)
        if f1 < f2:
            right = b
        else:
            left = a
    return (left + right) / 2

def Fletcher_Reeves(grad_f, grad_f_old):
    return -norm2sq(grad_f)/norm2sq(grad_f_old)

def Polak_Ribier(grad_f, grad_f_old):
    return dot_product(grad_f, grad_f - grad_f_old)/norm2sq(grad_f_old)

def run(A, sino, x0, eps=1e-10, n_it=100, step='CG'):
    '''
    Conjugate Gradient and gradient descent algorithms to minimize the objective function
            0.5*||A*x - sino||_2^2
    '''
    max_v = sino.size

    en_ar = []
    alpha_ar = []
    beta_ar = []

    x_k = x0.copy()
    grad_f = grad_fun(A, sino, x_k)
    p_k = -np.copy(grad_f)
    grad_f_old = np.copy(grad_f)
    eng = 1.0
    eng_old = 2.0
    k = 0

    while (np.abs(eng_old - eng) > eps) and (k < n_it):
        if step == 'const':
            alpha = 0.00001
        elif step == 'steepest':
            alpha = ternary_search(A, sino, x_k, -grad_f, 0.0, 10.0, eps=1e-10)
        elif step == 'CG':
            alpha = ternary_search(A, sino, x_k, p_k, 0.0, 10.0, eps=1e-10)
        
        x_k = x_k + alpha * p_k
        grad_f_old = np.copy(grad_f)
        grad_f = grad_fun(A, sino, x_k)
        
        beta = 0.0
        if step == 'CG':
            beta = Polak_Ribier(grad_f, grad_f_old)
            #beta = Fletcher_Reeves(grad_f, grad_f_old)
            if beta < 0.0:
                beta = 0.0
        
        p_k = -grad_f + beta * p_k

        eng_old = np.copy(eng)
        eng = np.sum(((A*x_k).reshape(sino.shape) - sino).ravel()**2) / max_v
        en_ar.append(eng)
        alpha_ar.append(alpha)
        beta_ar.append(beta)
        k += 1
        
        if k%10 == 0 or k<20:
            plt.imsave('s_{:004}.png'.format(k), x_k, vmin=0, vmax=0.1, cmap='gray')

    return {'rec': x_k,
            'iter': k,
            'energy': en_ar,
            'alpha': alpha_ar,
            'beta': beta_ar
            }
