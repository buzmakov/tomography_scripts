import numpy as np
import os

def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())

def dot_product(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())

def grad_fun(A, sino, D, x):
    return 2.0 * (A.T * ((D*(A*x).reshape(sino.shape) - sino)*D)).reshape(x.shape)

def fun(A, sino, D, x):
    return np.sum(((D*(A*x).reshape(sino.shape)- sino))**2)

def ternary_search(A, sino, D, x_0, grad, left, right, eps=1e-10):
    while right - left > eps:
        a = (left * 2 + right) / 3
        b = (left + right * 2) / 3
        f1 = fun(A, sino, D, x_0 + a * grad)
        f2 = fun(A, sino, D, x_0 + b * grad)
        if f1 < f2:
            right = b
        else:
            left = a
    return (left + right) / 2

def Fletcher_Reeves(grad_f, grad_f_old):
    return -norm2sq(grad_f)/norm2sq(grad_f_old)

def Polak_Ribier(grad_f, grad_f_old):
    return dot_product(grad_f, grad_f - grad_f_old)/norm2sq(grad_f_old)

def run(A, sino, D, x0, eps=1e-10, n_it=100, step='CG'):
    '''
    Conjugate Gradient and gradient descent algorithms to minimize the objective function
            0.5*||A*x - sino||_2^2
    '''
    
    # FOR ROI
    X,Y = np.meshgrid(np.arange(x0.shape[0]),np.arange(x0.shape[1]))
    X -= x0.shape[0]//2
    Y -= x0.shape[1]//2
    mask = (X**2+Y**2)<(x0.shape[0]//2)**2-10

    r = A * np.ones(x0.shape, dtype='float32')
    r[r == 0.0] = 1.0
    r = 1.0 / r
    max_v = sino.size

    en_ar = []
    alpha_ar = []
    beta_ar = []
    grad_ar = []
    x_k_ar = []

    x_k = x0.copy()
    grad_f = grad_fun(A, sino, D, x_k)
    p_k = -np.copy(grad_f)
    grad_f_old = np.copy(grad_f)
    eng = 1.0
    eng_old = 2.0
    k = 0

    while (np.abs(eng_old - eng) > eps) and (k < n_it):
        if step == 'const':
            alpha = 0.00001
        elif step == 'steepest':
            alpha = ternary_search(A, sino, D, x_k, -grad_f, 0.0, 0.3, eps=1e-14)
        elif step == 'CG':
            alpha = ternary_search(A, sino, D, x_k, p_k, 0.0, 0.3, eps=1e-14)
        
        x_k = x_k + alpha * p_k
        
        # FIX for ROI
        x_k*=mask
        x_k*=x_k>0
        
        grad_ar.append(p_k)
        grad_f_old = np.copy(grad_f)
        grad_f = grad_fun(A, sino, D, x_k)
        
        beta = 0.0
        if step == 'CG':
            beta = Polak_Ribier(grad_f, grad_f_old)
            #beta = Fletcher_Reeves(grad_f, grad_f_old)
            if beta < 0.0:
                beta = 0.0
        
        p_k = -grad_f + beta * p_k

        eng_old = np.copy(eng)
        eng = np.sum(r * ((D*(A*x_k).reshape(sino.shape) - sino)).ravel()**2) / max_v
        en_ar.append(eng)
        alpha_ar.append(alpha)
        beta_ar.append(beta)
        x_k_ar.append(x_k)
        k += 1

    return {'rec': x_k,
                    'iter': k,
                    'energy': en_ar,
                    'alpha': alpha_ar,
                    'beta': beta_ar,
                    'grad': grad_ar,
                    'x_k': x_k_ar
                    }

from numba import cuda, jit


def fun_gpu(A, sino, D, x):
    
    Ax =np.zeros(sino.shape, dtype='float32')
    A.FP(x, out=Ax)
    t=D*Ax-sino
    t=t*t
    res = np.sum(t)
    return res 

def ternary_search_gpu(A, sino, D, x_0, grad, left, right, eps=1e-10):
    while right - left > eps:
        a = (left * 2 + right) / 3
        b = (left + right * 2) / 3
        f1 = fun_gpu(A, sino, D, x_0 + a * grad)
        f2 = fun_gpu(A, sino, D, x_0 + b * grad)
        if f1 < f2:
            right = b
        else:
            left = a
    return (left + right) / 2

def run_gpu(A, sino, D, x0, eps=1e-10, n_it=100, step='CG'):
    '''
    Conjugate Gradient and gradient descent algorithms to minimize the objective function
            0.5*||A*x - sino||_2^2
    '''
    
    # FOR ROI
    X,Y = np.meshgrid(np.arange(x0.shape[0]),np.arange(x0.shape[1]))
    X -= x0.shape[0]//2
    Y -= x0.shape[1]//2
    mask = (X**2+Y**2)<(x0.shape[0]//2)**2-10

    r = A * np.ones(x0.shape, dtype='float32')
    r[r == 0.0] = 1.0
    r = 1.0 / r
    max_v = sino.size

    en_ar = []
    alpha_ar = []
    beta_ar = []
    grad_ar = []
    x_k_ar = []

    x_k = x0.copy()
    grad_f = grad_fun(A, sino, D, x_k)
    p_k = -np.copy(grad_f)
    grad_f_old = np.copy(grad_f)
    eng = 1.0
    eng_old = 2.0
    k = 0

    while (np.abs(eng_old - eng) > eps) and (k < n_it):
        if step == 'const':
            alpha = 0.00001
        elif step == 'steepest':
            alpha = ternary_search_gpu(A, sino, D, x_k, -grad_f, 0.0, 0.3, eps=1e-14)
        elif step == 'CG':
            alpha = ternary_search_gpu(A, sino, D, x_k, p_k, 0.0, 0.3, eps=1e-14)
        
        x_k = x_k + alpha * p_k
        
        # FIX for ROI
        x_k*=mask
        x_k*=x_k>0
        
        grad_ar.append(p_k)
        grad_f_old = np.copy(grad_f)
        grad_f = grad_fun(A, sino, D, x_k)
        
        beta = 0.0
        if step == 'CG':
            beta = Polak_Ribier(grad_f, grad_f_old)
            #beta = Fletcher_Reeves(grad_f, grad_f_old)
            if beta < 0.0:
                beta = 0.0
        
        p_k = -grad_f + beta * p_k

        eng_old = np.copy(eng)
        eng = np.sum(r * ((D*(A*x_k).reshape(sino.shape) - sino)).ravel()**2) / max_v
        en_ar.append(eng)
        alpha_ar.append(alpha)
        beta_ar.append(beta)
        x_k_ar.append(x_k)
        k += 1

    return {'rec': x_k,
                    'iter': k,
                    'energy': en_ar,
                    'alpha': alpha_ar,
                    'beta': beta_ar,
                    'grad': grad_ar,
                    'x_k': x_k_ar
                    }