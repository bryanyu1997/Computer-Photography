''' Functions in deblur flow '''

import numpy as np
import cv2 as cv
from scipy import ndimage, signal
from scipy.signal import convolve2d

import sys
DBL_MIN = sys.float_info.min





########################################################
def Wiener_deconv(img_in, k_in, SNR_F):
    """ Wiener deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                SNR_F (float): Wiener deconvolution parameter
            Returns:
                Wiener_result (uint8 ndarray, shape(height, width, ch)): Wiener-deconv image
                
            Todo:
                Wiener deconvolution
    """
    


    return Wiener_result









########################################################
def RL(img_in, k_in, max_iter):
    """ RL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                
            Returns:
                RL_result (uint8 ndarray, shape(height, width, ch)): RL-deblurred image
                
            Todo:
                RL deconvolution
    """

    
    
    return RL_result



########################################################
def BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk):
    """ BRL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_result (uint8 ndarray, shape(height, width, ch)): BRL-deblurred image
                
            Todo:
                BRL deconvolution
    """


    return BRL_result
    





########################################################
def RL_energy(img_in, k_in, I_in):
    """ RL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                
            Returns:
                RL_energy (float): RL_energy
                
            Todo:
                Calculate RL energy
    """


    
    return RL_energy




########################################################
def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk):
    """ BRL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_energy (float): BRL_energy
                
            Todo:
                Calculate BRL energy
    """

    
    return BRL_energy