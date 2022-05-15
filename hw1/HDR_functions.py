''' Functions in HDR flow '''

import os
from re import I
import cv2 as cv
import numpy as np

Z = 256  # intensity levels
Z_max = 255
Z_min = 0
gamma = 2.2


def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = np.transpose(img, (2,0,1))
    return img


def SaveImg(img, path):
    img = np.transpose(img, (1,2,0))
    cv.imwrite(path, img)
    
    
def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, ch, height, width)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]

    ''' Load private test images '''
    # filenames = ['IMG_3098.JPG', 'IMG_3099.JPG', 'IMG_3100.JPG', 'IMG_3101.JPG', 'IMG_3102.JPG', 'IMG_3103.JPG']
    # exposure_times = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, ch, height, width))
        
    Returns:
        sample (uint8 ndarray, shape (N, ch, height_sample_size, width_sample_size))
    """
    # trivial periodic sample
    sample = img_list[:, :, ::64, ::64]
    
    return sample


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    
    ''' TODO '''
    # Matrix initialize
    middle = int(0.5*(255+0))
    i, j = img_samples.shape[1]*img_samples.shape[2], img_samples.shape[0]
    M, N = i*j+1+254, 256+i
    A_mat = np.zeros((M, N))
    b_mat = np.zeros((M, 1))
    wij = np.arange(256)
    wij[middle+1:] = 255 - wij[middle+1:]

    inp = img_samples.reshape(-1)
    # k = 0 
    # img_samples = img_samples.reshape(j, i)
    for k in range(inp.shape[0]) :
            # Fill Data-term
            A_mat[k, inp[k]] = 1 * wij[inp[k]]
            A_mat[k, 256+k%i] = -1 * wij[inp[k]]
            # Measurements
            b_mat[k] = wij[inp[k]] * np.log(etime_list[k//i])

    """Why the value=1 here?????"""
    A_mat[i*j, middle] = 1

    for k in range(254) :
        # Smoothness
        A_mat[i*j+k+1, k] = 1 * lambda_ * wij[k+1]
        A_mat[i*j+k+1, k+1] = -2 * lambda_ * wij[k+1]
        A_mat[i*j+k+1, k+2] = 1 * lambda_ * wij[k+1]
    
    response = np.linalg.lstsq(A_mat, b_mat, rcond=None)[0][:256].squeeze().astype('float32')

    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    ''' TODO '''
    g_ij = response[img_list]
    e_ij = np.kron(np.log(etime_list)[:, None, None], np.ones((img_list.shape[1], img_list.shape[2])))
    wij = np.arange(256)
    wij[128:] = 255 - wij[128:]
    radiance = ((wij[img_list]+1e-5) * (g_ij - e_ij)).sum(0) / (wij[img_list]+1e-5).sum(0)
    radiance = np.exp(radiance)

    return radiance


def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:,ch,:,:], exposure_times, lambda_)
        radiance[ch,:,:] = ConstructRadiance(img_list[:,ch,:,:], response, exposure_times)
        
    return radiance


def WhiteBalance(src, y_range, x_range):
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (ch, height, width))
    """
   
    ''' TODO '''
    avg = src[:, y_range[0]:y_range[1], x_range[0]:x_range[1]].reshape(3, -1).mean(1)       # B, G, R
    src[1, :, :] = src[1, :, :] * (avg[2] / avg[1])         # R_avg / G_avg
    src[0, :, :] = src[0, :, :] * (avg[2] / avg[0])         # R_avg / B_avg



    return src


def GlobalTM(src, scale=1.0):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    gamma = 2.2
    X_max = src.reshape(3, -1).max(1)
    X_head = scale * (np.log2(src) - np.log2(X_max)[:, None, None]) + np.log2(X_max)[:, None, None]
    X_head = np.power(2, X_head)
    result = np.power(X_head, 1/gamma)
    # result = cv.normalize(result, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    """Clip to transform in (0, 255)"""
    result[result<0] = 0
    result[result>1] = 1
    result = result*255
    result = result.astype('uint8')
    return result


def LocalTM(src, imgFilter, scale=3.0):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    # Separate Intensity map
    I_map = src.mean(0)
    C_ratio = src / I_map
    L_map = np.log2(I_map)

    # Separate the detail layer
    L_base = imgFilter(L_map)
    L_detail = L_map - L_base

    # Compress the contrast 
    L_base_prime = (L_base - L_base.max()) * scale / (L_base.max() - L_base.min())
    # Reconstruct intensity map
    I_map_prime = 2**(L_base_prime+L_detail)
    C_out = C_ratio * I_map_prime

    # Transform to 0-255
    C_out = C_out**(1/gamma)
    C_out[C_out<0] = 0 
    C_out[C_out>1] = 1 
    result = C_out*255
    result = result.astype('uint8')

    return result


def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    w_X,w_Y = np.mgrid[-(N//2):N//2+1:1, -(N//2):N//2+1:1]
    weight = np.exp(-((w_X**2 + w_Y**2) / (2*(sigma_s**2))))
    src_pad = np.pad(src, ((N//2, N//2), (N//2, N//2)), 'symmetric')

    result = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            conv_map = src_pad[i:i+N, j:j+N] * weight    # perform element-wise product
            result[i, j] = conv_map.sum() / weight.sum()
            
    result = result.astype('float32')
    
    return result


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    w_X,w_Y = np.mgrid[-(N//2):N//2+1:1, -(N//2):N//2+1:1]
    weight = np.exp(-((w_X**2 + w_Y**2) / (2*(sigma_s**2))))
    src_pad = np.pad(src, ((N//2, N//2), (N//2, N//2)), 'symmetric')
    
    result = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            w_ij = weight * np.exp(-(src[i, j] - src_pad[i:i+N, j:j+N])**2/(2*(sigma_r**2)))
            conv_map = src_pad[i:i+N, j:j+N] * w_ij    # perform element-wise product
            result[i, j] = conv_map.sum() / w_ij.sum()
            
    result = result.astype('float32')
            
    return result
