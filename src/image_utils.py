import os
import skimage
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray

def resize_image(img, image_size=(320, 320)):
    """
    ---------
    Arguments
    ---------
    img : ndarray
        ndarray of shape (H, W, 3) or (H, W) i.e. RGB and grayscale respectively
    image_size : tuple of ints
        image size to be used for resizing

    -------
    Returns
    -------
    resized image ndarray of shape (H_resized, W_resized, 3) or (H_resized, W_resized)
    if RGB returns resized image ndarray in range [0, 255]
    if grasycale returns resized image ndarray in range [0, 1]
    """
    img_resized = resize(img, image_size)
    return img_resized

def convert_rgb2gray(img_rgb):
    """
    ---------
    Arguments
    ---------
    img_rgb : ndarray
        ndarray of shape (H, W, 3) i.e. RGB

    -------
    Returns
    -------
    grayscale image ndarray of shape (H, W)
    """
    img_gray = rgb2gray(img_rgb)
    return img_gray

def convert_lab2rgb(img_lab):
    """
    ---------
    Arguments
    ---------
    img_lab : ndarray
        ndarray of shape (H, W, 3) i.e. Lab

    -------
    Returns
    -------
    RGB image ndarray of shape (H, W, 3) i.e. RGB space
    """
    img_rgb = lab2rgb(img_lab)
    return img_rgb

def convert_rgb2lab(img_rgb):
    """
    ---------
    Arguments
    ---------
    img_rgb : ndarray
        ndarray of shape (H, W, 3) i.e. RGB

    -------
    Returns
    -------
    Lab image ndarray of shape (H, W, 3) i.e. Lab space
    """
    img_lab = rgb2lab(img_rgb)
    return img_lab

def apply_image_ab_post_processing(img_ab):
    """
    ---------
    Arguments
    ---------
    img_ab : ndarray
        pre-processed ndarray of shape (H, W, 2) i.e. ab channels in Lab space in range [-1, 1]

    -------
    Returns
    -------
    post-processed ab channels ndarray of shape (H, W, 2) in range [-110, 110]
    """
    img_ab = img_ab * 110.
    return img_ab

def apply_image_l_pre_processing(img_l):
    """
    ---------
    Arguments
    ---------
    img_l : ndarray
        ndarray of shape (H, W) i.e. L channel in Lab space in range [0, 100]

    -------
    Returns
    -------
    pre-processed L channel ndarray of shape (H, W) in range [-1, 1]
    """
    img_l = (img_l / 50.) - 1
    return img_l

def apply_image_ab_pre_processing(img_ab):
    """
    ---------
    Arguments
    ---------
    img_ab : ndarray
        ndarray of shape (H, W, 2) i.e. ab channels in Lab space in range [-110, 110]

    -------
    Returns
    -------
    pre-processed ab channels ndarray of shape (H, W, 2) in range [-1, 1]
    """
    img_ab = (img_ab) / 110.
    return img_ab

def concat_images_l_ab(img_l, img_ab):
    """
    ---------
    Arguments
    ---------
    img_l : ndarray
        ndarray of shape (H, W, 1) i.e. L channel
    img_ab : ndarray
        ndarray of shape (H, W, 2) i.e. ab channels

    -------
    Returns
    -------
    Lab space ndarray of shape (H, W, 3)
    """
    img_lab = np.concatenate((img_l, img_ab), axis=-1)
    return img_lab

def read_image(file_img):
    """
    ---------
    Arguments
    ---------
    file_img : str
        full path of the image

    -------
    Returns
    -------
    ndarray of shape (H, W, 3) for RGB or (H, W) for grayscale
    """
    img = imread(file_img)
    return img

def save_image_rgb(file_img, img_arr):
    """
    ---------
    Arguments
    ---------
    file_img : str
        full path of the image
    img_arr : ndarray
        image ndarray to be saved, of shape (H, W, 3) for RGB or (H, W) for grasycale
    """
    imsave(file_img, img_arr)
    return

def rescale_grayscale_image_l_channel(img_gray):
    """
    ---------
    Arguments
    ---------
    img_gray : ndarray
        grayscale image of shape (H, W) in range [0, 1]

    -------
    Returns
    -------
    L channel ndarray of shape (H, W) in range [0, 100]
    """
    img_l_rescaled = (img_gray) * 100.
    return img_l_rescaled
