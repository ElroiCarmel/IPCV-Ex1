"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208762971


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap='grey')
    else:
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space in YIQ range
    """
    rgb_to_yiq = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3, -1)
    img_yiq = imgRGB.reshape(-1, 3)
    img_yiq = img_yiq.dot(rgb_to_yiq.T).reshape(imgRGB.shape)
    return img_yiq


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_to_rgb = np.array([1, 0.956, 0.619, 1, -0.272, -0.647, 1, -1.106, 1.703]).reshape(3, -1)
    img_rgb = imgYIQ.reshape(-1, 3)
    img_rgb = img_rgb.dot(yiq_to_rgb.T).reshape(imgYIQ.shape)
    return img_rgb


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image in grayscale or RGB colorspace having values in range [0,1]
    :return (imgEq,histOrg,histEQ)
    """
    if len(imgOrig.shape) == 2:  # If it's GREYSCALE
        img_cpy = np.round(imgOrig.copy() * 255).flatten().astype(np.uint8)
        org_shape = imgOrig.shape
        hist_org = np.histogram(img_cpy, bins=256, range=(0, 256))[0].astype(int)
        lut = np.cumsum(hist_org) / imgOrig.size
        lut = np.ceil(lut * 255).astype(int)
        img_eq = lut[img_cpy]
        hist_eq = np.histogram(img_eq.copy(), bins=256, range=(0, 256))[0].astype(int)
        img_eq = (img_eq / 255.0).reshape(org_shape)
        return img_eq, hist_org, hist_eq
    else:  # If it's RGB
        img_yiq = transformRGB2YIQ(imgOrig)
        y = np.round(np.multiply(img_yiq[:, :, 0], 255)).astype(np.uint8).flatten()
        y_hist = np.histogram(y, bins=256, range=(0, 256))[0].astype(int)
        cdf = np.cumsum(y_hist)
        cdf_norm = cdf / y.size
        lut = np.ceil(np.multiply(cdf_norm, 255)).astype(np.uint8)
        y_eq = lut[y]
        y_eq_hist = np.histogram(y_eq, bins=256, range=(0, 256))[0].astype(int)
        y_eq = y_eq.reshape(imgOrig.shape[:2])
        y_eq_norm = y_eq / 255.0
        print(y_eq_norm)
        img_yiq[:, :, 0] = y_eq_norm
        img_rgb = transformYIQ2RGB(img_yiq)
        return img_rgb, y_hist, y_eq_hist


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
