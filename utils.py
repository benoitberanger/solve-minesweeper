from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize(img, factor):
    h = int(img.shape[0] * factor)
    w = int(img.shape[1] * factor)
    new_dim = (h, w)

    new_img = cv2.resize(img, new_dim)
    return new_img, new_dim


def resize_and_match(image, template, factor):

    new_img, new_dim = resize(template, factor)

    result = cv2.matchTemplate(image, new_img, cv2.TM_CCOEFF_NORMED)

    return result, new_img, new_dim


def get_good_pos(array):
    array_clean = np.sort(np.unique(array))
    return array_clean


def image_diff(img1, img2):
    diff    = img1-img2
    squared = diff**2
    summed  = np.sum(squared)
    # don't need to do sqrt
    return summed
