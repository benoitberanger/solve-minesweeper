from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize_and_match(image, template, factor):

    w = int(template.shape[1] * factor)
    h = int(template.shape[0] * factor)
    dim = (w, h)

    resized = cv2.resize(template, dim)

    result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)

    return result, dim


def locate_tiles(image, template, scales=np.arange(1,10,0.2), threshold_low=0.90, threshold_high=0.99):
    # small scale test
    for scale in scales:
        result, _ = resize_and_match(image, template, scale)
        map_low = np.argwhere(result > threshold_low)
        if map_low.any():
            dist = scales[1] - scales[0]

            # precise scale test
            for small_scale in np.arange(scale - dist, scale + dist, dist / 10):
                result, dim_high = resize_and_match(image, template, small_scale)
                map_high = np.argwhere(result > threshold_high)
                if map_high.any():
                    return map_high, np.array(dim_high)


def get_good_pos(array):
    array_clean = np.sort(np.unique(array))
    return array_clean




