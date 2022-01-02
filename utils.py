import cv2
import numpy as np

def locate_face(image, template, scales=np.arange(1,10,0.2), threshold_low=0.7, threshold_high=0.90):

    # small scale test
    for scale in scales:
        result = resize_and_match(image, template, scale)
        map_low = np.argwhere(result > threshold_low)
        if map_low.any():
            dist = scales[1]-scales[0]

            # precise scale test
            for small_scale in np.arange(scale-dist, scale+dist, dist/10):
                result = resize_and_match(image, template, small_scale)
                map_high = np.argwhere(result > threshold_high)
                if map_high.any() and map_high.shape[0]==1:
                    return map_high


def resize_and_match(image, template, factor):

    w = int(template.shape[1] * factor)
    h = int(template.shape[0] * factor)
    dim = (w, h)

    resized = cv2.resize(template, dim)

    result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)

    return result