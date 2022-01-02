import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils

convert_grayscle = 1

# all screen image
screenshot_all = ImageGrab.grab()
image_rgb = np.array(screenshot_all)

if convert_grayscle:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
else:
    image = image_rgb

# load template image
# template_rgb = cv2.imread('media/t-3.png')
template_rgb = cv2.imread('media/face0.png')
if convert_grayscle:
    template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
else:
    template = template_rgb
# plt.imshow(template)
# plt.show()

face_pos = utils.locate_face(image, template)
