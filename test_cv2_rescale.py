import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y/%d/%m %I:%M:%S', level=logging.INFO)
logging.info('START')

convert_grayscale = 1

# screenshot
logging.info('Screenshot')
screenshot_all = ImageGrab.grab()
image_rgb = np.array(screenshot_all)
if convert_grayscale:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
else:
    image = image_rgb

# face image recognition
logging.info('Loading "FACE" image')
template_face = utils.load_img('media/face0.png', convert_grayscale)
logging.info('locate_face()')
face_pos = utils.locate_face(image, template_face)

# grid location
logging.info('Loading "TILE" image')
template_tile = utils.load_img('media/t-3.png', convert_grayscale)
logging.info('locate_grid()')
grid_pos = utils.locate_grid(image, template_tile)

