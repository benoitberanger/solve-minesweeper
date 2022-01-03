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
face_pos, face_dim = utils.locate_face(image, template_face)

# grid location
logging.info('Loading "TILE" image')
template_tile = utils.load_img('media/t-3.png', convert_grayscale)
logging.info('locate_grid()')
tiles_pos, tile_dim = utils.locate_tiles(image, template_tile)

tiles_h = tiles_pos[:, 0]
tiles_w = tiles_pos[:, 1]

tiles_h_clean = utils.get_good_pos(tiles_h)
tiles_w_clean = utils.get_good_pos(tiles_w)
grid_size_h = len(tiles_h_clean)
grid_size_w = len(tiles_w_clean)
logging.info(f'grid size = ({grid_size_h},{grid_size_w})')

tiles_pos_2d = np.ndarray((grid_size_h,grid_size_w), dtype=object)
for i in range(grid_size_h):
    for j in range(grid_size_w):
        tiles_pos_2d[i][j] = (tiles_h_clean[i], tiles_w_clean[j])
0
