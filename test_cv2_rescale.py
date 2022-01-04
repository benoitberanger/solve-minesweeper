import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
import logging
from classes import *

logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y/%d/%m %I:%M:%S', level=logging.INFO)
logging.info('START')

logging.info('Screenshot')
screenshot = Screenshot()
screenshot.take_it()

face = Face()
logging.info('Loading "FACE" image')
face.load_face_ok('media/face0.png')
face.load_face_ko('media/face2.png')
logging.info('locating face')
face.locate(screenshot.img_curr)


quit()

# reset -> click on th face
pyautogui.leftClick(face_pos[1], face_pos[0])
image = utils.take_screenshot(convert_grayscale)  # reset input image

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
        tiles_pos_2d[i][j] = (tiles_h_clean[i], tiles_w_clean[j]) + tile_dim/2

# reset -> click on th face
pyautogui.leftClick(face_pos[1], face_pos[0])

# click in the center tile
idx_middle_h = round(grid_size_h/2)
idx_middle_w = round(grid_size_w/2)
pyautogui.leftClick(tiles_pos_2d[idx_middle_h][idx_middle_w][1], tiles_pos_2d[idx_middle_h][idx_middle_w][0])
