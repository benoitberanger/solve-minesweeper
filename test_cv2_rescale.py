import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
import logging
from classes import *

# logger
logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y/%d/%m %I:%M:%S', level=logging.INFO)

# screenshot of the whole monitor
logging.info('1st Screenshot')
screenshot = Screenshot()
screenshot.take_it()

# face location
face = Face()

logging.info('Loading "FACE" image')
face.load_face_ok('media/face0.png')
face.load_face_ko('media/face2.png')

logging.info('locating face')
face.locate(screenshot.img_curr)

# make sure it's reseted
face.click()

# grid location
grid = Grid()

logging.info('Loading "TILE" image')
grid.load_tile('media/t-3.png')

logging.info('locating grid')
grid.locate(screenshot.img_curr)

quit()

# reset -> click on th face
pyautogui.leftClick(face_pos[1], face_pos[0])

# click in the center tile
idx_middle_h = round(grid_size_h/2)
idx_middle_w = round(grid_size_w/2)
pyautogui.leftClick(tiles_pos_2d[idx_middle_h][idx_middle_w][1], tiles_pos_2d[idx_middle_h][idx_middle_w][0])
