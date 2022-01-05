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
face.scale_face_ko()

# make sure it's reseted
face.click()

# grid location
grid = Grid()

logging.info('Loading "TILE" image')
grid.load_tile_u('media/t-3.png')
grid.load_tile_t0('media/t0.png')
grid.load_tile_t1('media/t1.png')
grid.load_tile_t2('media/t2.png')
grid.load_tile_t3('media/t3.png')

logging.info('locating grid')
grid.locate(screenshot.img_curr)
grid.scale_tiles()

state = face.check()

quit()
