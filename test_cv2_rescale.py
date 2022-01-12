import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
import logging
from classes import *

# logger
logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%Y/%d/%m %I:%M:%S',
    level=logging.INFO,
)

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
grid.load_tile('media/t0.png', 't0')
grid.load_tile('media/t1.png', 't1')
grid.load_tile('media/t2.png', 't2')
grid.load_tile('media/t3.png', 't3')
grid.load_tile('media/t4.png', 't4')
grid.load_tile('media/t5.png', 't5')
grid.load_tile('media/t6.png', 't6')
grid.load_tile('media/t7.png', 't7')
grid.load_tile('media/t8.png', 't8')
grid.load_tile('media/t-1.png', 'b')
grid.load_tile('media/t-3.png', 'u')
grid.load_tile('media/t-4.png', 'f')

logging.info('locating grid')
grid.locate(screenshot.img_curr)
grid.scale_tiles()
grid.capture()

# lets roll
face.click()
grid.click(
    round(grid.grid_size_h/2),
    round(grid.grid_size_w/2),
)
pyautogui.moveTo(0, 0)
if not face.check():
    logging.error('lol, first random click we died...')
    quit()
grid.capture()
grid.analyze()
print(grid.tiles_curr_state)

quit()
