import numpy as np
from PIL import ImageGrab
import cv2
from matplotlib import pyplot as plt


class Zone:
    def __init__(self, center, dim):
        self.center = center
        self.dim    = dim

    def __repr__(self):
        return str(self.__dict__)


class Image(Zone):
    def __init__(self, center=(None, None), dim=(None, None), img_prev=np.array([]), img_curr=np.array([])):
        super().__init__(center, dim)
        self.img_prev = img_prev
        self.img_curr = img_curr

    def show(self):
        plt.imshow(self.img_curr, cmap='gray')
        plt.show()


class Screenshot(Image):
    def __init__(self, center=(None, None), dim=(None, None), img_prev=np.array([]), img_curr=np.array([])):
        super().__init__(center, dim, img_prev, img_curr)

    def take_it(self):

        # get raw image in rgb
        scr = ImageGrab.grab()
        image_rgb = np.array(scr)

        # convert rbg to grayscale
        image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # store previous screenshot & save 'dim' and 'center'
        if self.img_curr.size != 0:
            self.img_prev = self.img_curr
            self.dim = self.img_curr.shape
            self.get_center()
        self.img_curr = image_gs

    def get_center(self):
        self.center = (int(self.dim[0]/2), int(self.dim[1]/2))


class Face(Image):
    def __init__(self, center, dim):
        super().__init__(center, dim)


class Tile(Image):
    def __init__(self, center, dim):
        super().__init__(center, dim)


class Grid:
    def __init__(self):
        self.Zone = Zone
        self.Tiles = np.ndarray

