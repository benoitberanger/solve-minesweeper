import numpy as np
from PIL import ImageGrab
import cv2


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


class Screenshot(Image):
    def __init__(self, center=(None, None), dim=(None, None), img_prev=np.array([]), img_curr=np.array([])):
        super().__init__(center, dim, img_prev, img_curr)

    def take_it(self):
        scr = ImageGrab.grab()
        image_rgb = np.array(scr)
        image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        if self.img_curr.size != 0:
            self.img_prev = self.img_curr
        self.img_curr = image_gs
        self.dim = self.img_curr.shape
        self.get_center()

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

