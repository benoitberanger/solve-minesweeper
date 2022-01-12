import matplotlib.pyplot
import numpy as np
import pyautogui
from PIL import ImageGrab
import cv2
from matplotlib import pyplot as plt
import utils
import logging
import sys

empty_scalar = np.empty([1, 1]).fill(np.nan)


class Point:
    def __init__(self, *args):
        if len(args) == 0:
            self._x = empty_scalar
            self._y = empty_scalar
        elif len(args) == 1:
            self.xy = args[0]
        elif len(args) == 2:
            self.x, self.y = args[0], args[1]
        else:
            raise SyntaxError

    def __repr__(self):
        return f"({self.x},{self.y})"

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.array(value, dtype=np.int)  # force int conversion

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.array(value, dtype=np.int)  # force int conversion

    @property
    def xy(self):
        return np.array([self._x, self._y])

    @xy.setter
    def xy(self, value):
        self.x, self.y = value


class Zone:
    def __init__(self):
        self.dim    = Point()
        self.pos    = Point()
        self.center = Point()

    def __repr__(self):
        return str(self.__dict__)

    def click(self):
        pyautogui.click(self.center.y, self.center.x)


class Image(Zone):
    def __init__(self):
        super().__init__()
        self.img_prev = np.array([])
        self.img_curr = np.array([])

    def show(self):
        plt.imshow(self.img_curr, cmap='gray')
        plt.show()

    def load(self, fname, prop_name):
        img_rgb = cv2.imread(fname)
        img_gs = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        setattr(self, prop_name, img_gs)
        self.dim.xy = img_gs.shape


class Screenshot(Image):
    def __init__(self):
        super().__init__()

    def take_it(self):

        # get raw image in rgb
        scr = ImageGrab.grab()
        image_rgb = np.array(scr)

        # convert rbg to grayscale
        image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # store previous screenshot & save 'dim' and 'center'
        if self.img_curr.size == 0:
            self.dim.xy = image_gs.shape
            self.get_center()
        else:
            self.img_prev = self.img_curr
        self.img_curr = image_gs

    def get_center(self):
        self.center.xy = self.dim.xy/2


class Face(Image):
    def __init__(self):
        super().__init__()
        self.fname_face_ok = ''
        self.fname_face_ko = ''
        self.img_face_ok = np.array([])
        self.img_face_ko = np.array([])
        self.dim_scaled = Point()
        self.img_face_ok_scaled = np.array([])
        self.img_face_ko_scaled = np.array([])

    def load_face_ok(self, fname):
        self.fname_face_ok = fname
        self.load(fname, 'img_face_ok')

    def load_face_ko(self, fname):
        self.fname_face_ko = fname
        self.load(fname, 'img_face_ko')

    def locate(self, img_screenshot, scales=np.arange(1,10,0.2), threshold_low=0.7, threshold_high=0.90):

        # small scale test
        for scale in scales:
            result, _, _ = utils.resize_and_match(img_screenshot, self.img_face_ok, scale)
            map_low = np.argwhere(result > threshold_low)
            if map_low.any():
                logging.info('roughly found "face_ok"')
                dist = scales[1] - scales[0]

                # precise scale test
                for small_scale in np.arange(scale - dist, scale + dist, dist / 10):
                    result, template_high, dim_high = utils.resize_and_match(img_screenshot, self.img_face_ok, small_scale)
                    map_high = np.argwhere(result > threshold_high)
                    if map_high.any() and map_high.shape[0] == 1:
                        logging.info('precisely found "face_ok"')
                        self.pos.xy = map_high[0]
                        self.center.xy = map_high[0] + np.array(dim_high)/2
                        self.img_face_ok_scaled = template_high
                        self.dim_scaled.xy = dim_high
                        return
                logging.warning('**NOT** precisely found "face_ok"')
                sys.exit()
        logging.warning('**NOT** roughly found "face_ok"')
        sys.exit()

    def scale_face_ko(self):
        self.img_face_ko_scaled = cv2.resize(self.img_face_ko, self.dim_scaled.xy)

    def capture(self):

        # get raw image in rgb
        bbox = (self.pos.y,
                self.pos.x,
                self.pos.y+self.dim_scaled.y,
                self.pos.x+self.dim_scaled.x)
        scr = ImageGrab.grab(bbox=bbox)
        image_rgb = np.array(scr)

        # convert rbg to grayscale
        image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # store previous screenshot & save 'dim' and 'center'
        if self.img_curr.size != 0:
            self.img_prev = self.img_curr
        self.img_curr = image_gs

    def check(self):
        self.capture()
        is_ok = utils.image_diff(self.img_curr, self.img_face_ok_scaled)
        is_ko = utils.image_diff(self.img_curr, self.img_face_ko_scaled)
        return is_ok > is_ko


class Tile(Image):
    def __init__(self):
        super().__init__()


class Grid(Image):
    def __init__(self):
        super().__init__()

        self.grid_size_h = empty_scalar
        self.grid_size_w = empty_scalar

        self.list_sprite = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8',
                            'b', 'u', 'f',
                            ]

        for sprite in self.list_sprite:
            setattr(self, f'fname_tile_{sprite}', '')
            setattr(self, f'img_tile_{sprite}', np.ndarray([]))
            setattr(self, f'img_tile_scaled_{sprite}', np.ndarray([]))

        self.dim_scaled = Point()

        self.tiles_prev_state = np.ndarray([])
        self.tiles_curr_state = np.ndarray([])
        self.tiles_fov = (Point(), Point())
        self.tiles_pos = np.ndarray([])

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def __load_tile(fname):
        img_rgb = cv2.imread(fname)
        img_tile = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        return img_tile

    def load_tile(self, fname, field):
        setattr(self, f'fname_tile_{field}', fname)
        setattr(self, f'img_tile_{field}', self.__load_tile(fname))

    def locate(self, img_screenshot, scales=np.arange(1, 10, 0.2), threshold_low=0.90, threshold_high=0.99):

        # small scale test
        for scale in scales:
            result, _, _ = utils.resize_and_match(img_screenshot, self.img_tile_u, scale)
            map_low = np.argwhere(result > threshold_low)
            if map_low.any():
                logging.info('roughly found "tile"')
                dist = scales[1] - scales[0]

                # precise scale test
                for small_scale in np.arange(scale - dist, scale + dist, dist / 10):
                    result, template_high, dim_high = utils.resize_and_match(img_screenshot, self.img_tile_u, small_scale)
                    map_high = np.argwhere(result > threshold_high)
                    if map_high.any():

                        logging.info('precisely found "tile"')

                        self.img_tile_scaled_u = template_high
                        self.dim_scaled.xy = dim_high

                        tiles_pos = map_high
                        tile_dim = np.array(dim_high)

                        tiles_h = tiles_pos[:, 0]
                        tiles_w = tiles_pos[:, 1]

                        tiles_h_clean = utils.get_good_pos(tiles_h)
                        tiles_w_clean = utils.get_good_pos(tiles_w)
                        self.grid_size_h = len(tiles_h_clean)
                        self.grid_size_w = len(tiles_w_clean)
                        logging.info(f'grid size = ({self.grid_size_h},{self.grid_size_w})')

                        self.tiles_prev_state = np.empty([self.grid_size_h, self.grid_size_w]).fill(np.nan)
                        self.tiles_curr_state = self.tiles_prev_state
                        self.tiles_fov[0].xy = (tiles_h_clean[ 0], tiles_w_clean[ 0])
                        self.tiles_fov[1].xy = (tiles_h_clean[-1], tiles_w_clean[-1]) + tile_dim

                        self.tiles_pos = np.ndarray([self.grid_size_h, self.grid_size_w], dtype=object)
                        for i in range(self.grid_size_h):
                            for j in range(self.grid_size_w):
                                self.tiles_pos[i][j] = Point( (tiles_h_clean[i], tiles_w_clean[j]) + tile_dim / 2 )

                        return

                logging.warning('**NOT** precisely found "tile"')
                sys.exit()

        logging.warning('**NOT** roughly found "tile"')
        sys.exit()

    def scale_tiles(self):
        self.img_tile_scaled_t0 = cv2.resize(self.img_tile_t0, self.dim_scaled.xy)
        self.img_tile_scaled_t1 = cv2.resize(self.img_tile_t1, self.dim_scaled.xy)
        self.img_tile_scaled_t2 = cv2.resize(self.img_tile_t2, self.dim_scaled.xy)
        self.img_tile_scaled_t3 = cv2.resize(self.img_tile_t3, self.dim_scaled.xy)

    def capture(self):

        # get raw image in rgb
        bbox = (self.tiles_fov[0].y,
                self.tiles_fov[0].x,
                self.tiles_fov[1].y,
                self.tiles_fov[1].x)
        scr = ImageGrab.grab(bbox=bbox)
        image_rgb = np.array(scr)

        # convert rbg to grayscale
        image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # store previous screenshot & save 'dim' and 'center'
        if self.img_curr.size != 0:
            self.img_prev = self.img_curr
        self.img_curr = image_gs

    def click(self, idx_x, idx_y):
        pos = self.tiles_pos[idx_x, idx_y]
        pyautogui.click(pos.y, pos.x)

    def analyze(self):

        self.img_curr
