import numpy as np


class Zone:
    def __init__(self, center, dim):
        self.center = center
        self.dim    = dim

    def __repr__(self):
        return f"<{__name__}.{self.__class__.__name__}: center=({self.center}), dim = ({self.dim})>"


class Image(Zone):
    def __init__(self, center, dim):
        super().__init__(center, dim)
        self.Array = np.ndarray()


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

