import numpy as np


def read_pixels_on_a_line(img: np.array, line: np.array) -> np.array:
    return np.array([img[point_y, point_x] for point_x, point_y in line])