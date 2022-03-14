import numpy as np
from scipy.ndimage import rotate

def read_pixels_on_a_line(img: np.array, line: np.array) -> np.array:
    return np.array([img[point_y, point_x] for point_x, point_y in line])

def make_circle(sinogram: np.array, alpha: float) -> np.array:
    diameter = len(sinogram[0])
    circle = np.zeros((diameter, diameter), dtype= 'complex_')
    
    for row in sinogram:
        circle[diameter//2] = row
        circle = rotate(circle, alpha, reshape=False)
    return circle