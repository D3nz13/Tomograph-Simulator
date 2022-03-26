import numpy as np
from scipy.ndimage import rotate

def read_pixels_on_a_line(img: np.ndarray, line: np.ndarray) -> np.ndarray:
    return np.array([img[point_y, point_x] for point_x, point_y in line])


def make_circle(sinogram: np.ndarray, alpha: float) -> np.ndarray:
    diameter = len(sinogram[0])
    circle = np.zeros((diameter, diameter), dtype= 'complex_')
    
    for row in sinogram:
        circle[diameter//2] = row
        circle = rotate(circle, alpha, reshape=False)
    return circle


def create_kernel(length: int) -> np.ndarray:
    right = [1]
    for i in range(1, length):
        if i%2 == 0:
            right.append(0)
        else:
            right.append(-4/(np.pi**2*i**2))
    
    left = right[1::][::-1]
    left.extend(right)

    return np.array(left)
