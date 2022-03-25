import numpy as np


def bresenham(start, end):
    (x0, y0), (x1, y1) = start, end

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1

    error = dx + dy

    illuminated = []

    while True:
        illuminated.append(np.array([x0, y0]))
        if x0 == x1 and y0 == y1: break
        e2 = 2*error
        
        if e2 >= dy:
            if x0 == x1: break
            error += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1: break
            error += dx
            y0 += sy
    
    return np.array(illuminated)