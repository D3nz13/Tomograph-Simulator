import numpy as np


def illuminate_low(start, end):
    (x0, y0), (x1, y1) = start, end

    dx, dy = x1 - x0, y1 - y0
    yi = 1

    if dy < 0:
        y1 = -1
        dy = -dy
    
    D = 2*dy - dx
    y = y0

    illuminated_points = []

    for x in range(x0, x1+1):
        illuminated_points.append(np.array([x, y]))
        if D > 0:
            y += yi
            D += 2*(dy - dx)
        else:
            D += 2*dy
    
    return np.array(illuminated_points)


def illuminate_high(start, end):
    (x0, y0), (x1, y1) = start, end

    dx, dy = x1 - x0, y1 - y0
    xi = 1

    if dx < 0:
        xi = -1
        dx = -dx
    
    D = 2*dx - dy
    x = x0

    illuminated_points = []

    for y in range(y0, y1+1):
        illuminated_points.append(np.array([x, y]))
        if D > 0:
            x += xi
            D += 2*(dx - dy)
        else:
            D += 2*dx
    
    return np.array(illuminated_points)


def bresenham(start, end):
    (x0, y0), (x1, y1) = start, end

    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return illuminate_low(end, start)
        else:
            return illuminate_low(start, end)
    else:
        if y0 > y1:
            return illuminate_high(end, start)
        else:
            return illuminate_high(start, end)