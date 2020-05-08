''' Adapted from https://github.com/hzwer/ICCV2019-LearningToPaint '''

import cv2
import numpy as np

MAX_ITERATIONS = 100

def normal(x, width):
    return (int)(x * (width - 1) + 0.5)

def draw(f, width=128, max_brush_width=None, opacity=None, max_length=None):
    """ Draw a stroke onto a blank canvas
    Parameters
    ----------
    f : []
        Definition of bezier curve: x0, y0, x1, y1, x2, y2, width_start, width_end, opacity_start, opacity_end
    width : int, optional
        Width and Height of canvas. (Default 128)
    max_brush_width : (int, int), optional
        Override the start and end brush width in f
    opacity : (int, int), optional
        Override the start and end opacity in f
    max_length : int, optional
        Maximum brush stroke length.
    """
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f

    if max_brush_width is not None:
        z0, z2 = max_brush_width
    if opacity is not None:
        w0, w2 = opacity

    frac = 1. / MAX_ITERATIONS
    if max_length is not None:
        curve_length = bezier_curve_length(x0,y0, x1,y1, x2,y2)
        if curve_length > max_length:
            # Don't draw the full curve. Make the increment less by scaling.
            frac = frac * (max_length / curve_length)

    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    
    for i in range(MAX_ITERATIONS):
        t = i * frac
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))


def bezier_curve_length(x0,y0, x1,y1, x2,y2):
    ''' Return the arc length of a bezier curve defined by the given three points
    See https://malczak.linuxpl.com/blog/quadratic-bezier-curve-length/
    '''
    ax = x0 - 2*x1 + x2
    ay = y0 - 2*y1 + y2

    bx = 2*x1 - 2*x0
    by = 2*y1 - 2*y0

    A = max(4*(ax**2 + ay**2), 1e-7)
    B = max(4*(ax*bx + ay*by), 1e-7)
    C = max(bx**2 + by**2, 1e-7)

    Sabc = 2*(A + B + C)**(1/2)
    A_2 = A**(1/2)
    A_32 = 2*A*A_2
    C_2 = 2*C**(1/2)
    BA = B/A_2

    return ( A_32*Sabc + \
          A_2*B*(Sabc-C_2) + \
          (4*C*A-B*B)*np.log( (2*A_2+BA+Sabc)/(BA+C_2) ) \
        )/(4*A_32);