import numpy as np


def region_to_bbox(region, center=True):
    """Take a quadrangle region, and return a bounding box.

    Args:
        region (4- or 8-tuple of floats): Either a bounding box of the form (x, y, w, h) where x, y are the coordinates
         of the upper left corner of the bounding box, or a quadrangle defined by the corners (x1, y1, ..., x4, y4).
        center (boolean): If true, bounding box is returned in (x, y, w, h) format, otherwise, it is returned in
        (cx, cy, w, h) format, where cx, and cy are the coordinates of the center of the bounding box, not its top-left
        corner.

    Returns:
        (int, int, int, int): Bounding box in either (x, y, w, h) format for (cx, cy, w, h) format.
    """

    n = len(region)
    assert n == 4 or n == 8, 'GT region format is invalid, should have 4 or 8 entries.'

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


def _rect(region, center):
    
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return cx, cy, w, h
    else:
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    area1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    area2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(area1 / area2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx - w / 2, cy - h / 2, w, h
