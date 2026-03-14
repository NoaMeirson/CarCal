import numpy as np
import cv2

from models import Point, Polygon


def mask_to_polygon(mask: np.ndarray) -> Polygon | None:
    """
    Converts a binary mask to a polygon using the largest external contour.
    """
    binary_mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 3:
        return None

    points = [
        Point(x=float(point[0][0]), y=float(point[0][1]))
        for point in largest_contour
    ]

    if len(points) < 3:
        return None

    return Polygon(points=points)