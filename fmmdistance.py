import skfmm
import numpy as np


def distance_from_edge(image):
    """
    Compute the distance transform of the input image.

    Parameters
    ----------
    image : ndarray
        Binary image.

    Returns
    -------
    distance : ndarray
        Distance transform of the input image.
    """
    mask = np.logical_not(image)
    phi = np.full_like(image, 1, dtype=float)
    phi[mask] = -1

    distance = skfmm.distance(phi, dx=1)

    return distance


def distance_from_seed_set(image, seed_set):
    """
    Compute the distance transform of the input image from a given set of pixels.

    Parameters
    ----------
    image : ndarray
        Binary image.
    seed_set : list of tuples
        List of (i, j) coordinates of the seed pixels.

    Returns
    -------
    distance : ndarray
        Distance transform of the input image from the given point.
    """
    img_bin = image > 0

    # Check that at least one pixel in the seed set is inside the object
    if not np.any(img_bin[tuple(zip(*seed_set))]):
        raise ValueError(
            "The seed set must contain at least one pixel inside the object."
        )

    start = np.full_like(image, 1, dtype=float)
    for i, j in seed_set:
        start[i, j] = -1

    mask = np.logical_not(img_bin)
    phi = np.ma.masked_array(start, mask)

    distance = skfmm.distance(phi, dx=1)

    return distance
