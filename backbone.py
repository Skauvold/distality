# Find a backbone for the polygon by first finding the medial axis and then pruning it to a single linestring

import numpy as np

from skimage.morphology import medial_axis
from networkx import all_pairs_shortest_path_length, Graph


def extract_foreground_ij(image):
    """
    Find the pixel indices of the foreground of a binary image.

    Parameters
    ----------
    medial : ndarray
        Medial axis of a binary image.

    Returns
    -------
    eyes : list
        List of row indices of the foreground pixels.
    jays : list
        List of column indices of the foreground pixels.
    """
    # Find the coordinates of the endpoints
    pixel_indices = np.argwhere(image)

    # Convert the coordinates to a list of tuples
    index_tuples = [tuple(coord) for coord in pixel_indices]

    eyes = [i for (i, j) in index_tuples]
    jays = [j for (i, j) in index_tuples]

    return eyes, jays


def create_graph_from_connected_points(eyes, jays):
    """
    Build a graph from the given set of points.

    Nodes represent either endpoints (points with exactly one neighbor),
    or junctions (points with more than two neighbors). The neighborhood
    of a point is defined by the 8-connectivity.

    Two nodes share an edge if there is a direct path between them. The
    weight of the edge is the path length between the two nodes. The path
    length is the sum of the Euclidean distances between adjacent points
    along the path.

    Parameters
    ----------
    eyes : list
        List of row indices of the connected points.
    jays : list
        List of column indices of the connected points.

    Returns
    -------
    graph : networkx.Graph
        Graph of connected points.
    """

    graph = Graph()

    # TODO: Map out the network.
    # - Start at the first point and check out the neighborhood.
    # - Pick a direction to walk in. Store other possible directions in a queue.
    # - When an endpoint or a junction is encountered, add an edge (or node) to the graph.
    # - Continue until all points have been visited (the queue is empty).

    return graph
