# Find a backbone for the polygon by first finding the medial axis and then pruning it to a single linestring

import numpy as np
from skimage.morphology import medial_axis
from networkx import all_pairs_shortest_path_length, Graph

from pointutils import IndexPointCollection


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

    # Prepare the points
    points = IndexPointCollection(eyes, jays)

    # Initialize the queue that will hold pairs of starting points and directions
    queue = []

    # Find starting point
    start_point = points.points[0]

    # Find neighbors of the starting point
    neighbors = points.neighbors(start_point)

    # Add an entry to the queue for each neighbor of the starting point
    for neighbor in neighbors:
        queue.append((start_point, neighbor))

    # Add starting point to the graph
    graph.add_node(start_point)

    while queue:
        # Get the next pair of points from the queue
        previous_point, current_point = queue.pop(0)

        # Walk until a node (endpoint or junction) is encountered
        node_point, entry_point, node_type, distance_walked = points.walk_to_node(
            current_point, previous_point
        )

        # Add the node to the graph
        graph.add_node(node_point)

        # Add the edge to the graph
        graph.add_edge(previous_point, node_point, weight=distance_walked)

        # If the node is a junction, add the neighbors to the queue
        if node_type == "junction":
            other_neighbors = points.other_neighbors(node_point, entry_point)
            for neighbor in other_neighbors:
                queue.append((node_point, neighbor))

    return graph
