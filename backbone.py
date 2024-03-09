# Find a backbone for the polygon by first finding the medial axis and then pruning it to a single linestring

import numpy as np
from skimage.morphology import medial_axis
from networkx import Graph
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import (
    all_pairs_bellman_ford_path_length,
)

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
    segments: Dict
        Dictionary from source nodes to dictionaries from destination nodes to lists of points along the path.
    """

    graph = Graph()
    segments = {}

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
        node_point, entry_point, node_type, distance_walked, segment = (
            points.walk_to_node(current_point, previous_point)
        )

        # Store the segment
        if previous_point not in segments:
            segments[previous_point] = {}
        segments[previous_point][node_point] = segment

        # Add the node to the graph
        graph.add_node(node_point)

        # Add the edge to the graph
        graph.add_edge(previous_point, node_point, weight=distance_walked)

        # If the node is a junction, add the neighbors to the queue
        if node_type == "junction":
            other_neighbors = points.other_neighbors(node_point, entry_point)
            for neighbor in other_neighbors:
                queue.append((node_point, neighbor))

    return graph, segments


def find_longest_path(graph, segments):
    """
    Find the longest path in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph of connected points.
    segments: Dict
        Dictionary from source nodes to dictionaries from destination nodes to lists of points along the path.

    Returns
    -------
    longest_path_nodes : list
        List of nodes in the longest path.
    longest_path_points: list
        List of points in the longest path.
    """
    # Compute pairwise shortest distances between all nodes in the graph
    distances = dict(all_pairs_bellman_ford_path_length(graph, weight="weight"))

    # Find the longest path
    longest_distance = 0
    longest_path_src_dst = []

    for src, dstlist in distances.items():
        for dst, distance in dstlist.items():
            if distance > longest_distance:
                longest_distance = distance
                longest_path_src_dst = [src, dst]

    # Find the longest path
    longest_path_nodes = list(
        shortest_path(
            graph, source=longest_path_src_dst[0], target=longest_path_src_dst[1]
        )
    )

    # Find the points in the longest path
    longest_path_points = []

    for i in range(len(longest_path_nodes) - 1):
        start_node = longest_path_nodes[i]
        end_node = longest_path_nodes[i + 1]
        if start_node in segments and end_node in segments[start_node]:
            longest_path_points.extend(segments[start_node][end_node])
        elif end_node in segments and start_node in segments[end_node]:
            longest_path_points.extend(segments[end_node][start_node][::-1])
        else:
            raise ValueError("No segment found between nodes")

    return longest_path_nodes, longest_path_points
