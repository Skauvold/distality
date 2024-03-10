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


def extend_to_boundary(path, image, k=10):
    """
    Extend the path to the edge of the foreground region in the image.

    Parameters
    ----------
    path : list
        List of points in the path. The path is assumed to lie inside the
        foreground region of the image.
    image : ndarray
        Binary image. Foreground pixels are represented by 1s. Bckground
        pixels are represented by 0s. There should be a contiguous region
        of foreground pixels around the path.
    k : int
        Number of path points to use for estimating the direction of the
        path at the boundary.

    Returns
    -------
    extended_path : list
        List of points in the extended path. The order of the pixels in the
        original path is preserved.
    """

    # Get the first k points in the path and Find the extension backward from the start of the path
    first_k_points = path[:k]
    first_k_points.reverse()
    start_extension = extend_locally(first_k_points, image)
    start_extension.reverse()

    # Get the last k points in the path and Find the extension forward from the end of the path
    last_k_points = path[-k:]
    end_extension = extend_locally(last_k_points, image)

    # Assemble the extended path
    extended_path = start_extension + path + end_extension


def extend_locally(path, image):
    """
    Extend a path locally to the edge of the foreground region in the image.

    Parameters
    ----------
    path : list
        List of points in the path. The path is assumed to lie inside the
        foreground region of the image. This should be a short, relatively
        straight segment of the path, with a consistent direction.
    image : ndarray
        Binary image. Foreground pixels are represented by 1s. Bckground
        pixels are represented by 0s. There should be a contiguous region
        of foreground pixels around the path.

    Returns
    -------
    extension : list
        List of points in the extension. The extension does not include
        the points in the original path.
    """

    # Identify the last point in the path. Then, for all other points
    # in the path, compute the normalized displacement vector from
    # the last point. Use the negative average displacement vector as
    # the direction of the extension.
    last_point = path[-1]
    displacements = []
    for point in path[:-1]:
        displacement = (point[0] - last_point[0], point[1] - last_point[1])
        norm = np.linalg.norm(displacement)
        displacements.append((displacement[0] / norm, displacement[1] / norm))

    direction = np.mean(displacements, axis=0) * -1

    # Place integer-coordinate points that are close to the line
    # and next to already-placed points until the boundary of the
    # foreground region is reached.

    boundary_reached = False
    current_point = last_point
    extension = []

    while not boundary_reached:
        # Find the neighbors of the current point
        neighbors = [
            (current_point[0] + i, current_point[1] + j)
            for i in range(-1, 2)
            for j in range(-1, 2)
            if (i, j) != (0, 0)
        ]

        # Filter out neighbors that are outside the image
        neighbors = [
            neighbor
            for neighbor in neighbors
            if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]
        ]

        neighbor_cosines = []
        for neighbor in neighbors:
            displacement = (neighbor[0] - last_point[0], neighbor[1] - last_point[1])
            norm = np.linalg.norm(displacement)
            cosine = np.dot(displacement, direction) / norm
            neighbor_cosines.append(cosine)

        # Choose the neighbor with the largest cosine
        best_neighbor = neighbors[np.argmax(neighbor_cosines)]

        # Add the best neighbor to the extension
        extension.append(best_neighbor)

        # Update the current point
        current_point = best_neighbor

        # Check if the current point is on the boundary
        if image[current_point] == 0:
            boundary_reached = True

    return extension
