from skimage.io import imread
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import medial_axis
from networkx.algorithms.shortest_paths.weighted import (
    all_pairs_bellman_ford_path_length,
)

from backbone import (
    extract_foreground_ij,
    create_graph_from_connected_points,
    find_longest_path,
    extend_to_boundary,
)

path_to_image = Path("data/01_blob.png")

original = imread(path_to_image)
img = original[:, :, 0]


# Step 1: Compute medial axis
medial = medial_axis(img)

# Plot the original image and the medial axis
fig, axs = plt.subplots(1, 7, figsize=(18, 8))

axs[0].imshow(img, cmap="gray")
axs[0].axis("off")
axs[0].set_title("Original")

axs[1].imshow(medial, cmap="gray")
axs[1].axis("off")
axs[1].set_title("Medial axis (pixels)")

# Step 2: Extract medial axis points
i_medial, j_medial = extract_foreground_ij(medial)

# Plot the endpoints in a separate figure
axs[2].imshow(img, cmap="gray")
axs[2].axis("off")
axs[2].set_title("Medial axis (points)")
axs[2].plot(j_medial, i_medial, "ro")


# Step 3: Make a graph of medial axis endpoints and junctions
# and the distances between them
graph, segments = create_graph_from_connected_points(i_medial, j_medial)

# Visualize the graph using the networkx library
axs[3].set_title("Graph of connected points")
pos = nx.spring_layout(graph)
nx.draw(
    graph,
    pos=pos,
    ax=axs[3],
    with_labels=True,
    font_weight="bold",
    node_size=1000,
    node_color="skyblue",
)

# Display weigths on the edges
edge_labels = {(u, v): round(d["weight"], 2) for u, v, d in graph.edges(data=True)}
nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=axs[3])


# indicate the node locations on the original image
axs[4].imshow(img, cmap="gray")
axs[4].axis("off")
axs[4].set_title("Nodes")
axs[4].plot(j_medial, i_medial, "ro")

# Get the node coordinates from the graph
node_is = [node._row_index for node in graph.nodes]
node_js = [node._col_index for node in graph.nodes]

# Plot the node coordinates on the original image
axs[4].plot(node_js, node_is, "bo")

# Step 4: Compute pairwise shortest distances between all nodes in the graph
# Step 5: Identify the endpoints with the longest distance between them
# Step 6: Form the backbone from the coordinates of the longest path
longest_path_nodes, longest_path_points = find_longest_path(graph, segments)

# Visualize the longest path
axs[5].imshow(img, cmap="gray")
axs[5].axis("off")
axs[5].set_title("Longest path")
axs[5].plot(j_medial, i_medial, "ro")

# Plot the longest path on the original image
for i in range(len(longest_path_nodes) - 1):
    start = longest_path_nodes[i]
    end = longest_path_nodes[i + 1]
    axs[5].plot(
        [start._col_index, end._col_index],
        [start._row_index, end._row_index],
        color="cyan",
    )

# Visualize the longest path
axs[6].imshow(img, cmap="gray")
axs[6].axis("off")
axs[6].set_title("Longest path points")
axs[6].plot(j_medial, i_medial, "ro")

# Plot the longest path on the original image
i_path = [point._row_index for point in longest_path_points]
j_path = [point._col_index for point in longest_path_points]
axs[6].plot(j_path, i_path, color="cyan")

# Plot the first and last points in the path in a different color
axs[6].plot(j_path[0], i_path[0], "s", color="lightgreen")
axs[6].plot(j_path[-1], i_path[-1], "s", color="pink")

# Step 7: Extend the longest path to the boundary of the image
full_backbone = extend_to_boundary(longest_path_points, img)

# Visualize the extended path
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")

# Plot the longest path on the original image
i_path = [i for (i, j) in full_backbone]
j_path = [j for (i, j) in full_backbone]
ax.plot(j_path, i_path, color="green")

# Plot the first and last points in the path in a different color
ax.plot(j_path[0], i_path[0], "s", color="lightgreen")
ax.plot(j_path[-1], i_path[-1], "s", color="pink")

plt.show()
