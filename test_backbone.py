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
    collect_points_along_path,
)

path_to_image = Path("data/01_blob.png")
original = imread(path_to_image)
img = original[:, :, 0]


# Step 1: Compute medial axis
medial = medial_axis(img)

# Plot the original image and the medial axis
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img, cmap="gray")
axs[0].axis("off")
axs[0].set_title("Original")

axs[1].imshow(medial, cmap="gray")
axs[1].axis("off")
axs[1].set_title("Medial axis")

# Step 2: Extract medial axis points
i_medial, j_medial = extract_foreground_ij(medial)

# Plot the endpoints in a separate figure
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.axis("off")
ax.set_title("Original")
ax.plot(j_medial, i_medial, "ro")


# Step 3: Make a graph of medial axis endpoints and junctions
# and the distances between them
graph = create_graph_from_connected_points(i_medial, j_medial)

# Visualize the graph using the networkx library
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title("Graph of connected points")
pos = nx.spring_layout(graph)
nx.draw(
    graph,
    pos=pos,
    ax=axs[0],
    with_labels=True,
    font_weight="bold",
    node_size=1000,
    node_color="skyblue",
)

# Display weigths on the edges
edge_labels = {(u, v): round(d["weight"], 2) for u, v, d in graph.edges(data=True)}
nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=axs[0])


# indicate the node locations on the original image
axs[1].imshow(img, cmap="gray")
axs[1].axis("off")
axs[1].set_title("Original")
axs[1].plot(j_medial, i_medial, "ro")

# Get the node coordinates from the graph
node_is = [node._row_index for node in graph.nodes]
node_js = [node._col_index for node in graph.nodes]

# Plot the node coordinates on the original image
axs[1].plot(node_js, node_is, "bo")

# Step 4: Compute pairwise shortest distances between all nodes in the graph
# Step 5: Identify the endpoints with the longest distance between them
longest_path = find_longest_path(graph)

# Visualize the longest path
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.axis("off")
ax.set_title("Original")
ax.plot(j_medial, i_medial, "ro")

# Plot the longest path on the original image
for i in range(len(longest_path) - 1):
    start = longest_path[i]
    end = longest_path[i + 1]
    ax.plot([start._col_index, end._col_index], [start._row_index, end._row_index], "g")


# Step 6: Extend the longest path to the boundary of the image
points_on_path = collect_points_along_path(
    i_medial, j_medial, longest_path, tolerance=2
)

# Visualize the longest path
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.axis("off")
ax.set_title("Original")
ax.plot(j_medial, i_medial, "ro")

# Plot the longest path on the original image
i_path = [point._row_index for point in points_on_path]
j_path = [point._col_index for point in points_on_path]
ax.plot(j_path, i_path, color="cyan")

plt.show()

# Step 7: Form the backbone from the coordinates of the longest path
