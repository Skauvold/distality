from skimage.io import imread
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.morphology import medial_axis

from backbone import (
    extract_foreground_ij,
    create_graph_from_connected_points,
)

path_to_image = Path("data/02_blob.png")
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

plt.show()

# Step 3: Make a graph of medial axis endpoints and junctions
# and the distances between them
graph = create_graph_from_connected_points(i_medial, j_medial)


# Step 4: Compute pairwise shortest distances between all pairs of endpoints

# Step 5: Identify the longest path in the graph

# Step 6: Extend the longest path to the boundary of the image

# Step 7: Form the backbone from the coordinates of the longest path
