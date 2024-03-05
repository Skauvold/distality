from skimage.io import imread
from pathlib import Path
import matplotlib.pyplot as plt

from fmmdistance import distance_from_edge, distance_from_seed_set


path_to_image = Path("data/05_bend.png")
original = imread(path_to_image)
img = original[:, :, 0]

distance = distance_from_edge(img)

img_dimensions = img.shape
seed_set = [(img_dimensions[0] // 3, img_dimensions[1] // 4)]
distance_from_seed = distance_from_seed_set(img, seed_set)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img, cmap="gray")
axs[0].axis("off")
axs[0].set_title("Original")

axs[1].imshow(distance, cmap="inferno")
axs[1].axis("off")
axs[1].set_title("Distance transform from edge")

axs[2].imshow(distance_from_seed, cmap="inferno")
axs[2].axis("off")
axs[2].set_title("Distance transform from seed set")

plt.show()
