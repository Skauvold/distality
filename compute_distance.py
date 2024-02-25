import skfmm
from skimage.io import imread
from skimage.morphology import medial_axis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # Get path and coordinates from command line argument
    if len(sys.argv) < 4:
        print("Usage: python compute_parameters.py <path_to_image> <i> <j>")
        sys.exit(1)
    
    # Read the image
    path_string = sys.argv[1]
    path_to_image = Path(path_string)
    original = imread(path_to_image)

    # Convert to binary
    blob = original[:, :, 0] > 0

    i = int(sys.argv[2])
    j = int(sys.argv[3])

    start = np.full_like(blob, 1, dtype=float)
    start[i, j] = -1

    fig = plt.figure()
    axs = fig.subplots(1, 2)

    # Display the image
    axs[0].imshow(original, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original')

    # Mask the start array with the blob array
    mask = np.logical_not(blob)
    phi = np.ma.masked_array(start, mask)

    # Compute distance from (i, j) within the blob using the fast marching method
    distance = skfmm.distance(phi, dx=1)
    
    # Display the distance transform
    axs[1].imshow(distance, cmap='inferno')
    axs[1].axis('off')
    axs[1].set_title('Distance transform')

    plt.show()
    sys.exit(0)