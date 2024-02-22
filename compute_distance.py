from skimage.io import imread
from skimage.morphology import medial_axis
from pathlib import Path
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

    start = blob.copy()
    start.fill(1)
    start[i, j] = 0

    fig = plt.figure()
    axs = fig.subplots(1, 2)

    # Display the image
    axs[0].imshow(original, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original')

    # Use the medial axis function to find the distance transform
    medial_axis, distance = medial_axis(start, blob, return_distance=True)
    
    # Display the distance transform
    axs[1].imshow(distance, cmap='inferno')
    axs[1].axis('off')
    axs[1].set_title('Distance transform')

    plt.show()
    sys.exit(0)