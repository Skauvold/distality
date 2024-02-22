from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize
from pathlib import Path
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # Get path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python compute_parameters.py <path_to_image>")
        sys.exit(1)
    
    # Read the image
    path_string = sys.argv[1]
    path_to_image = Path(path_string)
    original = imread(path_to_image)

    # Convert to grayscale
    img = original[:, :, 0]

    fig = plt.figure()
    axs = fig.subplots(2, 2)

    # Display the image
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Original')

    # Find the image's medial axis
    medial_axis, distance = medial_axis(img, return_distance=True)

    # Display the medial axis
    axs[0, 1].imshow(medial_axis, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Medial axis')
    
    # Display the distance transform
    axs[1, 0].imshow(distance, cmap='inferno')
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Distance transform')

    # Find the image's skeleton
    skel = skeletonize(img)

    # Display the skeleton
    axs[1, 1].imshow(skel, cmap='gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Skeleton')

    plt.show()
    sys.exit(0)