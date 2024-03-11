import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

from backbone import backbone
from fmmdistance import distance_from_edge, distance_from_seed_set


def main(country_name):
    world_file_path = Path(
        "./data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
    )
    world = gpd.read_file(world_file_path)

    country = world.loc[world["NAME"] == country_name]

    # Make a binary image of the country
    # Pixels inside the country are 1, outside are 0
    nrows, ncols = 128, 128
    xv, yv = np.meshgrid(
        np.linspace(country.bounds.minx, country.bounds.maxx, ncols),
        np.linspace(country.bounds.miny, country.bounds.maxy, nrows),
    )

    # Convert the coordinates to a 1D array
    xv = xv.flatten()
    yv = yv.flatten()

    # Create a GeoDataFrame with the coordinates
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xv, yv))

    # Use the within method to check if the points are inside the country
    gdf["is_inside"] = gdf.within(country.unary_union)

    # Reshape the array to a 2D array
    is_inside = gdf["is_inside"].values.reshape(nrows, ncols)

    # Plot the binary image
    plt.imshow(is_inside, origin="lower", cmap="gray")

    # Find the backbone
    backbone_pixels = backbone(is_inside)
    i_backbone = [i for i, j in backbone_pixels]
    j_backbone = [j for i, j in backbone_pixels]

    # Plot the backbone
    plt.plot(j_backbone, i_backbone, "r-")

    # Compute the distance transform from the edge
    distance_from_border = distance_from_edge(is_inside)

    # Compute the distance transform from the backbone pixels
    seed_set = list(zip(i_backbone, j_backbone))
    distance_from_backbone = distance_from_seed_set(is_inside, seed_set)

    # Identify the proximal and distal points
    # These are the first and last point of the backbone
    proximal_point = seed_set[0]
    distal_point = seed_set[-1]

    # Compute the distance from the proximal and distal points
    distance_from_proximal = distance_from_seed_set(is_inside, [proximal_point])
    distance_from_distal = distance_from_seed_set(is_inside, [distal_point])

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))

    # Plot the distance from the edge
    axs[0, 0].imshow(distance_from_border, cmap="inferno", origin="lower")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Distance transform from edge")

    # Plot the distance from the backbone
    axs[0, 1].imshow(distance_from_backbone, cmap="inferno", origin="lower")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Distance transform from backbone")

    # Plot the distance from the proximal point
    axs[1, 0].imshow(distance_from_proximal, cmap="inferno", origin="lower")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Distance transform from proximal point")

    # Plot the distance from the distal point
    axs[1, 1].imshow(distance_from_distal, cmap="inferno", origin="lower")
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Distance transform from distal point")

    # Compute distality and peripherality values
    distality = distance_from_proximal / (distance_from_proximal + distance_from_distal)
    peripherality = distance_from_backbone / (
        distance_from_backbone + distance_from_border
    )

    # Plot the distality and peripherality values
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the distality
    axs[0].imshow(distality, cmap="inferno", origin="lower")
    axs[0].axis("off")
    axs[0].set_title("Distality")

    # Plot the peripherality
    axs[1].imshow(peripherality, cmap="inferno", origin="lower")
    axs[1].axis("off")
    axs[1].set_title("Peripherality")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_country.py <country_name>")
        sys.exit(1)

    country_name = sys.argv[1]
    main(country_name)
    sys.exit(0)
