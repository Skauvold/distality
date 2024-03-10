import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from backbone import backbone

world_file_path = Path("./data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
world = gpd.read_file(world_file_path)

country = world.loc[world["NAME"] == "Vietnam"]

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

plt.show()
