import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

world_file_path = Path("./data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
world = gpd.read_file(world_file_path)

country = world.loc[world["NAME"] == "Senegal"]

country.plot()
plt.show()
