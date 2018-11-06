import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.util
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

vector_crs = ccrs.PlateCarree()
land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
    edgecolor='face',facecolor='dimgray', linewidth=0)

fig = plt.figure(figsize=(16, 9))
matplotlib.rcParams.update({'font.size': 10})

crs_sps = ccrs.PlateCarree(central_longitude=-150)
crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

ax = plt.subplot(111, projection=crs_sps)
ax.add_feature(land_50m)
ax.set_extent([-180, -120, 0, 60], ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black',
    alpha=0.8, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlocator = mticker.FixedLocator([-180, -170, -160, -150, -140, -130, -120])
gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

im = ax.pcolormesh(lons, lats, u_magnitude, transform=vector_crs, vmin=0, vmax=1, cmap='Blues_r')

clb = fig.colorbar(im, ax=ax, extend='max', fraction=0.046, pad=0.1)
clb.ax.set_title(r'm/s')

rock_lats, rock_lons = [], []
paper_lats, paper_lons = [], []
scissors_lats, scissors_lons = [], []

for microbe in pset:
    if microbe.species == 1:
        rock_lats.append(microbe.lat)
        rock_lons.append(microbe.lon)
    elif microbe.species == 2:
        paper_lats.append(microbe.lat)
        paper_lons.append(microbe.lon)
    elif microbe.species == 3:
        scissors_lats.append(microbe.lat)
        scissors_lons.append(microbe.lon)

ax.plot(rock_lons, rock_lats, marker='o', linestyle='', color='red', ms=4, label='Rocks', transform=vector_crs)
ax.plot(paper_lons, paper_lats, marker='o', linestyle='', color='lime', ms=4, label='Papers', transform=vector_crs)
ax.plot(scissors_lons, scissors_lats, marker='o', linestyle='', color='cyan', ms=4, label='Scissors', transform=vector_crs)

plt.title(str(t))
ax.legend()

# plt.show()

png_filename = "advected_microbes_" + str(n).zfill(4) + ".png"
print("Saving figure: {:s}".format(png_filename))
plt.savefig(png_filename, dpi=300, format='png', transparent=False)

plt.close('all')