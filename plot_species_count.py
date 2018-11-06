import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

iters = range(1460)
time = []
n_rocks = []
n_papers = []
n_scissors = []

for i in iters:
	nc_filename = "advected_microbes_" + str(i).zfill(4) + ".nc"
	print("Processing {:s}...".format(nc_filename))

	data = xr.open_dataset(nc_filename)

	species = data['species'][:,0].values
	n_rocks.append(np.sum(species == 1))
	n_papers.append(np.sum(species == 2))
	n_scissors.append(np.sum(species == 3))

fig = plt.figure(figsize=(9, 9))
ax = plt.subplot(111)

ax.plot(iters, n_rocks, color='red', label='Rocks')
ax.plot(iters, n_papers, color='green', label='Papers')
ax.plot(iters, n_scissors, color='blue', label='Scissors')
plt.xlabel("Time")
plt.ylabel("Microbe species count")
plt.title("Rock, paper, scissors")
ax.legend()

# plt.show()
plt.savefig("species_count.png", dpi=300, format='png', transparent=False)