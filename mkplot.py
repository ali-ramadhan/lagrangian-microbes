import numpy as np
import matplotlib.pyplot as plt

filename = '/home/alir/hawaii_npac/0000649728_Theta_10800.8150.1_1080.3720.40'

with open(filename, 'rb') as f:
    nx, ny = 1080, 3720  # parse; advance file-pointer to data segment
    data = np.fromfile(f, dtype='>f4', count=nx*ny)
    array = np.reshape(data, [nx, ny], order='F')

print(array.min())
print(array.max())

xex=-np.linspace(0.5,nx-0.5,nx)
yex=np.linspace(0.5,ny-0.5,ny)

marr = np.ma.masked_where(array == 0, array)

plt.imshow(np.transpose(marr))
plt.colorbar()
# plt.show()
# plt.imshow(array, aspect='auto', interpolation='none',
#            extent=extents(nx) + extents(ny), origin='lower')
plt.savefig('py.png')
