import numpy as np
import matplotlib.pyplot as plt

filename = '/home/alir/hawaii_npac/0000969408_Theta_10800.8150.1_1080.3720.90'

with open(filename, 'rb') as f:
    nx, ny = 1080, 3720  # parse; advance file-pointer to data segment
    data = np.fromfile(f, dtype='>f4', count=nx*ny)
    array = np.reshape(data, [nx, ny], order='F')
    data2 = np.fromfile(f, dtype='>f4', count=nx*ny)
    array2 = np.reshape(data, [nx, ny], order='F')

print('array.min() = {:f}'.format(array.min()))
print('array.max() = {:f}'.format(array.max()))

# xex=-np.linspace(0.5,nx-0.5,nx)
# yex=np.linspace(0.5,ny-0.5,ny)

marr = np.ma.masked_where(array == 0, array)

fig = plt.figure(figsize=(9, 16))

plt.imshow(np.transpose(marr), cmap='jet')
# plt.imshow(np.transpose(marr), cmap='RdBu', vmin=-2.5, vmax=2.5)

plt.colorbar()
plt.gca().invert_yaxis()
plt.savefig('py.png', dpi=600, format='png', transparent=False)
