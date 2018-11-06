import os
import numpy as np
import matplotlib.pyplot as plt


u_filename = '/home/alir/hawaii_npac/0000969408_U_10800.8150.1_1080.3720.90'
v_filename = '/home/alir/hawaii_npac/0000969408_V_10800.8150.1_1080.3720.90'

level = 0

with open(u_filename, 'rb') as f:
    nx, ny = 1080, 3720  # parse; advance file-pointer to data segment
    record_length = 4  # [bytes]

    f.seek(level * record_length * nx*ny, os.SEEK_SET)

    u_data = np.fromfile(f, dtype='>f4', count=nx*ny)
    u_array = np.reshape(u_data, [nx, ny], order='F')

with open(v_filename, 'rb') as f:
    nx, ny = 1080, 3720  # parse; advance file-pointer to data segment
    record_length = 4  # [bytes]

    f.seek(level * record_length * nx*ny, os.SEEK_SET)

    v_data = np.fromfile(f, dtype='>f4', count=nx*ny)
    v_array = np.reshape(v_data, [nx, ny], order='F')

u_marr = np.ma.masked_where(u_array == 0, u_array)
v_marr = np.ma.masked_where(v_array == 0, v_array)

u_magnitude = np.sqrt(u_marr*u_marr + v_marr*v_marr)

fig = plt.figure(figsize=(9, 16))

plt.imshow(np.transpose(u_magnitude), cmap='Blues_r', vmin=0, vmax=1)
# plt.imshow(np.transpose(marr), cmap='RdBu', vmin=-2.5, vmax=2.5)

plt.colorbar()
plt.gca().invert_yaxis()
plt.savefig('u_magnitude.png', dpi=600, format='png', transparent=False)
