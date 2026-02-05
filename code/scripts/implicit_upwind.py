import numpy as np
from geometric_fv.grid import Grid1D
import matplotlib.pyplot as plt

mesh = Grid1D.uniform(0.0, 1.0, 50)

x_c = mesh.centers

u0 = np.sin(2*np.pi*x_c)

ncells = mesh.ncells
nghost = 1
u_new = np.zeros(ncells + 2 * nghost)
u_old = np.pad(u0, (nghost,nghost), 'constant', constant_values=0.0)

cfl = 1.0
u_new[0] = 0.0
for t in range(3):
    for i in range(nghost, ncells + nghost):
        u_new[i] = (u_old[i] + cfl * u_new[i-1]) / (1. + cfl)
    u_old[:] = u_new[:]

