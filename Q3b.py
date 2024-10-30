import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# First, we need to load in the data
heat_data = np.load('heat.npz')

time = heat_data['time']         # The temporal grid
x    = heat_data['x']            # The spatial grid
u    = heat_data['u']            # The solution (first axis is time, second is space)
kx   = heat_data['kx']           # The wavenumbers relevant to the grid
nu   = float(heat_data['nu'])    # The viscosity coefficient

Nt, Nx = u.shape


coeff = np.zeros((Nt, Nx), dtype=complex)
for i in range(Nt):
    coeff[i, :] = fft(u[i, :])

cmap = plt.get_cmap('plasma')

norm = plt.Normalize(vmin=np.min(kx[:16]), vmax=np.max(kx[:16]))

plt.figure(figsize=(8, 6))

for i in range(0, 16, 1):
    specific_wavenumber = kx[i]
    coefficients_vs_time = coeff[:, i]
    plt.plot(time, np.abs(coefficients_vs_time), color=cmap(norm(specific_wavenumber)))


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Wavenumber Value')

plt.xlabel('Time (s)')
plt.ylabel('Magnitude of Coefficient')
plt.title('Coefficients vs Time the First 16 Wavenumbers')
plt.grid(True)
plt.tight_layout()
plt.savefig('Q3b.png')
plt.show()
