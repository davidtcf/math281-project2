import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from matplotlib.collections import LineCollection
import matplotlib

# Load the Burgers equation data
Burgers_data = np.load('Burgers.npz')
time = Burgers_data['time']         # Temporal grid
x = Burgers_data['x']               # Spatial grid
u = Burgers_data['u']               # Solution (u at each time and space point)
kx = Burgers_data['kx']             # Wavenumbers for the grid
nu = float(Burgers_data['nu'])      # Viscosity coefficient

# Determine the number of time and space points
Nt, Nx = u.shape

# Select time indices for plotting (to get evolution at specific times)
plot_inds = np.linspace(0, Nt - 1, 10, dtype=int)  # Select 10 evenly spaced time indices

# Initialize a single figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the physical-space solution `u(x, t)` over time with `u` on y-axis
# Generate lines for each time slice
u_lines = [np.column_stack([x, u[ii, :]]) for ii in plot_inds]
u_segments = LineCollection(u_lines, cmap=matplotlib.cm.viridis)
u_segments.set_array(time[plot_inds])

# Physical-space plot setup (left subplot)
ax1.add_collection(u_segments)
ax1.axis('tight')
cbar1 = plt.colorbar(u_segments, ax=ax1)
cbar1.set_label('time', rotation=-90, labelpad=20)
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.set_title('Physical-Space Solution $u(x, t)$')

# Plot the power spectrum for each selected time on the right subplot
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=time[plot_inds].min(), vmax=time[plot_inds].max())

for ind in plot_inds:
    # Compute the Fourier transform and power spectrum
    u_hat = fft(u[ind, :])
    power_spectrum = np.abs(u_hat[:Nx//2])**2  # Only positive wavenumbers
    color = cmap(norm(time[ind]))  # Map time to color
    ax2.plot(kx[:Nx//2], power_spectrum, color=color, label=f't = {time[ind]:.2f}')

# Spectral-space plot setup (right subplot)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Wavenumber (k)')
ax2.set_ylabel('Power Spectrum $|c_n|^2$')
ax2.set_title('Power Spectrum Over Time')
ax2.grid(True)

# Add a color bar for time to match the colors used in the power spectrum plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for colorbar
cbar2 = plt.colorbar(sm, ax=ax2, label='time')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Q4a.png', dpi=300, format="png")
plt.show()
