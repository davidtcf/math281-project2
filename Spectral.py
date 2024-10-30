import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, ifft

# Parameters
Nx = 128
x = np.linspace(0, 1, Nx+1)[:-1]
kx = fftfreq(Nx, d=1./Nx)
slopes = [-4, -3, -2, -1, 0, 1]

# Set up figure for multiple plots
fig, axes = plt.subplots(len(slopes), 2, figsize=(10, 18), gridspec_kw=dict(wspace=0.3, hspace=0.5))

# Generate plots for each slope
for i, slope in enumerate(slopes):
    coefs = np.exp(1j * 2 * np.pi * np.random.rand(Nx)) * np.power(np.abs(kx), slope)
    coefs[kx == 0] = 1.
    coefs[Nx//2+1:] = np.conj(coefs[1:Nx//2][::-1])
    y_tmp = ifft(coefs).real

    # Physical space plot
    axes[i, 0].plot(x, y_tmp)
    axes[i, 0].set_title(f'Physical Space (Slope = {slope})')
    axes[i, 0].set_xlabel('x')
    axes[i, 0].set_ylabel('y')
    axes[i, 0].grid(True)

    # Spectral space plot
    axes[i, 1].plot(kx[:Nx//2], np.abs(coefs)[:Nx//2])
    axes[i, 1].set_yscale('log')
    axes[i, 1].set_xscale('log')
    axes[i, 1].set_title(f'Spectral Space (Slope = {slope})')
    axes[i, 1].set_xlabel('wavenumber')
    axes[i, 1].set_ylabel('abs(y hat)')
    axes[i, 1].grid(True)

plt.tight_layout()
plt.savefig('Spectral.png', dpi=1200, format="png")
plt.show()
