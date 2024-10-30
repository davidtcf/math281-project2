import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from scipy.fftpack import fft, ifft, fftfreq

# Set up the grids
Nx = 512
Lx = 30
dx = Lx / Nx
x = np.arange( dx/2, Lx, dx ) - Lx / 2
kx = 2 * np.pi * fftfreq(Nx, d = dx)
x_real = np.arange(-16, 16, 0.01)

y1 = np.exp(-x**2)
y1_hat = fft(y1)
y2 = np.tanh(5-np.abs(x))
y2_hat = fft(y2)
y3 = np.cos(np.pi*(x/2)**2)
y3_hat = fft(y3)

y1_phys = np.exp(-x_real**2)
y2_phys = np.tanh(5-np.abs(x_real))
y3_phys = np.cos(np.pi*(x_real/2)**2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot( kx[:Nx//2], np.abs(y1_hat[:Nx//2])**2, label = r'$\exp \left(-x^2\right)$', color = 'red')
ax1.plot( kx[:Nx//2], np.abs(y2_hat[:Nx//2])**2, label = r'$\tanh (5-|x|)$', color = 'blue')
ax1.plot( kx[:Nx//2], np.abs(y3_hat[:Nx//2])**2, label = r'$\cos \left(\pi(x / 2)^2\right)$', color = 'green')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Wavenumber')
ax1.set_ylabel(r'Spectral Power $\left|c_n\right|^2$')
ax1.set_title('Power Spectra plots')
ax1.grid(True)
ax1.legend()

ax2.plot( x_real, y1_phys, label = r'$\exp \left(-x^2\right)$', color = 'red')
ax2.plot( x_real, y2_phys, label = r'$\tanh (5-|x|)$', color = 'blue')
ax2.plot( x_real, y3_phys, label = r'$\cos \left(\pi(x / 2)^2\right)$', color = 'green')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Physical-space plots')
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig('Q2.png', dpi=300, format="png")
plt.show()
