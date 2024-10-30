import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection

from scipy.fftpack import fft, ifft, fftfreq

plt.rcParams['figure.dpi'] = 300

# Set up the spatial grid
Nx = 128
Lx = 2 * np.pi
dx = Lx / Nx

x = np.arange( dx/2, Lx, dx )

# Get the wavenumbers
kx = 2 * np.pi * fftfreq(Nx, d = dx)

y = np.sin(x)

# Compute the derivative
#   we inclue the .real at the end
#   to remove imaginary components that come
#   from rounding error / machine precision etc
# Note that 1j is the python notation for the imaginary unit
dydx_spectral = ifft( (1j * kx) * fft(y) ).real
d2ydx2_spectral = ifft( (1j * kx)**2 * fft(y) ).real
d3ydx3_spectral = ifft( (1j * kx)**3 * fft(y) ).real
d4ydx4_spectral = ifft( (1j * kx)**4 * fft(y) ).real

dydx_true = np.cos(x)
d2ydx2_true = -np.sin(x)
d3ydx3_true = -np.cos(x)
d4ydx4_true = np.sin(x)

plt.rcParams['font.size'] = 20
fig, axes = plt.subplots(2, 1, sharex=True, figsize = (14, 20))
axes[0].plot( x, dydx_true,           linewidth = 4, label = 'True Derivative' )
axes[0].plot( x, dydx_spectral, '--', linewidth = 3, label = 'Spectral Derivative' )
axes[1].plot( x, dydx_true - dydx_spectral )
axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('First Derivative')
axes[1].set_ylabel('Error')
axes[0].legend()
axes[1].legend()
axes[0].set_title('First Derivative')
plt.savefig("Q1a1.png")

fig, axes = plt.subplots(2, 1, sharex=True, figsize = (14, 20))
axes[0].plot( x, d2ydx2_true,           linewidth = 4, label = 'True Derivative' )
axes[0].plot( x, d2ydx2_spectral, '--', linewidth = 3, label = 'Spectral Derivative' )
axes[1].plot( x, d2ydx2_true - d2ydx2_spectral )
axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('Second Derivative')
axes[1].set_ylabel('Error')
axes[0].legend()
axes[1].legend()
axes[0].set_title('Second Derivative')
plt.savefig("Q1a2.png")

fig, axes = plt.subplots(2, 1, sharex=True, figsize = (14, 20))
axes[0].plot( x, d3ydx3_true,           linewidth = 4, label = 'True Derivative' )
axes[0].plot( x, d3ydx3_spectral, '--', linewidth = 3, label = 'Spectral Derivative' )
axes[1].plot( x, d3ydx3_true - d3ydx3_spectral )
axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('Third Derivative')
axes[1].set_ylabel('Error')
axes[0].legend()
axes[1].legend()
axes[0].set_title('Third Derivative')
plt.savefig("Q1a3.png")

fig, axes = plt.subplots(2, 1, sharex=True, figsize = (14, 20))
axes[0].plot( x, d4ydx4_true,           linewidth = 4, label = 'True Derivative' )
axes[0].plot( x, d4ydx4_spectral, '--', linewidth = 3, label = 'Spectral Derivative' )
axes[1].plot( x, d4ydx4_true - d4ydx4_spectral )
axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('Fourth Derivative')
axes[1].set_ylabel('Error')
axes[0].legend()
axes[1].legend()
axes[0].set_title('Fourth Derivative')
plt.savefig("Q1a4.png")


gridspec_props = dict(wspace = 0.05, hspace = 0.5, left = 0.1, right = 0.8, bottom = 0.1, top = 0.9)
fig, axes = plt.subplots(2, 1, figsize=(14,20), gridspec_kw = gridspec_props)
i=0
for Nmodes in [1,10]:
    def f(x):
        return np.sin(Nmodes * x)

    def f_prime(x):
        return np.cos(Nmodes * x) * Nmodes

    Lx = 2*np.pi
    Nxs = np.power(2, np.arange(2, 9))

    # Create some arrays to store the error values
    err_2 = np.zeros(Nxs.shape)
    err_4 = np.zeros(Nxs.shape)
    err_spec = np.zeros(Nxs.shape)

   

    for Nx, ind in zip(Nxs, range(len(Nxs))):
        
        # Grid with chosen resolution
        x = np.linspace(0, Lx, Nx, endpoint=False)
        Delta_x = x[1] - x[0]
        kx2 = 2 * np.pi * fftfreq(Nx, d = Lx/Nx)
        # Function to differentiation
        y  = f(x)
        
        # True derivative
        yp = f_prime(x)
        
        # Compute the numerical derivatives
        Ord2 = (                      np.roll(y, -1) -   np.roll(y, 1)                 ) / ( 2*Delta_x)
        Ord4 = ( - np.roll(y, -2) + 8*np.roll(y, -1) - 8*np.roll(y, 1) + np.roll(y, 2) ) / (12*Delta_x)
        dydx_spectral2 = ifft( (1j * kx2) * fft(y) ).real
        
        # Store the error in the derivatives
        err_2[ind] = np.sqrt(np.mean( (Ord2 - yp)**2 ))
        err_4[ind] = np.sqrt(np.mean( (Ord4 - yp)**2 ))
        err_spec[ind] = np.sqrt(np.mean( (dydx_spectral2 - yp)**2 ))
        
    ##
    ## Add some formatting to the figures, and plot the error values.
    ##
    
    #print(i)
    axes[i].plot(Lx/Nxs, err_2, '-o', label='2nd order')
    axes[i].plot(Lx/Nxs, err_4, '-o', label='4th order')
    axes[i].plot(Lx/Nxs, err_spec, '-o', label='Spectral ')
    axes[i].plot(Lx/Nxs, (Lx/Nxs)**2, '--k', label='$dx^2$')
    axes[i].plot(Lx/Nxs, (Lx/Nxs)**4, '-.k', label='$dx^4$')
    axes[i].set_yscale('log')
    axes[i].set_xscale('log')
    i+=1
    


axes[0].legend(bbox_to_anchor=(1., 0, 0.25, 1))
axes[0].set_xlabel('$\Delta x$')
axes[0].set_ylabel('Finite Difference Error Nmodes=1')
axes[0].set_title('Demonstration of Convergence Order')

axes[1].set_xlabel('$\Delta x$')
axes[1].set_ylabel('Finite Difference Error Nmodes=10')
axes[1].set_title('Demonstration of Convergence Order')

    
axes[0].legend(bbox_to_anchor=(1., -1, 0.25, 2))
plt.tight_layout()
plt.savefig("Q1b.png")

