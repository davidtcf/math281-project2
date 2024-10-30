import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Load in the data
Burgers_data = np.load('Burgers.npz')
plt.rcParams['font.size'] = 20
time = Burgers_data['time']         # The temporal grid
x    = Burgers_data['x']            # The spatial grid
u    = Burgers_data['u']            # The solution (first axis is time, second is space)
kx   = Burgers_data['kx']           # The wavenumbers relevant to the grid
nu   = float(Burgers_data['nu'])    # The viscosity coefficient

Nt, Nx = u.shape

print('The data has {0:d} time points and {1:d} space points.'.format(Nt, Nx))
print('The viscous coefficient nu is {0:g}'.format(nu))


# Create a single figure for multiple time indices
plt.figure(figsize=(20,14),dpi=300)
maxtime = 400
plotindices = np.arange(0, 15, 1)
criticaltime=10

r=0
for t in plotindices:  # Choose specific time indices to plot

    if r<criticaltime:

        coeff = np.zeros((Nt, Nx), dtype=complex)
        for i in range(Nt):
            coeff[i, :] = fft(u[i, :])

        specpower = np.abs(coeff) ** 2
        
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=0, vmax=20)

        plt.plot(kx[:Nx//2], specpower[t, :Nx//2],color='black', label=f'Time Steps 0-9' if t==0 else '')  # Only plot positive frequencies
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log(Wavenumber)')
        plt.ylabel('log(Spectral Power)')
        plt.title('Power Spectrum at Various Times')
        plt.grid(True)
        plt.legend()


    if r>=criticaltime:
        coeff = np.zeros((Nt, Nx), dtype=complex)
        for i in range(Nt):
            coeff[i, :] = fft(u[i, :])

        specpower = np.abs(coeff) ** 2

        
        cmap = plt.get_cmap('Dark2')
        norm = plt.Normalize(vmin=criticaltime, vmax=15)

        plt.plot(kx[:Nx//2], specpower[t, :Nx//2],color=cmap(norm(t)),label=f'Time Step {t}')  # Only plot positive frequencies
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavenumber')
        plt.ylabel('Spectral Power')
        plt.title('Power Spectrum Between the 0th and 14th Time Step')
        plt.grid(True)
        name="anim"+f"{t:04}"
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
        print()
        
    r=r+1

plt.savefig("Q4b2.png",dpi=300,format="png")






# Create a single figure for multiple time indices
plt.figure(figsize=(20,14),dpi=300)
maxtime = 400
plotindices = np.arange(0, maxtime, 1)

r=0
for t in plotindices:  # Choose specific time indices to plot
        if r<12:
                
            coeff = np.zeros((Nt, Nx), dtype=complex)
            for i in range(Nt):
                coeff[i, :] = fft(u[i, :])

            specpower = np.abs(coeff) ** 2
            
            plt.plot(kx[:Nx//2], specpower[t, :Nx//2],alpha=0, label=f'Time 0-9' if t==0 else '')  # Only plot positive frequencies
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Wavenumber')
            plt.ylabel('Spectral Power')
            plt.title('Power Spectrum While Large Wavenumbers are Gaining Energy')
            plt.grid(True)
           
        if r>=12:
                    
                coeff = np.zeros((Nt, Nx), dtype=complex)
                for i in range(Nt):
                    coeff[i, :] = fft(u[i, :])

                specpower = np.abs(coeff) ** 2
                
                cmap = plt.get_cmap('rainbow')
                norm = plt.Normalize(vmin=12, vmax=maxtime)

                plt.plot(kx[:Nx//2], specpower[t, :Nx//2],color=cmap(norm(t)), label=f'Time 0-9' if t==0 else '')  # Only plot positive frequencies
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Wavenumber')
                plt.ylabel('Spectral Power')
                plt.title('Power Spectrum While Large Wavenumbers are Losing Energy')
                plt.grid(True)
                
        r=r+1

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Time Step')

plt.savefig("Q4b3.png",dpi=300,format="png")





plt.figure(figsize=(20,14),dpi=300)
maxtime = 12
plotindices = np.arange(0, maxtime, 1)

r=0
for t in plotindices:  # Choose specific time indices to plot

    coeff = np.zeros((Nt, Nx), dtype=complex)
    for i in range(Nt):
        coeff[i, :] = fft(u[i, :])

    specpower = np.abs(coeff) ** 2
    
    cmap = plt.get_cmap('magma')
    norm = plt.Normalize(vmin=0, vmax=maxtime)

    plt.plot(kx[:Nx//2], specpower[t, :Nx//2],color=cmap(norm(t)), label=f'Time 0-9' if t==0 else '')  # Only plot positive frequencies
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavenumber')
    plt.ylabel('Spectral Power')
    plt.title('Power Spectrum While Large Wavenumbers are Gaining Energy')
    plt.grid(True)



# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Time Step')

plt.savefig("Q4b1.png",dpi=300,format="png")
