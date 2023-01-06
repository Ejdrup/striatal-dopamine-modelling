#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:28:36 2022

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sim_space_neurons_3D(width = 100, depth = 10, dx_dy = 1, time = 1, D = 763, 
              inter_var_distance = 2.92, p_r = 0.06, f_rate = 4, n_neurons = 150):
    ## Generate overall simulation space
    
    # Simulation time, sec.
    t = time
    # Field size, um
    w = h = width
    # Depth of field
    depth = depth
    # Intervals in x-, y- directions, um
    dx = dy = dz = dx_dy
    # Steps per side
    nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz)
    # Calculate time step
    dx2, dy2 = dx*dx, dy*dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
    # Generate simulation space
    space0 = np.zeros((int(t/dt), nx, ny, nz))
    # Prep for image of all release sites
    space_init = np.zeros((nx, ny, nz))
    
    ## Generate firing pattern
    p_r = p_r
    # Number of varicosities
    n_varico = int((width**2*depth)/inter_var_distance)
    # Generate varicosities from random linear distribution
    x_varico = np.random.randint(0, high = (w/dx), size = n_varico)
    y_varico = np.random.randint(0, high = (w/dy), size = n_varico)
    z_varico = np.random.randint(0, high = (depth/dz), size = n_varico)
    
    # Assign neuron identity to terminals
    neuro_identity = np.random.randint(0, high = n_neurons, size = n_varico)
    # Firing pattern of each neuron
    neuron_firing = np.random.poisson(f_rate*dt,(n_neurons,int(t/dt)))
    neuron_firing[neuron_firing > 1] = 1 # avoid multiple release events on top of each other
    
    # Firing pattern of each terminal
    firing = neuron_firing[neuro_identity,:]
    # Add indv release prob of 0.06
    terminal_events = np.where(firing)
    terminal_refraction = np.random.choice(len(terminal_events[0]),
                     int(len(terminal_events[0])*(1-p_r)),
                     replace = False)

    firing[terminal_events[0][terminal_refraction],terminal_events[1][terminal_refraction]] = 0
    # terminal_refraction = np.random.poisson(1-p_r,firing.shape)
    # terminal_refraction[terminal_refraction > 1] = 1 # avoid multiple release events on top of each other
    # terminal_refraction = (terminal_refraction-1)*-1 # flip logic
    
    
    # Populate image of all release sites
    space_init[x_varico,y_varico,z_varico] = 1

    return space0, space_init, firing.T, np.array([x_varico,y_varico, z_varico]), np.array([time, dt, dx_dy, inter_var_distance])

def sim_space_3D(width = 100, depth = 10, dx_dy = 1, time = 1, D = 763, 
              inter_var_distance = 2.92, p_r = 0.06, f_rate = 4):
    ## Generate overall simulation space
    
    # Simulation time, sec.
    t = time
    # Field size, um
    w = h = width
    # Depth of field
    depth = depth
    # Intervals in x-, y- directions, um
    dx = dy = dz = dx_dy
    # Steps per side
    nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz)
    # Calculate time step
    dx2, dy2 = dx*dx, dy*dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
    # Generate simulation space
    space0 = np.zeros((int(t/dt), nx, ny, nz))
    # Prep for image of all release sites
    space_init = np.zeros((nx, ny, nz))
    
    ## Generate firing pattern
    p_r = p_r
    # Number of varicosities
    n_varico = int((width**2*depth)/inter_var_distance)
    # Generate varicosities from random linear distribution
    x_varico = np.random.randint(0, high = (w/dx), size = n_varico)
    y_varico = np.random.randint(0, high = (w/dy), size = n_varico)
    z_varico = np.random.randint(0, high = (depth/dz), size = n_varico)
    # Firing pattern
    firing = np.random.poisson(f_rate*dt*p_r,(n_varico,int(t/dt)))
    firing[firing > 1] = 1 # avoid multiple release events on top of each other
    
    # Populate image of all release sites
    space_init[x_varico,y_varico,z_varico] = 1

    return space0, space_init, firing.T, np.array([x_varico,y_varico, z_varico]), np.array([time, dt, dx_dy, inter_var_distance])

def sim_dynamics_3D(space0, release_sites, firing, var_list, 
                 Q = 3000, uptake_rate = 4*10**-6, Km = 210*10**-9,
                 Ds = 321.7237308146399, ECF = 0.21):
    # print(uptake_rate)
    # print(Q)
    # Extract parameters
    t = var_list[0]
    dt = var_list[1]
    dx_dy = var_list[2]
    
    # DA release per vesicle
    single_vesicle_vol = (4/3*np.pi*(0.025)**3) 
    voxel_volume = (dx_dy)**3 # Volume of single voxel
    single_vesicle_DA = 0.025 * Q/1000 # 0.025 M at Q = 1000
    Q_eff = single_vesicle_vol/voxel_volume * single_vesicle_DA * 1/ECF
    
    
    
    for i in range(int(t/dt)-1):
        # Add release events per time step
        space0[i, release_sites[0,:][np.where(firing[i,:])],
               release_sites[1,:][np.where(firing[i,:])],
               release_sites[2,:][np.where(firing[i,:])]] += Q_eff
    
        
        _, space0[i+1,:,:,:] = do_timestep_3D(space0[i,:,:,:], 
                                      uptake_rate, Ds, dt, dx_dy, Km)
    
    return space0

def do_timestep_3D(u0, uptake_rate, Ds, dt, dx_dy, Km):
    u = u0.copy()
    
    # Propagate with forward-difference in time, central-difference in space
    u = u0 + Ds * dt * (
        (np.roll(u0, 1, axis = 0) - \
         2*u0 + \
         np.roll(u0, -1, axis = 0))  /dx_dy**2 
              + \
        (np.roll(u0, 1, axis = 1) - \
         2*u0 + \
         np.roll(u0, -1, axis = 1))  /dx_dy**2
              + \
        (np.roll(u0, 1, axis = 2) - \
          2*u0 + \
          np.roll(u0, -1, axis = 2))  /dx_dy**2)
    
    
    # Simulate reuptake
    u = u - dt*(uptake_rate*u)/(Km + u)
    
    u0 = u.copy()
    return u0, u


#%% Simulate with 150 neurons or uncorrelated terminals

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4)
        
# DS
full_sim_DS = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150)
        
# DS with 150 neurons
full_sim_DS_synch = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 4)
        
# DS with neurons in 4 groups
full_sim_DS_groups = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

# %% Cross section
time_points = []
for i in range(3):
    time_points.append(np.argmin(abs(np.linspace(0,2,len(full_sim_DS_synch))-i)))

fig, (ax2, ax1, ax3, ax4) = plt.subplots(4,1,figsize = (3.5,3), dpi = 400)
# fig.text(-0.02,0.725, "Seperate terminals", va = "center", rotation = 90, color = "darkblue")
# fig.text(-0.02,0.33, "150 neurons", va = "center", rotation = 90, color = "cornflowerblue")
fig.suptitle("Cross-section over time", fontsize = 10, x = 0.56, y = 0.95)

fig.text(0.042,0.625, "Cross-section", fontsize = 10, rotation = 90, va = "center")

ax1.set_title("Uncoupled terminals", fontsize = 10, color = "darkblue")
ax1.set_ylabel("50 um", labelpad = 7)
ax1.set_yticks([])
ax1.set_xticks(time_points)
ax1.set_xticklabels([])
ax1.set_xlim(time_points[0],time_points[-1])
im = ax1.imshow(np.log10(full_sim_DS[:,25,:,25].T+10**-10), aspect = 12, vmin = -8.5, vmax = -6.5)
ax1.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)

ax2.set_title("150 neurons", fontsize = 10, color = "cornflowerblue")
ax2.set_ylabel("50 um", labelpad = 7)
ax2.set_yticks([])
ax2.set_xticks(time_points)
ax2.set_xticklabels([])
ax2.set_xlim(time_points[0],time_points[-1])
ax2.imshow(np.log10(full_sim_DS_synch[:,25,:,25].T+10**-10), aspect = 12, vmin = -8.5, vmax = -6.5)
ax2.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)

ax3.set_title("Max synchronicity", fontsize = 10, color = "blue")
ax3.set_ylabel("50 um", labelpad = 7)
ax3.set_yticks([])
ax3.set_xticks(time_points)
ax3.set_xticklabels([])
ax3.set_xlim(time_points[0],time_points[-1])
im = ax3.imshow(np.log10(full_sim_DS_groups[:,25,:,25].T+10**-10), aspect = 12, vmin = -9, vmax = -6)
ax3.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)


ax4.plot(np.log10(full_sim_DS[:,25,24,25]+10**-10), color = "darkblue", lw = 1, zorder = 3)
ax4.plot(np.log10(full_sim_DS_synch[:,25,24,25]+10**-10), color = "cornflowerblue", lw = 1, zorder = 2)
ax4.plot(np.log10(full_sim_DS_groups[:,25,24,25]+10**-10), color = "blue", lw = 1, zorder = 4)


ax4.set_ylim(-9,-6)
ax4.set_ylabel('[DA] (nM)')
ax4.set_yticks([-6, -7, -8, -9])
ax4.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
ax4.set_xlim(time_points[0],time_points[-1])
ax4.set_xlabel("Seconds")
ax4.set_xticks(time_points)
ax4.set_xticklabels([0,1,2])

fig.tight_layout()

kw = {
    'vmin': -8.5,
    'vmax': -6.5,
    'levels': np.linspace(-9, -6, 100),
}
im = plt.contourf(
    np.log10(full_sim_DS[200,:,:,20]+10**-10),np.log10(full_sim_DS[200,:,:,20]+10**-10),np.log10(full_sim_DS[200,:,:,20]+10**-10),
     **kw, alpha = 1
)
    
cbar_ax = fig.add_axes([1, 0.22, 0.017, 0.555])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [-6, -7, -8, -9])
cbar_ax.set_ylabel('[DA] (nM)', labelpad=5, rotation = 90)
cbar_ax.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
cbar_ax.set_ylim( -6, -9 )


#%% Extract histogram

hist_DS, _ = np.histogram(np.log10(full_sim_DS[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)
hist_DS_synch, _ = np.histogram(np.log10(full_sim_DS_synch[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)
hist_DS_groups, _ = np.histogram(np.log10(full_sim_DS_groups[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)

#%% Plot Histograms

fig, (ax1) = plt.subplots(1,1,figsize = (2.3,3), dpi = 400)
x_range = np.logspace(-9,-6,100)
ax1.set_title("[DA] distribution", fontsize = 10)
ax1.plot(x_range, hist_DS_synch, color = "cornflowerblue", lw = 1.5, zorder = 2)
ax1.plot(x_range, hist_DS, color = "darkblue", lw = 1.5, zorder = 1)
ax1.plot(x_range, hist_DS_groups, color = "blue", lw = 1.5, zorder = 0)
ax1.set_ylabel("Denisty")
ax1.set_xlabel("nM")
ax1.set_xscale("log")
ax1.set_xlim(10**-9, 10**-6)
ax1.set_xticks([10**-9, 10**-8,10**-7, 10**-6])
ax1.set_xticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'][::-1])
ax1.set_ylim(-0.05,2.5)
legend = ax1.legend(("Neurons", "Terminals", "Max"), frameon = False,
            loc = "upper right", handlelength = 1.3, bbox_to_anchor = [1.09,1.01], fontsize = 8)
legend.set_title('Coupling  ',prop={'size':8})

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Do the same for VS (named saved for overwriting to save RAM)

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*0.85, p_r = 0.06, f_rate = 4)
        
# DS
full_sim_DS = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*0.85, p_r = 0.06, f_rate = 4, n_neurons = int(150*0.85))
        
# DS with 150 neurons
full_sim_DS_synch = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*0.85, p_r = 0.06, f_rate = 4, n_neurons = 4)
        
# DS with neurons in 4 groups
full_sim_DS_groups = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)

#%% Extract histogram

hist_VS, _ = np.histogram(np.log10(full_sim_DS[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)
hist_VS_synch, _ = np.histogram(np.log10(full_sim_DS_synch[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)
hist_VS_groups, _ = np.histogram(np.log10(full_sim_DS_groups[2000:,:,:,:]).flatten(), bins = np.log10(np.logspace(-9,-6,101)), density = True)

#%% Plot DS and VS

fig, (ax1) = plt.subplots(1,1,figsize = (4,3.5), dpi = 400)
x_range = np.logspace(-9,-6,100)
ax1.set_title("[DA] distribution across space and time", fontsize = 10)

ax1.plot(x_range, hist_DS, color = "darkblue", lw = 1.5)
ax1.plot(x_range, hist_DS_synch, color = "blue", lw = 1.5)
ax1.plot(x_range, hist_DS_groups, color = "cornflowerblue", lw = 1.5)

ax1.plot(x_range, hist_VS, color = "darkred", lw = 1.5)
ax1.plot(x_range, hist_VS_synch, color = "red", lw = 1.5)
ax1.plot(x_range, hist_VS_groups, color = "lightcoral", lw = 1.5)

ax1.legend(("Terminals, DS","Neurons, DS","25% synch., DS","Terminals, VS","Neurons, VS","25% synch, VS"),
           ncol = 2, fontsize = 9, frameon = False, handlelength = 1.2, loc = "upper right")


ax1.set_ylabel("Denisty")
ax1.set_xlabel("M")
ax1.set_xscale("log")
ax1.set_xlim(10**-9, 10**-6)
ax1.set_ylim(-0.05,5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


