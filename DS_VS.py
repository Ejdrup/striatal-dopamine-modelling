#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:36:45 2022

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
from scipy.optimize import curve_fit


def sim_space_neurons_3D(width = 100, depth = 10, dx_dy = 1, time = 1, D = 763, 
              inter_var_distance = 2.92, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01):
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
    # Generate simulation space (one snapshot every 10 ms)
    space0 = np.zeros((int(t/Hz), nx, ny, nz))
    # Prep for image of all release sites
    space_ph = np.zeros((nx, ny, nz))
    
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

    return space0, space_ph, firing.T, np.array([x_varico,y_varico, z_varico]), np.array([time, dt, dx_dy, inter_var_distance, Hz])

def sim_dynamics_3D(space0, space_ph, release_sites, firing, var_list, 
                 Q = 3000, uptake_rate = 4*10**-6, Km = 210*10**-9,
                 Ds = 321.7237308146399, ECF = 0.21):
    # print(uptake_rate)
    # print(Q)
    # Extract parameters
    t = var_list[0]
    dt = var_list[1]
    dx_dy = var_list[2]
    Hz = var_list[4]
    
    # DA release per vesicle
    single_vesicle_vol = (4/3*np.pi*(0.025)**3) 
    voxel_volume = (dx_dy)**3 # Volume of single voxel
    single_vesicle_DA = 0.025 * Q/1000 # 0.025 M at Q = 1000
    Q_eff = single_vesicle_vol/voxel_volume * single_vesicle_DA * 1/ECF
    
    
    for i in tqdm(range(int(t/dt)-1)):
        
        # Add release events per time step
        space_ph[release_sites[0,:][np.where(firing[i,:])],
               release_sites[1,:][np.where(firing[i,:])],
               release_sites[2,:][np.where(firing[i,:])]] += Q_eff
        
        # Apply gradient operator to simulate diffusion
        space_ph, u = do_timestep_3D(space_ph, 
                                      uptake_rate, Ds, dt, dx_dy, Km)
        
        # Save snapshot at specified Hz
        if i%int(Hz/dt) == 0:
            space0[int(i/(Hz/dt)),:,:,:] = space_ph
            
        
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


def impulse(t,dt,b,tau):
    return b*(dt/tau)*np.exp(-t/tau)

def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)


#%% Simulate both regions
# Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)

# DS
     
full_sim_DS = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)

# VS
     
full_sim_VS = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)


# Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.6), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)

# VS reduced
     
full_sim_VS_reduced = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)

# %% Steady state image of the two regions

# Define dimensions
Nx, Ny, Nz = 100, 100, 100
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# Create a figure with 3D ax
fig = plt.figure(figsize=(4.5, 2.5), dpi = 400)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_proj_type('ortho')
fig.text(0.26, 0.90, "Dorsal striatum", fontsize = 10, ha = "center")
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_proj_type('ortho')
fig.text(0.745, 0.90, "Ventral striatum", fontsize = 10, ha = "center")
# fig.text(0.53, 0.88, "Snapshot of steady state [DA]", fontsize = 10, ha = "center")

cmap = "magma"

kw = {
    'vmin': -8.5,
    'vmax': -6.5,
    'levels': np.linspace(-9, -6, 100),
}

# Plot contour surfaces of DS
data = np.log10(full_sim_DS[200,:,:,:])
for i in range(3):
    _ = ax1.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw, cmap = cmap,
    )

    _ = ax1.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw, cmap = cmap,
    )
    C = ax1.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw, cmap = cmap,
    )
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax1.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax1.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax1.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax1.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax1.set_zlim(-99,0)
ax1.set_zlabel('100 \u00B5m', labelpad = -12)
ax1.set_zticks([])
ax1.set_xlim(0,99)
ax1.set_xlabel('100 \u00B5m', labelpad = -12)
ax1.set_xticks([])
ax1.set_xlim(0,99)
ax1.set_ylabel('100 \u00B5m', labelpad = -12)
ax1.set_yticks([])

ax1.w_xaxis.line.set_color([0.4,0.4,0.4])
ax1.w_yaxis.line.set_color([0.4,0.4,0.4])
ax1.w_zaxis.line.set_color([0.4,0.4,0.4])

# Set distance and angle view
ax1.view_init(30, -50)
ax1.dist = 11


# Plot contour surfaces of VS
data = np.log10(full_sim_VS[200,:,:,:])
for i in range(3):
    _ = ax2.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw, cmap = cmap,
    )

    _ = ax2.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw, cmap = cmap,
    )
    C = ax2.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw, cmap = cmap,
    )
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax2.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax2.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax2.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax2.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax2.set_zlim(-99,0)
ax2.set_zlabel('100 \u00B5m', labelpad = -12)
ax2.set_zticks([])
ax2.set_xlim(0,99)
ax2.set_xlabel('100 \u00B5m', labelpad = -12)
ax2.set_xticks([])
ax2.set_xlim(0,99)
ax2.set_ylabel('100 \u00B5m', labelpad = -12)
ax2.set_yticks([])

ax2.w_xaxis.line.set_color([0.4,0.4,0.4])
ax2.w_yaxis.line.set_color([0.4,0.4,0.4])
ax2.w_zaxis.line.set_color([0.4,0.4,0.4])

# Set distance and angle view
ax2.view_init(30, -50)
ax2.dist = 11

fig.tight_layout()

# # Colorbar
# cbaxes = fig.add_axes([1, 0.1, 0.25, 0.6]) 
# cbar = fig.colorbar(C, ax=cbaxes, fraction=0.022, pad=0.15, label='[DA] (nM)', ticks = [-6, -7, -8, -9])
# cbar.ax.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
# cbar.ax.invert_yaxis()



#%% Extract percentiles and histogram

dist_DA_DS = np.percentile(full_sim_DS[200:,:,:,:],np.linspace(0,100,201))
DS_mean = np.mean(full_sim_DS[200:,:,:,:])
dist_DA_VS = np.percentile(full_sim_VS[200:,:,:,:],np.linspace(0,100,201))
VS_mean = np.mean(full_sim_VS[200:,:,:,:])
dist_DA_VS_reduced = np.percentile(full_sim_VS_reduced[200:,:,:,:],np.linspace(0,100,201))

hist_DS, _ = np.histogram(np.log10(full_sim_DS[100:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)
hist_VS, _ = np.histogram(np.log10(full_sim_VS[100:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)
hist_VS_reduced, _ = np.histogram(np.log10(full_sim_VS_reduced[100:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)


#%% Plot Histograms with ratio

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (3,2.5), dpi = 400, gridspec_kw={"height_ratios": [3,1]})
x_range = np.logspace(-10,-5,999)
ax1.set_title("[DA] distribution", fontsize = 10)
ax1.plot(x_range, hist_DS, color = "cornflowerblue", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS, color = "indianred", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS_reduced, color = "indianred", lw = 1, zorder = 0, ls = "--")
ax1.set_ylabel("Denisty")
ax1.set_xlabel("nM")
ax1.set_xscale("log")
ax1.set_xlim(10**-9, 10**-6)
ax1.set_xticks([10**-9, 10**-8,10**-7, 10**-6])
ax1.set_xticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$','$10^{3}$'])
ax1.set_ylim(0,5)
ax1.set_yticks([0,5])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.legend(("DS","VS","VS, low"), frameon = False, handlelength = 1.1, fontsize = 8)

# Randoms spots
no_spots = 8
no_bins = 100
x_range = np.logspace(-10,-5,no_bins-1)
spots_DS = np.zeros((no_spots,no_bins-1))
spots_VS = np.zeros((no_spots,no_bins-1))

for i in range(no_spots):
    x_coor = np.random.randint(0,full_sim_VS.shape[1])
    y_coor = np.random.randint(0,full_sim_VS.shape[1])
    z_coor = np.random.randint(0,full_sim_VS.shape[1])
    spots_DS[i,:], _ = np.histogram(np.log10(full_sim_DS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)
    spots_VS[i,:], _ = np.histogram(np.log10(full_sim_VS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)


for i in range(no_spots):
    ax1.plot(x_range, spots_DS[i,:], color = "cornflowerblue", lw = .5, alpha = 0.5)
    ax1.plot(x_range, spots_VS[i,:], color = "indianred", lw = .5, alpha = 0.5)

ax2.plot(dist_DA_VS/dist_DA_DS, color = "dimgrey")
ax2.set_ylabel("Ratio")
ax2.set_xlabel("Percentiles")
ax2.set_ylim(0,6)
ax2.set_yticks([0,6])
ax2.set_xlim(0,100)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout(h_pad = 0)

#%% Plot Histograms and mean ratios

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (2.7,2.5), dpi = 400, gridspec_kw={"height_ratios": [1.5,1]})
x_range = np.logspace(-10,-5,999)
ax1.set_title("[DA] distribution", fontsize = 10)
ax1.plot(x_range, hist_DS, color = "cornflowerblue", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS, color = "indianred", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS_reduced, color = "indianred", lw = 1, zorder = 0, ls = "--")
ax1.set_ylabel("Denisty", labelpad = 15)
ax1.set_xlabel("nM")
ax1.set_xscale("log")
ax1.set_xlim(10**-9, 10**-6)
ax1.set_xticks([10**-9, 10**-8,10**-7, 10**-6])
ax1.set_xticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$','$10^{3}$'])
ax1.set_ylim(0,6)
ax1.set_yticks([0,3,6])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.legend(("DS","VS","VS, low"), frameon = False, handlelength = 1.1, fontsize = 8, loc = "upper right", bbox_to_anchor = [1.05,1])

# Randoms spots
no_spots = 8
no_bins = 100
x_range = np.logspace(-10,-5,no_bins-1)
spots_DS = np.zeros((no_spots,no_bins-1))
spots_VS = np.zeros((no_spots,no_bins-1))

for i in range(no_spots):
    x_coor = np.random.randint(0,full_sim_VS.shape[1])
    y_coor = np.random.randint(0,full_sim_VS.shape[1])
    z_coor = np.random.randint(0,full_sim_VS.shape[1])
    spots_DS[i,:], _ = np.histogram(np.log10(full_sim_DS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)
    spots_VS[i,:], _ = np.histogram(np.log10(full_sim_VS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)


for i in range(no_spots):
    ax1.plot(x_range, spots_DS[i,:], color = "cornflowerblue", lw = .5, alpha = 0.5)
    ax1.plot(x_range, spots_VS[i,:], color = "indianred", lw = .5, alpha = 0.5)
    
    
ax2.plot(dist_DA_DS[:]/DS_mean, color = "cornflowerblue")
ax2.plot(dist_DA_VS[:]/VS_mean, color = "indianred")
ax2.plot([0,200], [1,1], color = "k", ls = ":", lw = 0.8, zorder = 0)
ax2.set_yscale("log")
# ax2.plot(dist_DA_DS[:99]/DS_mean, color = "cornflowerblue")
# ax2.plot(dist_DA_VS[:99]/VS_mean, color = "indianred")
ax2.set_ylabel("Rel. to mean")
ax2.set_xlabel("Percentiles")
ax2.set_ylim(0.2,10)
# ax2.set_yticks([0,6])
ax2.set_xlim(0,200)
ax2.set_xticks([0,50,100,150,200])
ax2.set_xticklabels([0,25,50,75,100])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)   


fig.tight_layout(h_pad = 0)

#%% Plot Histograms and percentiles

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (2.7,2.5), dpi = 400, gridspec_kw={"height_ratios": [1.5,1]})
x_range = np.logspace(-10,-5,999)
ax1.set_title("[DA] distribution", fontsize = 10)
ax1.plot(x_range, hist_DS, color = "cornflowerblue", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS, color = "indianred", lw = 1.5, zorder = 10)
ax1.plot(x_range, hist_VS_reduced, color = "indianred", lw = 1, zorder = 0, ls = "--")
ax1.set_ylabel("Denisty", labelpad = 15)
ax1.set_xlabel("nM")
ax1.set_xscale("log")
ax1.set_xlim(10**-9, 10**-6)
ax1.set_xticks([10**-9, 10**-8,10**-7, 10**-6])
ax1.set_xticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$','$10^{3}$'])
ax1.set_ylim(0,6)
ax1.set_yticks([0,3,6])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.legend(("DS","VS","VS, low"), frameon = False, handlelength = 1.1, fontsize = 8, loc = "upper right", bbox_to_anchor = [1.05,1])

# Randoms spots
no_spots = 8
no_bins = 100
x_range = np.logspace(-10,-5,no_bins-1)
spots_DS = np.zeros((no_spots,no_bins-1))
spots_VS = np.zeros((no_spots,no_bins-1))

for i in range(no_spots):
    x_coor = np.random.randint(0,full_sim_VS.shape[1])
    y_coor = np.random.randint(0,full_sim_VS.shape[1])
    z_coor = np.random.randint(0,full_sim_VS.shape[1])
    spots_DS[i,:], _ = np.histogram(np.log10(full_sim_DS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)
    spots_VS[i,:], _ = np.histogram(np.log10(full_sim_VS[100:,x_coor,y_coor, z_coor]).flatten(), bins = np.log10(np.logspace(-10,-5,no_bins)), density = True)


for i in range(no_spots):
    ax1.plot(x_range, spots_DS[i,:], color = "cornflowerblue", lw = .5, alpha = 0.5)
    ax1.plot(x_range, spots_VS[i,:], color = "indianred", lw = .5, alpha = 0.5)
    
    
ax2.plot(dist_DA_DS*10**9, color = "cornflowerblue")
ax2.plot(dist_DA_VS*10**9, color = "indianred")
ax2.plot(dist_DA_VS_reduced*10**9, color = "indianred", ls = "--", lw = 1)

# ax2.plot([0,200], [1,1], color = "k", ls = ":", lw = 0.8, zorder = 0)
ax2.set_yscale("log")
# ax2.plot(dist_DA_DS[:99]/DS_mean, color = "cornflowerblue")
# ax2.plot(dist_DA_VS[:99]/VS_mean, color = "indianred")
ax2.set_ylabel("nM")
ax2.set_xlabel("Percentiles")
ax2.set_ylim(1, 1000)
# ax2.set_yticks([0,6])
ax2.set_xlim(0,200)
ax2.set_xticks([0,50,100,150,200])
ax2.set_xticklabels([0,25,50,75,100])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)   


fig.tight_layout(h_pad = 0)


#%% Bar of 80-20 distributions

bar_data_ax1 = [np.sum(dist_DA_DS > 20*10**-9)/len(dist_DA_DS)*100,
            np.sum(dist_DA_VS > 20*10**-9)/len(dist_DA_VS)*100,
            np.sum(dist_DA_DS < 20*10**-9)/len(dist_DA_DS)*100,
            np.sum(dist_DA_VS < 20*10**-9)/len(dist_DA_VS)*100]

fig, (ax1) = plt.subplots(1,1,figsize = (1.8,2.5), dpi = 400, gridspec_kw={'width_ratios': [1]})
fig.suptitle("[DA] distribution", fontsize = 10, y = 0.99, x = 0.53)

ax1.scatter([],[], marker = "s", s = 70, color = "cornflowerblue")
ax1.scatter([],[], marker = "s", s = 70, color = "indianred")
fig.legend(("DS","VS"), frameon = False, ncol = 2,columnspacing = 0.7, handletextpad = 0.5,
           bbox_to_anchor=[0.53,0.96], loc = "upper center")


ax1.text(0.5,-28, "Above\n20 nM", ha = "center")
ax1.text(3.5,-28, "Below\n20 nM", ha = "center")

ax1.bar([0,1,3,4], bar_data_ax1, color = ["cornflowerblue", "indianred", "cornflowerblue", "indianred"])
ax1.set_ylabel("Percentage")
ax1.set_ylim(0,100)
ax1.set_xticks([])
# ax1.set_xticklabels(["DS", "VS", "DS", "VS"])

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


fig.tight_layout()

#%% Cross section
time_points = []
for i in range(3):
    time_points.append(np.argmin(abs(np.linspace(0,2,len(full_sim_DS))-i)))

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = (3,2.5), dpi = 400)
# fig.text(-0.02,0.725, "Seperate terminals", va = "center", rotation = 90, color = "darkblue")
# fig.text(-0.02,0.33, "150 neurons", va = "center", rotation = 90, color = "cornflowerblue")
fig.suptitle("Cross-section over time", fontsize = 10, x = 0.56, y = 0.92)

# fig.text(0.042,0.625, "Cross-section", fontsize = 10, rotation = 90, va = "center")

# ax1.set_title("Uncoupled terminals", fontsize = 10, color = "darkblue")
ax1.set_ylabel("DS\n50 \u00B5m", labelpad = 7, color = "cornflowerblue")
ax1.set_yticks([])
ax1.set_xticks(time_points)
ax1.set_xticklabels([])
ax1.set_xlim(time_points[0],time_points[-1])
im = ax1.imshow(np.log10(full_sim_DS[:,50,25:-25,50].T+10**-10), aspect = 1.1, vmin = -8.5, vmax = -6.5, cmap = "magma")
ax1.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)

# ax2.set_title("150 neurons", fontsize = 10, color = "cornflowerblue")
ax2.set_ylabel("VS\n50 \u00B5m", labelpad = 7, color = "indianred")
ax2.set_yticks([])
ax2.set_xticks(time_points)
ax2.set_xticklabels([])
ax2.set_xlim(time_points[0],time_points[-1])
ax2.imshow(np.log10(full_sim_VS[:,51,25:-25,50].T+10**-10), aspect = 1.1, vmin = -8.5, vmax = -6.5, cmap = "magma")
ax2.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)


ax3.plot(np.log10(full_sim_DS[:,50,50,50]+10**-10), color = "cornflowerblue", lw = 1, zorder = 3)
ax3.plot(np.log10(full_sim_VS[:,51,49,50]+10**-10), color = "indianred", lw = 1, zorder = 2)

ax3.set_ylim(-9,-6)
ax3.set_ylabel('[DA] (nM)')
ax3.set_yticks([-6, -7, -8, -9])
ax3.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
ax3.set_xlim(time_points[0],time_points[-1])
ax3.set_xlabel("Seconds")
ax3.set_xticks(time_points)
ax3.set_xticklabels([0,1,2])

fig.tight_layout()

kw = {
    'vmin': -8.5,
    'vmax': -6.5,
    'levels': np.linspace(-9, -6, 100),
}
im = plt.contourf(
    np.log10(full_sim_DS[200,:,:,20]+10**-10),np.log10(full_sim_DS[200,:,:,20]+10**-10),np.log10(full_sim_DS[200,:,:,20]+10**-10),
     **kw, alpha = 1, cmap = "magma"
)
    
cbar_ax = fig.add_axes([1, 0.245, 0.017, 0.555])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [-6, -7, -8, -9])
cbar_ax.set_ylabel('[DA] (nM)', labelpad=5, rotation = 90)
cbar_ax.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
cbar_ax.set_ylim( -6, -9 )

#%% Simulate burst
# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.001)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 1 # Number of action potentials in a burst
burst_rate = 1000 # Burst firing rate (Hz)
burst_p_r = .2 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


sim_burst_DS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS = np.mean(full_sim, axis = (1,2,3))*10**9


# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25*(1/0.8), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.001)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)



sim_burst_VS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS = np.mean(full_sim, axis = (1,2,3))*10**9


#%%
rounds = 100
FSCV_all_DS = np.zeros((rounds,sim_burst_FSCV_DS[::100].shape[0]))
FSCV_all_VS = np.zeros((rounds,sim_burst_FSCV_VS[::100].shape[0]))
for i in range(rounds):
    FSCV_all_DS[i,:] = sim_burst_FSCV_DS[i::100]
    FSCV_all_VS[i,:] = sim_burst_FSCV_VS[i::100]
    
FSCV_top_DS = FSCV_all_DS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_VS = FSCV_all_VS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]

# Calculate FSCV response
# deconvolved_signal_DS = ss.convolve(impulse(np.linspace(0,19,20),0.1,7.3,1.5), np.mean(FSCV_mean_DS,axis = 0)-10, mode = "full")
# deconvolved_signal_VS = ss.convolve(impulse(np.linspace(0,19,20),0.1,7.3,1.5), np.mean(FSCV_mean_VS,axis = 0)-20, mode = "full")

deconvolved_signal_DS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS-sim_burst_FSCV_DS[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_VS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS-sim_burst_FSCV_VS[int(start_time/var_list[4]-1)], mode = "full")


# Fit exponential decay
ydata_DS = deconvolved_signal_DS[12:22]
xdata_DS = np.linspace(0,len(ydata_DS)-1,len(ydata_DS))
variables_FSCV_DS, _ = curve_fit(exp_decay, xdata = xdata_DS, ydata = ydata_DS)
uptake_fit_DS = exp_decay(xdata_DS,variables_FSCV_DS[0],variables_FSCV_DS[1])

ydata_VS = deconvolved_signal_VS[15:26]
xdata_VS = np.linspace(0,len(ydata_VS)-1,len(ydata_VS))
variables_FSCV_VS, _ = curve_fit(exp_decay, xdata = xdata_VS, ydata = ydata_VS)
uptake_fit_VS = exp_decay(xdata_VS,variables_FSCV_VS[0],variables_FSCV_VS[1])


fig, (ax1) = plt.subplots(1,1,figsize = (2.5,2.5), dpi = 400, gridspec_kw={"height_ratios": [1]})
ax1.set_title("Single stimulation", fontsize = 10)

ax1.plot(np.linspace(0,3,3000),sim_burst_DS-10, color = "cornflowerblue", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_DS[:-19], color = "cornflowerblue", ls = "--", lw = 1.2)

ax1.plot(np.linspace(0,3,3000),sim_burst_VS-27, color = "firebrick", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_VS[:-19], color = "firebrick", ls = "--", lw = 1.2)

ax1.plot((xdata_DS+13.7)*0.1,uptake_fit_DS, lw = 1, ls = "-", color = "k")
ax1.plot((xdata_VS+16.7)*0.1,uptake_fit_VS, lw = 1, ls = "-", color = "k")


ax1.legend(("DS, Sim.", "DS, FSCV", "VS, Sim.", "VS, FSCV", "Uptake fit"), ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
           bbox_to_anchor = [1.05,1.05], loc = "upper right", fontsize = 8)

ax1.plot([1,1],[400,380], lw = 2, color = "k")
ax1.text(0.94, 404, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax1.set_xlim(0.5,3)
ax1.set_xticks([1,2,3])
ax1.set_xlabel("Seconds")
ax1.set_xticklabels([0, 1, 2])
ax1.set_ylim(-20,400)
ax1.set_ylabel("\u0394[DA] (nM)")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Two-pane layout

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.1,2.5), dpi = 400, gridspec_kw={"height_ratios": [1]})
fig.suptitle("Single stimulation,\nmodelled FSCV response", fontsize = 10, y = 0.92)

# DS
ax1.plot(np.linspace(0,3,3000),sim_burst_DS-15, color = "dimgrey", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_DS[:-19]+3, color = "cornflowerblue", ls = "--", lw = 1.2)
ax1.legend(("Sim.", "FSCV"), title="DS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05, 1], loc = "upper right", fontsize = 8)
ax1.plot([1,1],[400,380], lw = 1.5, color = "k")
ax1.text(0.94, 405, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax1.set_xlim(0.5,3)
ax1.set_xticks([1,2,3])
ax1.set_xlabel("Seconds")
ax1.set_xticklabels([0, 1, 2])
ax1.set_ylim(-20,400)
ax1.set_ylabel("\u0394[DA] (nM)")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# VS
ax2.plot(np.linspace(0,3,3000),sim_burst_VS-35, color = "dimgrey", ls = "-", lw = 1.2)
ax2.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_VS[:-19], color = "firebrick", ls = "--", lw = 1.2)
ax2.legend(("Sim.", "FSCV"), title="VS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05, 1], loc = "upper right", fontsize = 8)
ax2.plot([1,1],[400,380], lw = 1.5, color = "k")
ax2.text(0.94, 405, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax2.set_xlim(0.5,3)
ax2.set_xticks([1,2,3])
ax2.set_xlabel("Seconds")
ax2.set_xticklabels([0, 1, 2])
ax2.set_ylim(-20,400)
ax2.set_yticks([])
# ax2.set_ylabel("\u0394[DA] (nM)")
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
#%% Wightman data 
#60 Hz

# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 120 # Number of action potentials in a burst
burst_rate = 60 # Burst firing rate (Hz)
burst_p_r = .06 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


sim_burst_DS_60 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS_60 = np.mean(full_sim, axis = (1,2,3))*10**9

# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25*(1/0.8), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)


sim_burst_VS_60 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS_60 = np.mean(full_sim, axis = (1,2,3))*10**9

# Wightman data 30 Hz

# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 120 # Number of action potentials in a burst
burst_rate = 30 # Burst firing rate (Hz)
burst_p_r = .06 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


sim_burst_DS_30 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS_30 = np.mean(full_sim, axis = (1,2,3))*10**9

# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25*(1/0.8), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)


sim_burst_VS_30 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS_30 = np.mean(full_sim, axis = (1,2,3))*10**9

# Wightman data 10 Hz

# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 120 # Number of action potentials in a burst
burst_rate = 10 # Burst firing rate (Hz)
burst_p_r = .06 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


sim_burst_DS_10 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS_10 = np.mean(full_sim, axis = (1,2,3))*10**9

# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 20, D = 763,
                  inter_var_distance = 25*(1/0.8), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)


sim_burst_VS_10 = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS_10 = np.mean(full_sim, axis = (1,2,3))*10**9

#%% Plot Wightman data
rounds = 10
FSCV_all_DS_60 = np.zeros((rounds,sim_burst_FSCV_DS_60[::10].shape[0]))
FSCV_all_VS_60 = np.zeros((rounds,sim_burst_FSCV_VS_60[::10].shape[0]))
FSCV_all_DS_30 = np.zeros((rounds,sim_burst_FSCV_DS_30[::10].shape[0]))
FSCV_all_VS_30 = np.zeros((rounds,sim_burst_FSCV_VS_30[::10].shape[0]))
FSCV_all_DS_10 = np.zeros((rounds,sim_burst_FSCV_DS_10[::10].shape[0]))
FSCV_all_VS_10 = np.zeros((rounds,sim_burst_FSCV_VS_10[::10].shape[0]))

for i in range(rounds):
    FSCV_all_DS_60[i,:] = sim_burst_FSCV_DS_60[i::10]
    FSCV_all_VS_60[i,:] = sim_burst_FSCV_VS_60[i::10]
    FSCV_all_DS_30[i,:] = sim_burst_FSCV_DS_30[i::10]
    FSCV_all_VS_30[i,:] = sim_burst_FSCV_VS_30[i::10]
    FSCV_all_DS_10[i,:] = sim_burst_FSCV_DS_10[i::10]
    FSCV_all_VS_10[i,:] = sim_burst_FSCV_VS_10[i::10]
    
FSCV_top_DS_60 = FSCV_all_DS_60[int(np.argmax(FSCV_all_DS_60)/(FSCV_all_DS_60.shape[1])),:]
FSCV_top_VS_60 = FSCV_all_VS_60[int(np.argmax(FSCV_all_DS_60)/(FSCV_all_VS_60.shape[1])),:]
FSCV_top_DS_30 = FSCV_all_DS_30[int(np.argmax(FSCV_all_DS_30)/(FSCV_all_DS_30.shape[1])),:]
FSCV_top_VS_30 = FSCV_all_VS_30[int(np.argmax(FSCV_all_DS_30)/(FSCV_all_VS_30.shape[1])),:]
FSCV_top_DS_10 = FSCV_all_DS_10[int(np.argmax(FSCV_all_DS_10)/(FSCV_all_DS_10.shape[1])),:]
FSCV_top_VS_10 = FSCV_all_VS_10[int(np.argmax(FSCV_all_DS_10)/(FSCV_all_VS_10.shape[1])),:]


# FSCV_top_DS = sim_burst_FSCV_DS[::100]
# FSCV_top_VS = sim_burst_FSCV_VS[::100]

# Calculate FSCV response
# deconvolved_signal_DS = ss.convolve(impulse(np.linspace(0,19,20),0.1,7.3,1.5), np.mean(FSCV_mean_DS,axis = 0)-10, mode = "full")
# deconvolved_signal_VS = ss.convolve(impulse(np.linspace(0,19,20),0.1,7.3,1.5), np.mean(FSCV_mean_VS,axis = 0)-20, mode = "full")

deconvolved_signal_DS_60 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_DS_60-sim_burst_FSCV_DS_60[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_VS_60 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_VS_60-sim_burst_FSCV_VS_60[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_DS_30 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_DS_30-sim_burst_FSCV_DS_30[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_VS_30 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_VS_30-sim_burst_FSCV_VS_30[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_DS_10 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_DS_10-sim_burst_FSCV_DS_10[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_VS_10 = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,0.012), FSCV_top_VS_10-sim_burst_FSCV_VS_10[int(start_time/var_list[4]-1)], mode = "full")


# Fit exponential decay

ydata_DS = deconvolved_signal_DS_60[32:50]
xdata_DS = np.linspace(0,len(ydata_DS)-1,len(ydata_DS))
variables_FSCV_DS, _ = curve_fit(exp_decay, xdata = xdata_DS, ydata = ydata_DS)
uptake_fit_DS = exp_decay(xdata_DS,variables_FSCV_DS[0],variables_FSCV_DS[1])

ydata_VS = deconvolved_signal_VS_60[60:70]
xdata_VS = np.linspace(0,len(ydata_VS)-1,len(ydata_VS))
variables_FSCV_VS, _ = curve_fit(exp_decay, xdata = xdata_VS, ydata = ydata_VS)
uptake_fit_VS = exp_decay(xdata_VS,variables_FSCV_VS[0],variables_FSCV_VS[1])


fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.8,2.7), dpi = 400, gridspec_kw={"height_ratios": [1]})
fig.suptitle("120 pulses,\nmodelled FSCV response", fontsize = 10, y = 0.88)

#DS
# ax1.set_title("DS, simulated FSCV", fontsize = 10, pad = 18)
# ax1.plot(np.linspace(0,5,5000),sim_burst_DS-10, color = "cornflowerblue", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,20,200)-1,deconvolved_signal_DS_60[:-19], color = "cornflowerblue", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,20,200)-1,deconvolved_signal_DS_30[:-19], color = "cornflowerblue", ls = "--", lw = 1.2)
ax1.plot(np.linspace(0,20,200)-1,deconvolved_signal_DS_10[:-19], color = "cornflowerblue", ls = ":", lw = 1.2)

ax1.plot([0,2], [2100, 2100], clip_on = False, color = "k")
ax1.text(-0.2,2150, "60", fontsize = 7, ha = "left")
ax1.plot([0,4], [2050, 2050], clip_on = False, color = "k")
ax1.text(4.2,2100, "30", fontsize = 7, ha = "right")
ax1.plot([0,12], [2000, 2000], clip_on = False, color = "k")
ax1.text(12,2050, "10 Hz", fontsize = 7, ha = "right")

ax1.legend(("60 Hz", "30 Hz", "10 Hz"), title="DS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05,.9], loc = "upper right", fontsize = 8)

ax1.set_xlim(-3,15)
ax1.set_xticks([0,5,10,15])
ax1.set_ylim(-20,2000)
ax1.set_yticks([0,1000,2000])
ax1.set_yticklabels([0,1,2])
ax1.set_xlabel("Seconds")
ax1.set_ylabel("\u0394[DA] (\u00B5M)")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# VS
# ax2.set_title("VS, simulated FSCV", fontsize = 10, pad = 18)
# ax1.plot(np.linspace(0,5,5000),sim_burst_VS-27, color = "firebrick", ls = "-", lw = 1.2)
ax2.plot(np.linspace(0,20,200)-1,deconvolved_signal_VS_60[:-19], color = "indianred", ls = "-", lw = 1.2)
ax2.plot(np.linspace(0,20,200)-1,deconvolved_signal_VS_30[:-19], color = "indianred", ls = "--", lw = 1.2)
ax2.plot(np.linspace(0,20,200)-1,deconvolved_signal_VS_10[:-19], color = "indianred", ls = ":", lw = 1.2)

ax2.plot([0,2], [9450, 9450], clip_on = False, color = "k")
ax2.text(-0.2,9675, "60", fontsize = 7, ha = "left")
ax2.plot([0,4], [9225, 9225], clip_on = False, color = "k")
ax2.text(4.2,9450, "30", fontsize = 7, ha = "right")
ax2.plot([0,12], [9000, 9000], clip_on = False, color = "k")
ax2.text(12,9225, "10 Hz", fontsize = 7, ha = "right")

ax2.legend(("60 Hz", "30 Hz", "10 Hz"), title="VS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05,.9], loc = "upper right", fontsize = 8)

ax2.set_xlim(-3,15)
ax2.set_xticks([0,5,10,15])
ax2.set_ylim(-100,9000)
ax2.set_yticks([0,3000,6000,9000])
ax2.set_yticklabels([0,3,6,9])
ax2.set_xlabel("Seconds")
ax2.set_ylabel("\u0394[DA] (\u00B5M)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# ax1.legend(("DS, Sim.", "DS, FSCV", "VS, Sim.", "VS, FSCV", "Uptake fit"), ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
#            bbox_to_anchor = [1.05,1.05], loc = "upper right", fontsize = 8)


fig.tight_layout()

#%% Actual DA traces

# fig, (ax1, ax2) = plt.subplots(1,2,figsize = (4,2.7), dpi = 400, gridspec_kw={"height_ratios": [1]})
# fig.suptitle("Simulated FSCV", fontsize = 10, y = 0.87)

# #DS
# # ax1.set_title("DS, simulated FSCV", fontsize = 10, pad = 18)
# # ax1.plot(np.linspace(0,5,5000),sim_burst_DS-10, color = "cornflowerblue", ls = "-", lw = 1.2)
# ax1.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_DS_60, color = "cornflowerblue", ls = "-", lw = 1.2)
# ax1.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_DS_30, color = "cornflowerblue", ls = "--", lw = 1.2)
# ax1.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_DS_10, color = "cornflowerblue", ls = ":", lw = 1.2)

# # ax1.plot([0,2], [2100, 2100], clip_on = False, color = "k")
# # ax1.text(0,2150, "60", fontsize = 7, ha = "left")
# # ax1.plot([0,4], [2050, 2050], clip_on = False, color = "k")
# # ax1.text(4,2100, "30", fontsize = 7, ha = "right")
# # ax1.plot([0,12], [2000, 2000], clip_on = False, color = "k")
# # ax1.text(12,2050, "10 Hz", fontsize = 7, ha = "right")

# ax1.legend(("60 Hz", "30 Hz", "10 Hz"), title="DS", title_fontsize = 8,
#            ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
#             bbox_to_anchor = [1.05,.8], loc = "upper right", fontsize = 8)

# ax1.set_xlim(-3,15)
# ax1.set_xticks([0,5,10,15])
# ax1.set_ylim(-20,1000)
# # ax1.set_yticks([0,1000,2000])
# # ax1.set_yticklabels([0,1,2])
# ax1.set_xlabel("Seconds")
# ax1.set_ylabel("[DA] (\u00B5M)")
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)

# # VS
# # ax2.set_title("VS, simulated FSCV", fontsize = 10, pad = 18)
# # ax1.plot(np.linspace(0,5,5000),sim_burst_VS-27, color = "firebrick", ls = "-", lw = 1.2)
# ax2.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_VS_60, color = "indianred", ls = "-", lw = 1.2)
# ax2.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_VS_30, color = "indianred", ls = "--", lw = 1.2)
# ax2.plot(np.linspace(0,20,2000)-1,sim_burst_FSCV_VS_10, color = "indianred", ls = ":", lw = 1.2)

# # ax2.plot([0,2], [9450, 9450], clip_on = False, color = "k")
# # ax2.text(0,9675, "60", fontsize = 7, ha = "left")
# # ax2.plot([0,4], [9225, 9225], clip_on = False, color = "k")
# # ax2.text(4,9450, "30", fontsize = 7, ha = "right")
# # ax2.plot([0,12], [9000, 9000], clip_on = False, color = "k")
# # ax2.text(12,9225, "10 Hz", fontsize = 7, ha = "right")

# ax2.legend(("60 Hz", "30 Hz", "10 Hz"), title="VS", title_fontsize = 8,
#            ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
#             bbox_to_anchor = [1.05,.8], loc = "upper right", fontsize = 8)

# ax2.set_xlim(-3,15)
# ax2.set_xticks([0,5,10,15])
# ax2.set_ylim(-60,3000)
# # ax2.set_yticks([0,3000,6000,9000])
# # ax2.set_yticklabels([0,3,6,9])
# ax2.set_xlabel("Seconds")
# ax2.set_ylabel("[DA] (\u00B5M)")
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)


# ax1.legend(("DS, Sim.", "DS, FSCV", "VS, Sim.", "VS, FSCV", "Uptake fit"), ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
#            bbox_to_anchor = [1.05,1.05], loc = "upper right", fontsize = 8)




# fig.tight_layout()

#%% Simulate burst overflow


VS_single_list = np.zeros((100,10))
VS_low_list = np.zeros((100,10))
VS_med_list = np.zeros((100,10))
VS_high_list = np.zeros((100,10))


width = 100
for i in range(10):
    ############## Single
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 1 # Number of action potentials in a burst
    burst_rate = 1000 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_VS_single = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)
    
    ############## Low burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 3 # Number of action potentials in a burst
    burst_rate = 10 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_VS_low = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)
    
    
    ############## Med burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 6 # Number of action potentials in a burst
    burst_rate = 20 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_VS_med = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)
    
    
    ############## High burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 12 # Number of action potentials in a burst
    burst_rate = 40 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_VS_high = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)
    
    VS_single_list[:,i] = np.max(np.log10(np.mean(full_sim_VS_single[25:55,:,50,:], axis = 2)), axis = 0)
    VS_low_list[:,i] = np.max(np.log10(np.mean(full_sim_VS_low[25:55,:,50,:], axis = 2)), axis = 0)
    VS_med_list[:,i] = np.max(np.log10(np.mean(full_sim_VS_med[25:55,:,50,:], axis = 2)), axis = 0)
    VS_high_list[:,i] = np.max(np.log10(np.mean(full_sim_VS_high[25:55,:,50,:], axis = 2)), axis = 0)
#%%

# fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [2.5,1]})
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [3,1.5]})
ax1.set_title("Spill-over from burst\n(40 terminals)", fontsize = 10)
ax1.set_ylabel("Peak DA after burst (\u00B5M)")
ax1.set_xlabel("Distance (\u00B5m)")
ax1.text(43,3.02, "Burst origin", rotation = 90, va = "top", ha = "right", color = "grey")

# ax1.plot(np.median(DS_low_list, axis = 1)*10**6, color = "cadetblue", lw = 1.5, zorder = 1)
# ax1.plot(np.median(DS_med_list, axis = 1)*10**6, color = "royalblue", lw = 1.5, zorder = 3)
# ax1.plot(np.median(DS_high_list, axis = 1)*10**6, color = "darkblue", lw = 1.5, zorder = 5)

# ax1.plot(np.median(DS_low_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 0)
# ax1.plot(np.median(DS_med_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 2)
# ax1.plot(np.median(DS_high_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 4)

ax1.plot(10**np.mean(VS_single_list, axis = 1)*10**6, color = "rosybrown", lw = 1.5, zorder = 1)
ax1.plot(10**np.mean(VS_low_list, axis = 1)*10**6, color = "lightcoral", lw = 1.5, zorder = 3)
ax1.plot(10**np.mean(VS_med_list, axis = 1)*10**6, color = "sienna", lw = 1.5, zorder = 5)
ax1.plot(10**np.mean(VS_high_list, axis = 1)*10**6, color = "darkred", lw = 1.5, zorder = 7)

ax1.legend(("1/10","3/10","6/20","12/40"), title = "APs/Hz", title_fontsize = 8,
           fontsize = 8, handlelength = 1, frameon = False, loc = "upper right", bbox_to_anchor = [1.1,1.02])

ax1.plot(10**np.mean(VS_single_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 0)
ax1.plot(10**np.mean(VS_low_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 2)
ax1.plot(10**np.mean(VS_med_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 4)
ax1.plot(10**np.mean(VS_high_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 6)

ax1.fill_between([44.5,54.5], [0,0], [3,3], color = "darkgrey", zorder = -1)

ax1.set_ylim(0,3)
ax1.set_xlim(0,99)
ax1.set_xticks([0,49.5,99])
ax1.set_xticklabels([-50,0,50])

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)


# sphere_100nm = [((49.5-np.where(10**np.mean(DS_low_list, axis = 1)*10**6 > 0.1)[0][0])*2)**3,
#                 ((49.5-np.where(10**np.mean(DS_med_list, axis = 1)*10**6 > 0.1)[0][0])*2)**3,
#                 ((49.5-np.where(10**np.mean(DS_high_list, axis = 1)*10**6 > 0.1)[0][0])*2)**3]

# ax2.bar([0,1,2], sphere_100nm)

# ax2.spines["right"].set_visible(False)
# ax2.spines["top"].set_visible(False)


ax2.set_title("Volume\n(>100 nM)", fontsize = 10)
area_list = [(4/3*np.pi*np.sum(VS_single_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(VS_low_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(VS_med_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(VS_high_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3)]
color_list = ["rosybrown", "lightcoral", "sienna","darkred"]
label_list = ["1/10","3/10","6/20","12/40"]
for i in range(4):
    ax2.scatter([np.repeat(i,10)+(np.random.rand(10)-0.5)*0.2], area_list[i],
                color = "w", edgecolor = color_list[i], s = 15)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = color_list[i], lw = 1.5, zorder = 10)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = "w", lw = 3.5, zorder = 9)
    
    ax2.text(i,0,label_list[i], ha = "center", va = "top", rotation = 90)

ax2.set_ylim(0,200)
ax2.set_ylabel("Relative to burst origin")
ax2.set_xlim(-0.7,3.5)
ax2.set_xticks([])


ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)

fig.tight_layout()

#%% Figure of spillover v2

fig, axes = plt.subplots(3,3, figsize = (3.1,2.5), dpi = 400, gridspec_kw={"height_ratios":[0.3,1,1]})

axes[0,0].set_title("3 APs/10 Hz", fontsize = 10, color = "lightcoral")
axes[0,1].set_title("6 APs/20 Hz", fontsize = 10, color = "sienna")
axes[0,2].set_title("12 APs/40 Hz", fontsize = 10, color = "darkred")
axes[1,0].set_ylabel("End")
axes[2,0].set_ylabel("+ 100 ms")


axes[0,0].plot(np.mean(full_sim_VS_low[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")
axes[0,1].plot(np.mean(full_sim_VS_med[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")
axes[0,2].plot(np.mean(full_sim_VS_high[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")

color_list = ["lightcoral", "sienna", "darkred"]
no_bursts = [3,6,12]

for i in range(3):
    axes[0,i].plot([0,54],[-500,-500], lw = 0.8, clip_on = False, color = color_list[i])
    bursts =  np.linspace(10,25,no_bursts[i])
    for j in range(len(bursts)):
        axes[0,i].plot([bursts[j],bursts[j]], [-470,-250], lw = 0.8, color = color_list[i], clip_on = False)

for i in range(3):
    axes.flatten()[i].set_ylim(-100,2200)
    axes.flatten()[i].spines["top"].set_visible(False)
    axes.flatten()[i].spines["right"].set_visible(False)
    axes.flatten()[i].spines["left"].set_visible(False)
    axes.flatten()[i].spines["bottom"].set_visible(False)

# axes[0,0].text(-1,850, "0.5 \u00B5M", fontsize = 6, rotation = 90, ha = "right", va = "center")   
axes[0,0].plot([0,0], [1000,1500], lw = 0.8, color = "k")
axes[0,0].plot([0,10], [1500,1500], lw = 0.8, color = "k")


axes[1,0].imshow(np.log10(full_sim_VS_low[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,0].imshow(np.log10(full_sim_VS_low[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

axes[1,1].imshow(np.log10(full_sim_VS_med[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,1].imshow(np.log10(full_sim_VS_med[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

axes[1,2].imshow(np.log10(full_sim_VS_high[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,2].imshow(np.log10(full_sim_VS_high[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

for i in range(9):
    if i > 2:
        axes.flatten()[i].plot(np.sin(np.linspace(-np.pi,np.pi,100))*5+49.5,
                               np.cos(np.linspace(-np.pi,np.pi,100))*5+49.5,
                               color = "k", lw = .5, ls = "-", zorder = 10)
        # if i < 6:
            # axes.flatten()[i].plot([39.5,59.5],[49.5,49.5], lw = 0.5, color = "k", zorder = 10)
    axes.flatten()[i].set_xticks([])
    axes.flatten()[i].set_yticks([])
    
axes[2,2].plot([70,90],[92,92], lw = 1.5, color = "w", zorder = 10)

fig.tight_layout(h_pad = 0.8, w_pad = 0.7)
