#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:36:45 2022

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal as ss
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

def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def amp_impulse(t,tau):
    t = t*0.1
    return (-1)**0/(2*t+1)*np.exp(-(2*t+1)**2*t/tau)

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)
#%%
# Simulate release sites
# simulation, space_init, firing, release_sites, var_list = \
#         sim_space_3D(width = 100, depth = 100, dx_dy = 1, time = 2, D = 763,
#                   inter_var_distance = 25, p_r = 0.06, f_rate = 4)
        
# # DS
# full_sim_DS = sim_dynamics_3D(simulation, release_sites, firing, var_list, 
#                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

# Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)

# VS
     
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


#%% Release sites

# Define dimensions
Nx, Ny, Nz = 100, 100, 100
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# Create a figure with 3D ax
fig = plt.figure(figsize=(2.5, 2.5), dpi = 400)
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_proj_type('ortho')

fig.text(0.53, 0.88, "150 neurons\n40,000 release sites", fontsize = 10, ha = "center")

color_cycle = np.tile(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                      4000)
ax1.scatter(release_sites[0,:], release_sites[1,:], -release_sites[2,:], s = 1.5, lw = 0,
            color = color_cycle)

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax1.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=0.8, zorder=1e3)
ax1.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax1.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax1.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

ax1.w_xaxis.line.set_color([0.4,0.4,0.4])
ax1.w_yaxis.line.set_color([0.4,0.4,0.4])
ax1.w_zaxis.line.set_color([0.4,0.4,0.4])

# Set labels and zticks
ax1.set_zlim(-99,0)
ax1.set_zlabel('100 \u00B5m', labelpad = -10)
ax1.set_zticks([])
ax1.set_xlim(0,99)
ax1.set_xlabel('100 \u00B5m', labelpad = -10)
ax1.set_xticks([])
ax1.set_xlim(0,99)
ax1.set_ylabel('100 \u00B5m', labelpad = -10)
ax1.set_yticks([])

# Set distance and angle view
ax1.view_init(30, -50)
ax1.dist = 11

fig.tight_layout()

# %% Steady state image
# Define dimensions
Nx, Ny, Nz = 100, 100, 100
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# Create a figure with 3D ax
fig = plt.figure(figsize=(2.5, 2.5), dpi = 400)
ax2 = fig.add_subplot(111, projection='3d')
ax2.set_proj_type('ortho')

data = np.log10(full_sim[326,:,:,:])

fig.text(0.53, 0.92, "Snapshot of steady state [DA]", fontsize = 10, ha = "center")

cmap = "magma"

kw = {
    'vmin': -8.5,
    'vmax': -6.5,
    'levels': np.linspace(-9, -6, 100),
}
# Plot contour surfaces
for i in range(3):
    _ = ax2.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw, cmap = cmap
    )

    _ = ax2.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw, cmap = cmap
    )
    C = ax2.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw, cmap = cmap
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


# fig.text(0.7, 0.05, "[DA] (nM)")

# Colorbar
cbar = fig.colorbar(C, ax=ax2, fraction=0.022, pad=0.01, ticks = [-6, -7, -8, -9], orientation="horizontal")
cbar.ax.set_xticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
cbar.ax.invert_xaxis()

fig.tight_layout()


#%% Gather hist data
hist_DS, _ = np.histogram(np.log10(full_sim[100:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)
#%% Across time and histogram
x_range = np.logspace(-10,-5,999)


fig, (ax1, ax2) = plt.subplots(2,1, figsize = (2.5,2.5), dpi = 400)

time_points = []
for i in range(3):
    time_points.append(np.argmin(abs(np.linspace(0,2,len(full_sim))-i)))

ax1.set_title("Cross-section", fontsize = 10, color = "k")
ax1.set_ylabel("50 um", labelpad = 7)
ax1.set_xlabel("Seconds")
ax1.set_yticks([])
ax1.set_xticks(time_points)
ax1.set_xticklabels([0,1,2])
ax1.set_xlim(time_points[0],time_points[-1])
im = ax1.imshow(np.log10(full_sim[:,25,25:-25,25].T+10**-10), aspect = 1.2, vmin = -8.5, vmax = -6.5, cmap = "magma")
# ax1.plot([0,6104], [25,25], color = "w", ls = ":", lw = 0.8)



ax2.set_title("[DA] distribution", fontsize = 10)
ax2.plot(x_range, hist_DS, color = "cornflowerblue", lw = 1.5, zorder = 10)
ax2.set_ylabel("Denisty")
ax2.set_xlabel("nM")
ax2.set_xscale("log")
ax2.set_xlim(10**-9, 10**-6)
ax2.set_xticks([10**-9, 10**-8,10**-7, 10**-6])
ax2.set_xticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$','$10^{3}$'])
ax2.set_ylim(0,2)
ax2.set_yticks([0,2])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

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

#%% Plot
rounds = 100
FSCV_all_DS = np.zeros((rounds,sim_burst_FSCV_DS[::100].shape[0]))
for i in range(rounds):
    FSCV_all_DS[i,:] = sim_burst_FSCV_DS[i::100]
    
FSCV_top_DS = FSCV_all_DS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]

# 
deconvolved_signal_DS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS-sim_burst_FSCV_DS[int(start_time/var_list[4]-1)], mode = "full")
# deconvolved_signal_DS = ss.convolve(amp_impulse(np.linspace(0,19,20),0.043), FSCV_top_DS-sim_burst_FSCV_DS[int(start_time/var_list[4]-1)], mode = "full")


# Fit exponential decay
ydata_DS = deconvolved_signal_DS[11:17]
xdata_DS = np.linspace(0,len(ydata_DS)-1,len(ydata_DS))
variables_FSCV_DS, _ = curve_fit(exp_decay, xdata = xdata_DS, ydata = ydata_DS)
uptake_fit_DS = exp_decay(xdata_DS,variables_FSCV_DS[0],variables_FSCV_DS[1])



fig, (ax1) = plt.subplots(1,1,figsize = (2,2.5), dpi = 400, gridspec_kw={"height_ratios": [1]})
ax1.set_title("Single stimulation,\n(and FSCV response)", fontsize = 10)

ax1.plot(np.linspace(0,3,3000),sim_burst_DS-10, color = "cornflowerblue", ls = "-", lw = 1.2, zorder = 10)
ax1.plot(np.linspace(0,3,30)+.05,deconvolved_signal_DS[:-19], color = "darkgrey", ls = "-", lw = 1.2)

# ax1.plot((xdata_DS+12)*0.1,uptake_fit_DS, lw = 1, ls = ":", color = "k")


ax1.legend(("DS, Sim.", "DS, FSCV"), ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
           bbox_to_anchor = [1.05,1.05], loc = "upper right", fontsize = 8)

ax1.plot([1,1],[200,190], lw = 1.5, color = "k")
ax1.text(0.98, 202, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax1.set_xlim(0.8,2)
ax1.set_xticks([1,1.5,2])
ax1.set_xlabel("Seconds")
ax1.set_xticklabels([0,0.5,1])
ax1.set_ylim(-20,200)
ax1.set_ylabel("\u0394[DA] (nM)")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%%% Image of probe

fig, ax = plt.subplots(figsize = (3,2), dpi = 400)

# DA
ax.imshow(np.log10(full_sim[1004,:,:,10]), vmin = -8.5, vmax = -6.5, cmap = "magma")

# Probe
ax.fill_between(x = [46,53], y1 = [40,40], y2 = [100,100], color = "k", lw = 0.8, ls = "-", alpha = 0.5)

cbar = fig.colorbar(C, ax=ax, fraction=0.022, pad=0.01, ticks = [-6, -7, -8, -9], orientation="vertical")
cbar.ax.set_yticklabels(['$10^{3}$', '$10^{2}$', '$10^{1}$','$10^{0}$'])
cbar.ax.invert_yaxis()

ax.set_ylim(0,99)
ax.set_xlim(0,99)
ax.set_yticks([])
ax.set_xticks([])