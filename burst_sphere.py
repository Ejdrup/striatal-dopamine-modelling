#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:22:49 2022

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

#%% Simulate different DS bursts

DS_single_list = np.zeros((100,10))
DS_low_list = np.zeros((100,10))
DS_med_list = np.zeros((100,10))
DS_high_list = np.zeros((100,10))


width = 100
for i in range(1):
    ############## Single
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
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
    full_sim_DS_single = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
    
    ############## Low burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
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
    full_sim_DS_low = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
    
    
    ############## Med burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
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
    full_sim_DS_med = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
    
    
    ############## High burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
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
    full_sim_DS_high = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
    
    DS_single_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_single[25:55,:,50,:], axis = 2)), axis = 0)
    DS_low_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_low[25:55,:,50,:], axis = 2)), axis = 0)
    DS_med_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_med[25:55,:,50,:], axis = 2)), axis = 0)
    DS_high_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_high[25:55,:,50,:], axis = 2)), axis = 0)

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

ax1.plot(10**np.mean(DS_single_list, axis = 1)*10**6, color = "lightblue", lw = 1.5, zorder = 1)
ax1.plot(10**np.mean(DS_low_list, axis = 1)*10**6, color = "cadetblue", lw = 1.5, zorder = 3)
ax1.plot(10**np.mean(DS_med_list, axis = 1)*10**6, color = "royalblue", lw = 1.5, zorder = 5)
ax1.plot(10**np.mean(DS_high_list, axis = 1)*10**6, color = "darkblue", lw = 1.5, zorder = 7)

ax1.legend(("1/10","3/10","6/20","12/40"), title = "APs/Hz", title_fontsize = 8,
           fontsize = 8, handlelength = 1, frameon = False, loc = "upper right", bbox_to_anchor = [1.1,1.02])

ax1.plot(10**np.mean(DS_single_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 0)
ax1.plot(10**np.mean(DS_low_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 2)
ax1.plot(10**np.mean(DS_med_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 4)
ax1.plot(10**np.mean(DS_high_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 6)

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
area_list = [(4/3*np.pi*np.sum(DS_single_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_low_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_med_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_high_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3)]
color_list = ["lightblue", "cadetblue", "royalblue","darkblue"]
label_list = ["1/10","3/10","6/20","12/40"]
for i in range(4):
    ax2.scatter([np.repeat(i,10)+(np.random.rand(10)-0.5)*0.2], area_list[i],
                color = "w", edgecolor = color_list[i], s = 15)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = color_list[i], lw = 1.5, zorder = 10)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = "w", lw = 3.5, zorder = 9)
    
    ax2.text(i,0,label_list[i], ha = "center", va = "top", rotation = 90)

ax2.set_ylim(0,60)
ax2.set_ylabel("Relative to burst origin")
ax2.set_xlim(-0.7,3.5)
ax2.set_xticks([])


ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)

fig.tight_layout()

#%% Figure of spillover

fig, axes = plt.subplots(2,3, figsize = (3.6,2.5), dpi = 400)

axes[0,0].set_title("3 APs/10 Hz", fontsize = 10, color = "cadetblue")
axes[0,1].set_title("6 APs/20 Hz", fontsize = 10, color = "royalblue")
axes[0,2].set_title("12 APs/40 Hz", fontsize = 10, color = "darkblue")
axes[0,0].set_ylabel("End of burst")
axes[1,0].set_ylabel("100 ms after")

axes[0,0].imshow(np.log10(full_sim_DS_low[39,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")
axes[1,0].imshow(np.log10(full_sim_DS_low[45,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")

axes[0,1].imshow(np.log10(full_sim_DS_med[39,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")
axes[1,1].imshow(np.log10(full_sim_DS_med[45,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")

axes[0,2].imshow(np.log10(full_sim_DS_high[39,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")
axes[1,2].imshow(np.log10(full_sim_DS_high[45,:,:,50]), vmin = -8.5, vmax = -6.3, cmap = "magma")

for i in range(6):
    axes.flatten()[i].plot(np.sin(np.linspace(-np.pi,np.pi,100))*5+49.5,
                           np.cos(np.linspace(-np.pi,np.pi,100))*5+49.5,
                           color = "k", lw = .5, ls = "-", zorder = 10)
    axes.flatten()[i].set_xticks([])
    axes.flatten()[i].set_yticks([])
    
axes[1,2].plot([70,90],[92,92], lw = 1.5, color = "w", zorder = 10)

fig.tight_layout()

#%% Figure of spillover v2

fig, axes = plt.subplots(3,3, figsize = (3.1,2.5), dpi = 400, gridspec_kw={"height_ratios":[0.3,1,1]})

axes[0,0].set_title("3 APs/10 Hz", fontsize = 10, color = "cadetblue")
axes[0,1].set_title("6 APs/20 Hz", fontsize = 10, color = "royalblue")
axes[0,2].set_title("12 APs/40 Hz", fontsize = 10, color = "darkblue")
axes[1,0].set_ylabel("End")
axes[2,0].set_ylabel("+ 100 ms")


axes[0,0].plot(np.mean(full_sim_DS_low[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")
axes[0,1].plot(np.mean(full_sim_DS_med[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")
axes[0,2].plot(np.mean(full_sim_DS_high[15:70,45:55,45:55,50], axis = (1,2))*10**9, lw = 0.8, color = "k")

color_list = ["cadetblue", "royalblue", "darkblue"]
no_bursts = [3,6,12]

for i in range(3):
    axes[0,i].plot([0,54],[-500,-500], lw = 0.8, clip_on = False, color = color_list[i])
    bursts =  np.linspace(10,25,no_bursts[i])
    for j in range(len(bursts)):
        axes[0,i].plot([bursts[j],bursts[j]], [-470,-250], lw = 0.8, color = color_list[i], clip_on = False)

for i in range(3):
    axes.flatten()[i].set_ylim(-100,1800)
    axes.flatten()[i].spines["top"].set_visible(False)
    axes.flatten()[i].spines["right"].set_visible(False)
    axes.flatten()[i].spines["left"].set_visible(False)
    axes.flatten()[i].spines["bottom"].set_visible(False)

# axes[0,0].text(-1,850, "0.5 \u00B5M", fontsize = 6, rotation = 90, ha = "right", va = "center")   
axes[0,0].plot([0,0], [1000,1500], lw = 0.8, color = "k")
axes[0,0].plot([0,10], [1500,1500], lw = 0.8, color = "k")


axes[1,0].imshow(np.log10(full_sim_DS_low[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,0].imshow(np.log10(full_sim_DS_low[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

axes[1,1].imshow(np.log10(full_sim_DS_med[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,1].imshow(np.log10(full_sim_DS_med[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

axes[1,2].imshow(np.log10(full_sim_DS_high[39,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")
axes[2,2].imshow(np.log10(full_sim_DS_high[45,:,:,50]), vmin = -8.5, vmax = -6.5, cmap = "magma")

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



#%% Simulate for figure of spillover

############## Med burst
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
        
# Define the area

width = 100

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
full_sim_DS_med = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

############## High burst
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
        
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
full_sim_DS_high = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)

#%% Figure of spillover

fig, axes = plt.subplots(2,3, figsize = (3.6,2.5), dpi = 400)

axes[0,0].set_title("3 APs/10 Hz", fontsize = 10, color = "cadetblue")
axes[0,1].set_title("6 APs/20 Hz", fontsize = 10, color = "royalblue")
axes[0,2].set_title("12 APs/40 Hz", fontsize = 10, color = "darkblue")
axes[0,0].set_ylabel("End of burst")
axes[1,0].set_ylabel("100 ms after")

axes[0,0].imshow(np.log10(full_sim_DS_low[39,:,:,50]), vmin = -8.5, vmax = -6.3)
axes[1,0].imshow(np.log10(full_sim_DS_low[45,:,:,50]), vmin = -8.5, vmax = -6.3)

axes[0,1].imshow(np.log10(full_sim_DS_med[39,:,:,50]), vmin = -8.5, vmax = -6.3)
axes[1,1].imshow(np.log10(full_sim_DS_med[45,:,:,50]), vmin = -8.5, vmax = -6.3)

axes[0,2].imshow(np.log10(full_sim_DS_high[39,:,:,50]), vmin = -8.5, vmax = -6.3)
axes[1,2].imshow(np.log10(full_sim_DS_high[45,:,:,50]), vmin = -8.5, vmax = -6.3)

for i in range(6):
    axes.flatten()[i].plot(np.sin(np.linspace(-np.pi,np.pi,100))*5+49.5,
                           np.cos(np.linspace(-np.pi,np.pi,100))*5+49.5,
                           color = "k", lw = .5, ls = "-", zorder = 10)
    axes.flatten()[i].set_xticks([])
    axes.flatten()[i].set_yticks([])
    
axes[1,2].plot([70,90],[92,92], lw = 1.5, color = "w", zorder = 10)

fig.tight_layout()


#%% Simulate different VS bursts

width = 100

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


#%% Plot VS and DS

plt.plot(np.max(np.log10(np.mean(full_sim_DS_low[25:55,:,50,:], axis = 2)), axis = 0), color = "darkblue")
plt.plot(np.max(np.log10(np.mean(full_sim_DS_med[25:55,:,50,:], axis = 2)), axis = 0), color = "blue")
plt.plot(np.max(np.log10(np.mean(full_sim_DS_high[25:55,:,50,:], axis = 2)), axis = 0), color = "lightblue")

plt.plot(np.max(np.log10(np.mean(full_sim_VS_low[25:55,:,50,:], axis = 2)), axis = 0), color = "darkred")
plt.plot(np.max(np.log10(np.mean(full_sim_VS_med[25:55,:,50,:], axis = 2)), axis = 0), color = "red")
plt.plot(np.max(np.log10(np.mean(full_sim_VS_high[25:55,:,50,:], axis = 2)), axis = 0), color = "pink")
#%% on log scale
plt.plot(np.max(np.mean(full_sim_DS_low[25:55,:,50,:], axis = 2), axis = 0), color = "darkblue")
plt.plot(np.max(np.mean(full_sim_DS_med[25:55,:,50,:], axis = 2), axis = 0), color = "blue")
plt.plot(np.max(np.mean(full_sim_DS_high[25:55,:,50,:], axis = 2), axis = 0), color = "lightblue")

plt.plot(np.max(np.mean(full_sim_VS_low[25:55,:,50,:], axis = 2), axis = 0), color = "darkred")
plt.plot(np.max(np.mean(full_sim_VS_med[25:55,:,50,:], axis = 2), axis = 0), color = "red")
plt.plot(np.max(np.mean(full_sim_VS_high[25:55,:,50,:], axis = 2), axis = 0), color = "pink")
#%%
plt.plot(np.max(np.mean(full_sim_DS_low[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "darkblue") # 43
plt.plot(np.max(np.mean(full_sim_DS_med[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "blue") # 40
plt.plot(np.max(np.mean(full_sim_DS_high[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "lightblue") # 37

plt.plot(np.max(np.mean(full_sim_VS_low[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "darkred") # 39
plt.plot(np.max(np.mean(full_sim_VS_med[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "red") # 35
plt.plot(np.max(np.mean(full_sim_VS_high[25:55,:,50,:], axis = 2), axis = 0) > 100*10**-9, color = "pink") # 29

#%%
# plt.bar([0,1,2], [34**3/49**3,26**3/40**3,18**3/27**3])
n = 3
spread = [34**n,26**n,18**n]
conc = [2,1,0.3]
# plt.scatter(conc,spread)
plt.bar([0,1,2], spread)
#%% Simulate DS burst 

width = 100

# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 6 # Number of action potentials in a burst
burst_rate = 20 # Burst firing rate (Hz)
burst_p_r = 0.5 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Define the area

r_sphere = 5

ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
      (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))

        
# Simulate the dynamics
full_sim_DS = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)


sim_burst_DS = np.mean(full_sim_DS, axis = (1,2,3))*10**9

#%% Simulate VS burst 

width = 100

# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 6 # Number of action potentials in a burst
burst_rate = 20 # Burst firing rate (Hz)
burst_p_r = 0.5 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Define the area

r_sphere = 5

ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
      (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))

        
# Simulate the dynamics
full_sim_VS = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)


sim_burst_VS = np.mean(full_sim_VS, axis = (1,2,3))*10**9

#%% 

for i in range(30):
    plt.plot(np.log10(np.mean(full_sim_DS[50+i,:,50,:], axis = 1)), color = [0,0,1-(i*1/30)], lw = 1.5)

plt.fill_between([45,55], [-8.5,-8.5], [-6,-6], color = "grey")

#%%
plt.plot(np.max(np.log10(np.mean(full_sim_DS[50:80,:,50,:], axis = 2)), axis = 0), "b")

plt.plot(np.max(np.log10(np.mean(full_sim_VS[50:80,:,50,:], axis = 2)), axis = 0), "r")