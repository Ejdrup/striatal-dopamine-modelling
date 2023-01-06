#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:50:02 2022

@author: ejdrup
"""
#%% Load packages and functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
from scipy.optimize import curve_fit


def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)

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

#%% Simulate percentiles across different Vmax'es
vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_DS = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS = np.zeros((3,len(vmax_range)))

simulation_DS, space_init_DS, firing_DS, release_sites_DS, var_list_DS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, vmax_val in enumerate(vmax_range):
    print(i)
    
    # DS
    full_sim = sim_dynamics_3D(simulation_DS, space_init_DS, release_sites_DS, firing_DS, var_list_DS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
    
    
    # VS
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
#%% Plot vmax effect on percentiles
fig, (ax1) = plt.subplots(1,1,figsize = (2.6,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

ax1.set_title("Effect of DAT V$_{max}$ on [DA]", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax1.set_ylim(0,150)
ax1.set_xlim(0*10**-6,8*10**-6)
ax1.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6,8*10**-6])
ax1.set_xticklabels([0,2,4,6,8])
color_list = ["black","grey","lightgrey"][::-1]

DS_max_idx = np.argmin(abs(vmax_range-(7.1)*10**-6))
DS_min_idx = np.argmin(abs(vmax_range-(3.1)*10**-6))

VS_max_idx = np.argmin(abs(vmax_range-(2.6)*10**-6))
VS_min_idx = np.argmin(abs(vmax_range-(0.6)*10**-6))

linestyles = [":","--","-"]

ax1.plot(vmax_range,vmax_percentiles_DS[1,:]*10**9, color = "cornflowerblue")
ax1.plot(vmax_range,vmax_percentiles_VS[1,:]*10**9, color = "indianred")
    # ax1.plot(vmax_range,vmax_percentiles_VS[i,:]*10**9, color = "indianred")
    
# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax1.legend(("DS", "VS"), frameon = True,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right", bbox_to_anchor = [1.05,0.5])
legend.set_title('50$^{th}$ %',prop={'size': 8})

ax1.text((vmax_range[DS_max_idx]-vmax_range[DS_min_idx])/2+vmax_range[DS_min_idx],47,
          "DS V$_{max}$\n2\u03c3 literature range", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [vmax_range[DS_min_idx],vmax_range[DS_max_idx]], y1 = [150,150], 
                  color = "cornflowerblue", alpha = 0.5, lw = 0)
ax1.text((vmax_range[VS_max_idx]-vmax_range[VS_min_idx])/2+vmax_range[VS_min_idx],47,
          "VS V$_{max}$\n2\u03c3 literature range", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [vmax_range[VS_min_idx],vmax_range[VS_max_idx]], y1 = [150,150], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)



# ax2.set_title("DAT effect V$_{max}$ on [DA]", fontsize = 10)
# ax2.set_xlabel("DA Percentiles")
# ax2.set_ylim(0,150)
# ax2.set_ylabel("nM")

# for i in range(3):
#     ax2.plot([i-0.1,i-0.1],[vmax_percentiles_DS[i,DS_max_idx]*10**9,vmax_percentiles_DS[i,DS_min_idx]*10**9],
#              color = "cornflowerblue", lw = 2)
#     ax2.plot([i+0.1,i+0.1],[vmax_percentiles_VS[i,VS_max_idx]*10**9,vmax_percentiles_VS[i,VS_min_idx]*10**9],
#              color = "indianred", lw = 2)
    
# ax2.set_xticks([0,1,2])
# ax2.set_xticklabels(["10$^{th}$","50$^{th}$","90$^{th}$"])
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

fig.tight_layout()
#%% Plot of spread
fig, (ax1) = plt.subplots(1,figsize = (1.5,3), dpi = 400)
ax1.set_title("Vmax effect on DA levels", fontsize = 10)
ax1.set_xlabel("DA Percentiles")
ax1.set_ylabel("DA change (nM)")

for i in range(3):
    ax1.plot([i-0.1,i-0.1],[vmax_percentiles[i,DS_max_idx]*10**9,vmax_percentiles[i,DS_min_idx]*10**9],
             color = "darkblue", lw = 2)
    ax1.plot([i+0.1,i+0.1],[vmax_percentiles[i,VS_max_idx]*10**9,vmax_percentiles[i,VS_min_idx]*10**9],
             color = "darkred", lw = 2)
    
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(["10$^{th}$","50$^{th}$","90$^{th}$"])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


#%% Vmax dynamics at different Qs

vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_VS_1000 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_3000 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_10000 = np.zeros((3,len(vmax_range)))
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, vmax_val in enumerate(vmax_range):
    print(i)
    # VS 1000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 1000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_1000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])

    # VS 3000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_3000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
    
    # VS 3000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 10000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_10000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
    
#%% Vmax dynamics at different R%

vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_VS_3 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_6 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_12 = np.zeros((3,len(vmax_range)))
        
for i, vmax_val in enumerate(vmax_range):
    print(i)
    # VS 3%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.03, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_3[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])

    # VS 6%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_6[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
    
    # VS 12%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.30, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_12[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99])
     
#%% Vmax at different Qs

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5.5,2.7), dpi = 400, gridspec_kw={"width_ratios": [1,1]})

ax1.set_title("Effect of Q and DAT V$_{max}$ on [DA]", fontsize = 10)
ax1.set_ylabel("Relative [DA]")
ax1.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax1.set_ylim(0,1)
ax1.set_xlim(0*10**-6,8*10**-6)
ax1.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6,8*10**-6])
ax1.set_xticklabels([0,2,4,6,8])
color_list = ["black","grey","lightgrey"][::-1]

DS_max_idx = np.argmin(abs(vmax_range-(4.5+4.5*0.17)*10**-6))
DS_min_idx = np.argmin(abs(vmax_range-(4.5-4.5*0.17)*10**-6))

VS_max_idx = np.argmin(abs(vmax_range-(1.5+4.5*0.17)*10**-6))
VS_min_idx = np.argmin(abs(vmax_range-(1.5-4.5*0.17)*10**-6))

# linestyles = [":","--","-"]

# ax1.plot(vmax_range,vmax_percentiles_DS[1,:]*10**9, color = "cornflowerblue")
# ax1.plot(vmax_range,vmax_percentiles_VS[1,:]*10**9, color = "indianred")
ax1.plot(vmax_range,(vmax_percentiles_VS_1000[1,:]/np.max(vmax_percentiles_VS_1000[1,:])),
         color = "indianred", ls = "-")
ax1.plot(vmax_range,(vmax_percentiles_VS_3000[1,:]/np.max(vmax_percentiles_VS_3000[1,:])),
         color = "indianred", ls = "--")
ax1.plot(vmax_range,(vmax_percentiles_VS_10000[1,:]/np.max(vmax_percentiles_VS_10000[1,:])), 
         color = "indianred", ls = ":")

# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax1.legend(("1000", "3000", "10000"), frameon = False,
            handlelength = 1.6, prop={'size': 8}, loc = "upper right")
legend.set_title('Q',prop={'size': 8})


ax1.text((vmax_range[VS_max_idx]-vmax_range[VS_min_idx])/2+vmax_range[VS_min_idx],0.35,
          "VS V$_{max}$\n2\u03c3 literature range", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [vmax_range[VS_min_idx],vmax_range[VS_max_idx]], y1 = [1,1], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# With varing R%
ax2.set_title("Effect of R$_\%$ and DAT V$_{max}$ on [DA]", fontsize = 10)
ax2.set_ylabel("Relative [DA]")
ax2.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax2.set_ylim(0,1)
ax2.set_xlim(0*10**-6,8*10**-6)
ax2.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6,8*10**-6])
ax2.set_xticklabels([0,2,4,6,8])

ax2.plot(vmax_range,(vmax_percentiles_VS_3[1,:]/np.max(vmax_percentiles_VS_3[1,:])),
         color = "indianred", ls = "-")
ax2.plot(vmax_range,(vmax_percentiles_VS_6[1,:]/np.max(vmax_percentiles_VS_6[1,:])),
         color = "indianred", ls = "--")
ax2.plot(vmax_range,(vmax_percentiles_VS_12[1,:]/np.max(vmax_percentiles_VS_12[1,:])), 
         color = "indianred", ls = ":")

# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax2.legend(("3%", "6%", "12%"), frameon = False,
            handlelength = 1.6, prop={'size': 8}, loc = "upper right")
legend.set_title('R$_\%$',prop={'size': 8})


ax2.text((vmax_range[VS_max_idx]-vmax_range[VS_min_idx])/2+vmax_range[VS_min_idx],0.35,
          "VS V$_{max}$\n2\u03c3 literature range", ha = "center", fontsize = 8, rotation = 90)
ax2.fill_between(x = [vmax_range[VS_min_idx],vmax_range[VS_max_idx]], y1 = [1,1], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

#%% Schematic figure for Vmax

def MM_uptake(c, vmax, km):
    return (vmax*c)/(km+c)

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.35,2.5), dpi = 400)

DA_range = np.linspace(0,2000,2001)
uptake_high = MM_uptake(DA_range, 4500, 210)
uptake_low = MM_uptake(DA_range, 1500, 210)

ax1.set_title("Uptake rate\n(V$_{max}$ dependence)", fontsize = 10)
ax1.set_ylabel("1.5 \u00B5M s$^{-1}$", fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.plot(DA_range, uptake_low, color = "forestgreen")
ax1.plot([0,2000], [1500,1500], color = "dimgrey", ls = "--",zorder = 0)
ax1.plot([210,210], [0,MM_uptake(210, 1500, 210)], color = "dimgrey",zorder = 0)
ax1.plot([0,210], [MM_uptake(210, 1500, 210),MM_uptake(210, 1500, 210)], color = "dimgrey",zorder = 0)
ax1.set_xlim(0,2000)
ax1.set_ylim(0,5000)

ax2.set_ylabel("4.5 \u00B5M s$^{-1}$", fontsize = 10)
ax2.text(60,3700, "V$_{max}$", fontsize = 8, color = "dimgrey")
ax2.text(320,200*5/3.5, "K$_{m}$", fontsize = 8, color = "dimgrey", rotation = 90)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.plot(DA_range, uptake_high, color = "forestgreen")
ax2.plot([0,2000], [4500,4500], color = "dimgrey", ls = "--",zorder = 0)
ax2.plot([210,210], [0,MM_uptake(840, 4500, 840)], color = "dimgrey",zorder = 0)
ax2.plot([0,210], [MM_uptake(210, 4500, 210),MM_uptake(210, 4500, 210)], color = "dimgrey",zorder = 0)
ax2.set_xlim(0,2000)
ax2.set_ylim(0,5000)

fig.tight_layout()

 
#%% Something with Km akin to the Vmax range testing

km_range = np.linspace(210,6000,25)*10**-9
# km_range = np.linspace(1*10**-9,50000*10**-9,20)
km_percentiles_DS = np.zeros((3,len(km_range)))
km_mean_DS = np.zeros((len(km_range),))
km_percentiles_VS = np.zeros((3,len(km_range)))
km_mean_VS = np.zeros((len(km_range),))

# DS
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 25, depth = 25, dx_dy = 1, time = 30, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, km_val in enumerate(km_range):
    print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 4.5*10**-6, Km = km_val, Ds = 321.7237308146399)
    
    km_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    km_mean_DS[i] = np.mean(full_sim[int(full_sim.shape[0]/2):,:,:,:])
 
# VS    
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 25, depth = 25, dx_dy = 1, time = 30, D = 763,
                  inter_var_distance = (1/0.85)*25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
for i, km_val in enumerate(km_range):
    print(i)
    full_sim = sim_dynamics_3D(simulation , space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 1.55*10**-6, Km = km_val, Ds = 321.7237308146399)
    
    km_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    km_mean_VS[i] = np.mean(full_sim[int(full_sim.shape[0]/2):,:,:,:])
    
#%% Plot Km effect on percentiles
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5,2.5), dpi = 400)
ax1.set_title("Effect of K$_\mathrm{m}$ on DA levels", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("DAT K$_\mathrm{m}$ (\u00B5M)")
ax1.set_ylim(0,800)
ax1.set_xlim(0*10**-6,6*10**-6)
ax1.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6])
ax1.set_xticklabels([0,2,4,6])
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.5, prop={'size': 8}, loc = "upper left")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Dopamine\npercentiles',prop={'size': 8})


linestyles = [":","--","-"]
# DS and VS lines
for i in range(3):
    ax1.plot(km_range,km_percentiles_DS[i,:]*10**9, color = "cornflowerblue", ls = linestyles[i])
    ax1.plot(km_range,km_percentiles_VS[i,:]*10**9, color = "indianred", ls = linestyles[i])
    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("DAT K$_\mathrm{m}$ (\u00B5M)")
ax2.set_xlim(0*10**-6,6*10**-6)
ax2.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6])
ax2.set_xticklabels([0,2,4,6])
ax2.set_ylabel("Fold over baseline")
ax2.set_title("Compared to no inhibition", fontsize = 10)
ax2.set_ylim(0,30)


ax2.plot(km_range[:], km_mean_VS[:]/km_mean_VS[0]+0.5, color = "cornflowerblue")
ax2.plot(km_range[:], km_mean_VS[:]/km_mean_VS[0], color = "indianred")
legend = ax2.legend(("DS","VS"), frameon = False,
            handlelength = 1.5, fontsize = 8, loc = "upper left")
legend.set_title('Mean DA',prop={'size': 8})


ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

#%% Supporting relative difference

fig, ax1 = plt.subplots(1,1,figsize = (2.5,2.5), dpi = 400)

ax1.set_title("Effect of K$_\mathrm{m}$ on DA levels", fontsize = 10)
ax1.set_ylabel("Relative difference (VS/DS)")
ax1.set_xlabel("DAT K$_\mathrm{m}$ (\u00B5M)")
ax1.set_ylim(0,6)
ax1.set_xlim(0*10**-6,6*10**-6)
ax1.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6])
ax1.set_xticklabels([0,2,4,6])

linestyles = [":","--","-"]
for i in range(3):
    ax1.plot(km_range, km_percentiles_VS[i,:]/km_percentiles_DS[i,:], ls = linestyles[i], color = "k")
    
legend = ax1.legend(("10$^{th}$","50$^{th}$","99.5$^{th}$"), frameon = False,
            handlelength = 1.3, prop={'size': 8}, loc = "upper right", bbox_to_anchor=[1, 1.05])
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Dopamine\npercentiles',prop={'size': 8})
    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Schematic figure for Km

def MM_uptake(c, vmax, km):
    return (vmax*c)/(km+c)

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.35,2.5), dpi = 400)

DA_range = np.linspace(0,2000,2001)
uptake_high = MM_uptake(DA_range, 3000, 840)
uptake_low = MM_uptake(DA_range, 3000, 210)

ax1.set_title("Uptake rate\n(K$_m$ dependence)", fontsize = 10)
ax1.set_ylabel("210 nM", fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.plot(DA_range, uptake_low, color = "forestgreen")
ax1.plot([0,2000], [3000,3000], color = "dimgrey", ls = "--",zorder = 0)
ax1.plot([210,210], [0,MM_uptake(210, 3000, 210)], color = "dimgrey",zorder = 0)
ax1.plot([0,210], [MM_uptake(210, 3000, 210),MM_uptake(210, 3000, 210)], color = "dimgrey",zorder = 0)
ax1.set_xlim(0,2000)
ax1.set_ylim(0,3500)

ax2.set_ylabel("840 nM", fontsize = 10)
ax2.text(100,2400, "V$_{max}$", fontsize = 8, color = "dimgrey")
ax2.text(500,200, "K$_{m}$", fontsize = 8, color = "dimgrey", rotation = 90)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.plot(DA_range, uptake_high, color = "forestgreen")
ax2.plot([0,2000], [3000,3000], color = "dimgrey", ls = "--",zorder = 0)
ax2.plot([840,840], [0,MM_uptake(840, 3000, 840)], color = "dimgrey",zorder = 0)
ax2.plot([0,840], [MM_uptake(840, 3000, 840),MM_uptake(840, 3000, 840)], color = "dimgrey",zorder = 0)
ax2.set_xlim(0,2000)
ax2.set_ylim(0,3500)

fig.tight_layout()

#%% Plot "spatial buffering" showing lower concentrations are affected more
max_DA = 4000

DA_range = np.linspace(1,max_DA, max_DA)*10**-9
norm_uptake = (DA_range*3*10**-6)/(210*10**-9 + DA_range)
kapp_low = (DA_range*3*10**-6)/(840*10**-9 + DA_range)
kapp_high = (DA_range*3*10**-6)/(4200*10**-9 + DA_range)

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (8,3.5), dpi = 400)

# fig.suptitle("Cocaine effect on uptake", fontsize = 10)

ax1.set_title("Competitive inhibition of DAT", fontsize = 10)
ax1.plot(DA_range*10**6,norm_uptake*10**6, color = "dimgrey")
ax1.plot(DA_range*10**6,kapp_low*10**6, color = "darkgreen", ls = "-")
ax1.plot(DA_range*10**6,kapp_high*10**6, color = "purple", ls = "-")
ax1.set_ylim(0,3)
ax1.set_xlim(-0.1,4)
ax1.set_ylabel("Uptake rate (\u00B5M s$^{-1}$)")
ax1.set_xlabel("[DA] \u00B5M")
leg = ax1.legend(("0.21","0.84","4.20"), frameon = False, title = "K$_{\mathrm{app}}$ (\u00B5M)", bbox_to_anchor = [1.06,0.38], loc = "upper right")
leg._legend_box.align = "left"

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.set_title("Relative uptake reduction", fontsize = 10)
ax2.plot(DA_range*10**6,100-kapp_low/norm_uptake*100, color = "darkgreen", ls = "-")
ax2.plot(DA_range*10**6,100-kapp_high/norm_uptake*100, color = "purple", ls = "-")

# leg = ax2.legend(("1","6"), frameon = False, title = "K$_{\mathrm{app}}$ (\u00B5M)")
# leg._legend_box.align = "left"

ax2.set_ylim(0,100)
ax2.set_xlim(-0.1,4)
ax2.set_ylabel("Percentage")
ax2.set_xlabel("[DA] \u00B5M")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3.set_title("Absolute uptake reduction", fontsize = 10)
ax3.plot(DA_range*10**6,(norm_uptake-kapp_low)*10**6, color = "darkgreen", ls = "-")
ax3.plot(DA_range*10**6,(norm_uptake-kapp_high)*10**6, color = "purple", ls = "-")

ax3.set_ylim(0,3)
ax2.set_xlim(-0.1,4)
ax3.set_ylabel("Uptake rate (\u00B5M s$^{-1}$)")
ax3.set_xlabel("[DA] \u00B5M")

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)


fig.tight_layout()

#%% Simulate burst

# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 10, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.001)


# Define the burst
start_time = 3 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 1 # Number of action potentials in a burst
burst_rate = 1000 # Burst firing rate (Hz)
burst_p_r = .35 # Release probability per AP during bursts

burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
sim_burst_DS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9

# Simulate the dynamics with Cocaine
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399, Km = 840*10**-9)
sim_burst_DS_Cocaine = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS_Cocaine = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9


# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 10, D = 763,
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
sim_burst_FSCV_VS = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9

# Simulate the dynamics with Cocaine
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399, Km = 840*10**-9)
sim_burst_VS_Cocaine = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS_Cocaine = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9


#%%

rounds = 10
FSCV_all_DS = np.zeros((rounds,sim_burst_FSCV_DS[::10].shape[0]))
FSCV_all_DS_Cocaine = np.zeros((rounds,sim_burst_FSCV_DS_Cocaine[::10].shape[0]))
FSCV_all_VS = np.zeros((rounds,sim_burst_FSCV_VS[::10].shape[0]))
FSCV_all_VS_Cocaine = np.zeros((rounds,sim_burst_FSCV_VS_Cocaine[::10].shape[0]))
for i in range(rounds):
    FSCV_all_DS[i,:] = sim_burst_FSCV_DS[i::10]
    FSCV_all_DS_Cocaine[i,:] = sim_burst_FSCV_DS_Cocaine[i::10]
    FSCV_all_VS[i,:] = sim_burst_FSCV_VS[i::10]
    FSCV_all_VS_Cocaine[i,:] = sim_burst_FSCV_VS_Cocaine[i::10]
    
FSCV_top_DS = FSCV_all_DS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_DS_Cocaine = FSCV_all_DS_Cocaine[int(np.argmax(FSCV_all_DS_Cocaine)/(FSCV_all_DS_Cocaine.shape[1])),:]
FSCV_top_VS = FSCV_all_VS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_VS_Cocaine = FSCV_all_VS_Cocaine[int(np.argmax(FSCV_all_VS_Cocaine)/(FSCV_all_VS_Cocaine.shape[1])),:]


b_t = int(start_time/var_list[4]) # Burst time
deconvolved_signal_DS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS-np.mean(sim_burst_FSCV_DS[b_t-1000:b_t-1]), mode = "full")
deconvolved_signal_DS_Cocaine = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS_Cocaine-np.mean(sim_burst_FSCV_DS_Cocaine[b_t-1000:b_t-1]), mode = "full")
deconvolved_signal_VS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS-np.mean(sim_burst_FSCV_VS[b_t-1000:b_t-1]), mode = "full")
deconvolved_signal_VS_Cocaine = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS_Cocaine-np.mean(sim_burst_FSCV_VS_Cocaine[b_t-1000:b_t-1]), mode = "full")


# Fit exponential decay
# ydata_DS = deconvolved_signal_DS[12:22]
# xdata_DS = np.linspace(0,len(ydata_DS)-1,len(ydata_DS))
# variables_FSCV_DS, _ = curve_fit(exp_decay, xdata = xdata_DS, ydata = ydata_DS)
# uptake_fit_DS = exp_decay(xdata_DS,variables_FSCV_DS[0],variables_FSCV_DS[1])

# ydata_VS = deconvolved_signal_VS[15:26]
# xdata_VS = np.linspace(0,len(ydata_VS)-1,len(ydata_VS))
# variables_FSCV_VS, _ = curve_fit(exp_decay, xdata = xdata_VS, ydata = ydata_VS)
# uptake_fit_VS = exp_decay(xdata_VS,variables_FSCV_VS[0],variables_FSCV_VS[1])




fig, axes = plt.subplots(1,4,figsize = (5.5,2.7), dpi = 400, gridspec_kw={"height_ratios": [1]})
# axes[0].set_title("Burst response", fontsize = 10)
fig.suptitle("Effect of increasing DAT K$_m$", fontsize = 10, y = 0.92)

axes[0].set_title("DS, simulated", fontsize = 10)
axes[0].plot(np.linspace(0,20,2000),sim_burst_DS-np.mean(sim_burst_FSCV_DS[b_t-1000:b_t-1]), color = "cornflowerblue", ls = "-", lw = 1.2)
axes[0].plot(np.linspace(0,20,2000),sim_burst_DS_Cocaine-np.mean(sim_burst_FSCV_DS_Cocaine[b_t-1000:b_t-1]), color = "cornflowerblue", ls = ":", lw = 1.2)

axes[1].set_title("DS, FSCV", fontsize = 10)
axes[1].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,deconvolved_signal_DS[:-19], color = "cornflowerblue", ls = "-", lw = 1.2)
axes[1].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,deconvolved_signal_DS_Cocaine[:-19], color = "cornflowerblue", ls = ":", lw = 1.2)
axes[1].legend(("Control", "Cocaine"), frameon = False, handlelength = 1.05, loc = "upper center", bbox_to_anchor = [0.5,1.02], fontsize = 8)

axes[2].set_title("VS, simulated", fontsize = 10)
axes[2].plot(np.linspace(0,20,2000),sim_burst_VS-np.mean(sim_burst_FSCV_VS[b_t-1000:b_t-1]), color = "indianred", ls = "-", lw = 1.2)
axes[2].plot(np.linspace(0,20,2000),sim_burst_VS_Cocaine-np.mean(sim_burst_FSCV_VS_Cocaine[b_t-1000:b_t-1]), color = "indianred", ls = ":", lw = 1.2)

axes[3].set_title("VS, FSCV", fontsize = 10)
axes[3].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,
         deconvolved_signal_VS[:-19], color = "indianred", ls = "-", lw = 1.2)
axes[3].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,
         deconvolved_signal_VS_Cocaine[:-19], color = "indianred", ls = ":", lw = 1.2)
axes[3].legend(("Control", "Cocaine"), frameon = False, handlelength = 1.05, loc = "upper center", bbox_to_anchor = [0.5,1.02], fontsize = 8)

axes[0].set_ylabel("\u0394[DA] (nM)")



for i in range(4):
    # axes[i].set_xlim(2.5,5)
    # axes[i].set_xticks([3,4,5])
    # axes[i].set_xticklabels([0,1,2])
    axes[i].set_xlabel("Seconds")
    axes[i].set_ylim(-20,5000)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    
    if i > 0:
        axes[i].spines["left"].set_visible(False)
        axes[i].set_yticks([])



fig.tight_layout()

#%% Simulate burst under nomifensine

# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 10, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


# Define the burst
start_time = 5 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 20 # Number of action potentials in a burst
burst_rate = 10 # Burst firing rate (Hz)
burst_p_r = .06 # Release probability per AP during bursts

burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# # Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399)
sim_burst_DS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9

# Simulate the dynamics with Nomifensine
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 4.5*10**-6, Ds = 321.7237308146399, Km = 1000*10**-9)
sim_burst_DS_Cocaine = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS_Cocaine = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9


# VS 
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 10, D = 763,
                  inter_var_distance = 25*(1/0.8), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399)
sim_burst_VS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9

# Simulate the dynamics with Cocaine
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 1.55*10**-6, Ds = 321.7237308146399, Km = 1000*10**-9)
sim_burst_VS_Cocaine = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS_Cocaine = np.mean(full_sim[:,:,:,:], axis = (1,2,3))*10**9

#%%

rounds = 10
FSCV_all_DS = np.zeros((rounds,sim_burst_FSCV_DS[::10].shape[0]))
FSCV_all_DS_Cocaine = np.zeros((rounds,sim_burst_FSCV_DS_Cocaine[::10].shape[0]))
FSCV_all_VS = np.zeros((rounds,sim_burst_FSCV_VS[::10].shape[0]))
FSCV_all_VS_Cocaine = np.zeros((rounds,sim_burst_FSCV_VS_Cocaine[::10].shape[0]))
for i in range(rounds):
    FSCV_all_DS[i,:] = sim_burst_FSCV_DS[i::10]
    FSCV_all_DS_Cocaine[i,:] = sim_burst_FSCV_DS_Cocaine[i::10]
    FSCV_all_VS[i,:] = sim_burst_FSCV_VS[i::10]
    FSCV_all_VS_Cocaine[i,:] = sim_burst_FSCV_VS_Cocaine[i::10]
    
FSCV_top_DS = FSCV_all_DS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_DS_Cocaine = FSCV_all_DS_Cocaine[int(np.argmax(FSCV_all_DS_Cocaine)/(FSCV_all_DS_Cocaine.shape[1])),:]
FSCV_top_VS = FSCV_all_VS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_VS_Cocaine = FSCV_all_VS_Cocaine[int(np.argmax(FSCV_all_VS_Cocaine)/(FSCV_all_VS_Cocaine.shape[1])),:]


b_t = int(start_time/var_list[4]) # Burst time
deconvolved_signal_DS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS-np.mean(sim_burst_FSCV_DS[b_t-100:b_t-1]), mode = "full")
deconvolved_signal_DS_Cocaine = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS_Cocaine-np.mean(sim_burst_FSCV_DS_Cocaine[b_t-100:b_t-1]), mode = "full")
deconvolved_signal_VS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS-np.mean(sim_burst_FSCV_VS[b_t-100:b_t-1]), mode = "full")
deconvolved_signal_VS_Cocaine = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS_Cocaine-np.mean(sim_burst_FSCV_VS_Cocaine[b_t-100:b_t-1]), mode = "full")


# Fit exponential decay
# ydata_DS = deconvolved_signal_DS[12:22]
# xdata_DS = np.linspace(0,len(ydata_DS)-1,len(ydata_DS))
# variables_FSCV_DS, _ = curve_fit(exp_decay, xdata = xdata_DS, ydata = ydata_DS)
# uptake_fit_DS = exp_decay(xdata_DS,variables_FSCV_DS[0],variables_FSCV_DS[1])

# ydata_VS = deconvolved_signal_VS[15:26]
# xdata_VS = np.linspace(0,len(ydata_VS)-1,len(ydata_VS))
# variables_FSCV_VS, _ = curve_fit(exp_decay, xdata = xdata_VS, ydata = ydata_VS)
# uptake_fit_VS = exp_decay(xdata_VS,variables_FSCV_VS[0],variables_FSCV_VS[1])




fig, axes = plt.subplots(1,3,figsize = (5.5,2.7), dpi = 400, gridspec_kw={"width_ratios": [1,1,0.5]})
fig.suptitle("Effect of increasing DAT K$_m$ in VS", fontsize = 10, y = 0.92)

axes[0].set_title("Control", fontsize = 10)
axes[0].plot(np.linspace(0,10,1000),sim_burst_VS-np.mean(sim_burst_FSCV_VS[b_t-100:b_t-1]), color = "dimgrey", ls = "--", lw = 1.2)
axes[0].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,
          deconvolved_signal_VS[:-19], color = "indianred", ls = "-", lw = 1.2)
axes[0].legend(("Sim.", "FSCV"), frameon = False, handlelength = 1.35, loc = "upper center", bbox_to_anchor = [0.75,1.02], fontsize = 8)


axes[1].set_title("Nomifensine", fontsize = 10)
axes[1].plot(np.linspace(0,10,1000),sim_burst_VS_Cocaine-np.mean(sim_burst_FSCV_VS_Cocaine[b_t-100:b_t-1]), color = "dimgrey", ls = "--", lw = 1.2)
axes[1].plot(np.linspace(0,FSCV_all_DS.shape[1]/10,FSCV_all_DS.shape[1])+0.1,
          deconvolved_signal_VS_Cocaine[:-19], color = "indianred", ls = "-", lw = 1.2)
axes[1].legend(("Sim.", "FSCV"), frameon = False, handlelength = 1.35, loc = "upper center", bbox_to_anchor = [0.75,1.02], fontsize = 8)


for i in range(2):
    
    axes[i].plot([10,12],[2700,2700], lw = 1.5, color = "k")
    axes[i].text(11, 2750, "30 Hz", rotation = 0, ha = "center", va = "bottom", fontsize = 8)
    axes[i].set_xlim(8,20)
    axes[i].set_xticks([10,15,20])
    axes[i].set_xticklabels([0,5,10])
    axes[i].set_xlabel("Seconds")
    axes[i].set_ylim(-20,3000)
    axes[i].set_ylabel("\u0394[DA] (nM)")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)


axes[2].bar([0,1], 
            [(np.max(sim_burst_FSCV_VS_Cocaine)/np.max(sim_burst_FSCV_VS))*100,
             (np.max(deconvolved_signal_VS_Cocaine)/np.max(deconvolved_signal_VS))*100], color = ["dimgrey", "indianred"])
axes[2].set_ylabel("Max \u0394[DA] after stim.\n(% of control)")
axes[2].set_ylim(0,800)
axes[2].set_xticks([0,1])
axes[2].set_xticklabels(["Sim.", "FSCV"], rotation = 90)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

fig.tight_layout()

#%%
# https://www.jneurosci.org/content/21/16/6338
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2787452/
# https://www.researchgate.net/publication/12023018_Carboni_E_Spielewoy_C_Vacca_C_Nosten-Bertrand_M_Giros_B_Di_Chiara_G_Cocaine_and_amphetamine_increase_extra-cellular_dopamine_in_the_nucleus_accumbens_of_mice_lacking_the_dopamine_transporter_gene_J_Ne/figures?lo=1

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (6,2.5), dpi = 400, gridspec_kw={"width_ratios":[1,1,3,1]})

ax1.set_ylabel("Baseline increase (%)")
ax1.set_ylim(0,600)
ax1.set_xticks([0,1])
ax1.set_xticklabels(["DS", "VS"])
ax1.bar([0,1], [(np.median(FSCV_top_DS_Cocaine)/np.median(FSCV_top_DS))*100,
                (np.median(FSCV_top_VS_Cocaine)/np.median(FSCV_top_VS))*100], color = ["cornflowerblue", "indianred"])

ax2.set_ylabel("Baseline increase (nM)")
ax2.set_ylim(0,120)
ax2.set_xticks([0,1])
ax2.set_xticklabels(["DS", "VS"])
ax2.bar([0,1], [np.median(FSCV_top_DS_Cocaine)-np.median(FSCV_top_DS),
                np.median(FSCV_top_VS_Cocaine)-np.median(FSCV_top_VS)], color = ["cornflowerblue", "indianred"])

ax3.set_title("Simulated FSCV", fontsize = 10)
ax3.set_xlabel("Seconds")
ax3.set_ylim(-50,400)
ax3.set_ylabel("\u0394[DA] (nM)")
ax3.plot(deconvolved_signal_DS_Cocaine-np.median(deconvolved_signal_DS_Cocaine), color = "cornflowerblue")
ax3.plot(deconvolved_signal_VS_Cocaine-np.median(deconvolved_signal_VS_Cocaine), color = "indianred")

ax4.set_ylabel("Peak \u0394[DA] (nM)")
ax4.set_ylim(0,400)
ax4.set_xticks([0,1])
ax4.set_xticklabels(["DS", "VS"])
ax4.bar([0,1], [(np.max(deconvolved_signal_DS_Cocaine)-np.median(deconvolved_signal_DS_Cocaine)),
                (np.max(deconvolved_signal_VS_Cocaine)-np.median(deconvolved_signal_VS_Cocaine))],
        color = ["cornflowerblue", "indianred"])

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

    



fig.tight_layout()

#%% Combine burst and conc in one figure

fig, ax1 = plt.subplots(figsize = (2.5,2.5), dpi = 400)


ax1.set_title("Simulated FSCV", fontsize = 10)
ax1.set_xlabel("Seconds")
ax1.set_ylim(-50,400)
ax1.set_xlim(40,140)
ax1.set_ylabel("\u0394[DA] (nM)")
ax1.plot(deconvolved_signal_DS_Cocaine-np.median(deconvolved_signal_DS_Cocaine), color = "cornflowerblue")
ax1.plot(deconvolved_signal_VS_Cocaine-np.median(deconvolved_signal_VS_Cocaine), color = "indianred")

# Inset
axins = ax1.inset_axes([0.75, 0.45, 0.25, 0.55])
axins.bar([0,1], [(np.max(deconvolved_signal_DS_Cocaine)-np.median(deconvolved_signal_DS_Cocaine)),
                (np.max(deconvolved_signal_VS_Cocaine)-np.median(deconvolved_signal_VS_Cocaine))],
        color = ["cornflowerblue", "indianred"])
axins.set_ylim(0,400)
# axins.set_yticklabels(axins.get_yticks(), fontProperties)

axins.spines["top"].set_visible(False)
axins.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)





