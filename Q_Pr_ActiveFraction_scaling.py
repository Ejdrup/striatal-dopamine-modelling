#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:50:02 2022

@author: ejdrup
"""
#%% Load packages and functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


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
    
    
    for i in range(int(t/dt)-1):
        
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


#%% Q akin to the Vmax range testing

Q_range = np.linspace(1000,30000,30)
Q_percentiles_DS = np.zeros((4,len(Q_range)))
Q_percentiles_VS = np.zeros((4,len(Q_range)))

simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, Q_val in tqdm(enumerate(Q_range)):
    # print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = Q_val, uptake_rate = 4.5*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Q_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])
    
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, Q_val in tqdm(enumerate(Q_range)):
    # print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = Q_val, uptake_rate = 1.55*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Q_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])
#%% Plot Q effect on percentiles
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (7.5,2.5), dpi = 400)
ax1.set_title("Effect of Q on DA levels", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("Q (DA molecules)")
ax1.set_ylim(0,1500)
ax1.set_xlim(0,30000)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
# ax1.plot([],[], color = "k", ls = "-.")
ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper left")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":","--","-"]
# DS and VS lines
for i in range(3):
    ax1.plot(Q_range,Q_percentiles_DS[i,:]*10**9, color = "cornflowerblue", ls = linestyles[i])
    if i == 10:
        ax1.plot(Q_range+80,Q_percentiles_VS[i,:]*10**9, color = "indianred", ls = linestyles[i])
    else:
        ax1.plot(Q_range,Q_percentiles_VS[i,:]*10**9, color = "indianred", ls = linestyles[i])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Q (DA molecules)")
ax2.set_title("VS and DS difference", fontsize = 10)
ax2.set_ylabel("Relative difference")
ax2.set_ylim(0,10)
ax2.set_xlim(0,30000)

# "Fake" legends
# ax2.plot([],[], color = "k", ls = "-")
# # ax2.plot([],[], color = "k", ls = "-.")
# ax2.plot([],[], color = "k", ls = "--")
# ax2.plot([],[], color = "k", ls = ":")
# legend = ax2.legend(("90$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
#             handlelength = 1.2, prop={'size': 8}, loc = "upper left")
#             # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
# legend.set_title('Percentiles',prop={'size': 8})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = ":")
# ax1.text(130*10**-9, 150, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "right", va = "center", color = "dimgrey")

# Conventional Km indicator
# ax2.vlines(210*10**-9,0,100, color = "dimgrey", lw = 0.8, ls = "-")
# ax2.text(350*10**-9, 100, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")


linestyles = [":","--","-"]
# DS and VS relative line
ax2.plot(Q_range[:],(Q_percentiles_VS[2,:]/Q_percentiles_DS[2,:]), color = "k", ls = linestyles[2])
ax2.plot(Q_range[:],(Q_percentiles_VS[1,:]/Q_percentiles_DS[1,:]), color = "k", ls = linestyles[1])
ax2.plot(Q_range[:],(Q_percentiles_VS[0,:]/Q_percentiles_DS[0,:]), color = "k", ls = linestyles[0])

legend = ax2.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper left", ncol = 1, columnspacing = 0.7, bbox_to_anchor = [0,1])
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax3.set_title("99.5$^{th}$/50$^{th}$ ratio", fontsize = 10)
ax3.set_xlabel("Q (DA molecules)")
ax3.set_ylabel("Ratio")
ax3.set_ylim(0,15)
ax3.set_xlim(0,30000)
ax3.plot(Q_range[:],Q_percentiles_DS[2,:]/Q_percentiles_DS[1,:], color = "cornflowerblue")
ax3.plot(Q_range[:],Q_percentiles_VS[2,:]/Q_percentiles_VS[1,:], color = "indianred")

legend = ax3.legend(("DS", "VS"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Region',prop={'size': 8})


ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.tight_layout()
#%% Schematic figure Q

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.35,2.5), dpi = 400)
x = np.array([0.00179974, 0.01487147, 0.20527461, 0.23358012, 0.24426879,
       0.31504937, 0.38786932, 0.41819855, 0.43096693, 0.47512727,
       0.50667423, 0.55777785, 0.60687839, 0.64363345, 0.6934015 ,
       0.72484311, 0.85548847, 0.92080667, 0.9577633 , 0.98243126])
y = np.array([0.54981051, 0.71762819, 0.94544669, 0.04534824, 0.63905389,
       0.43699442, 0.64297696, 0.85857944, 0.41079574, 0.99825848,
       0.01134036, 0.34727426, 0.21816365, 0.47661489, 0.97639573,
       0.1033522 , 0.6, 0.73075464, 0.11881775, 0.3168923])

ax1.set_title("Quantal size\n(DA molecules)", fontsize = 10)
ax1.text(0.5,0.91,"Vesicle", ha = "center", fontsize = 8, color = "dimgrey")
ax1.text(0.5,0.62,"DA", ha = "center", fontsize = 8, color = "forestgreen")
ax1.set_ylabel("3000", fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.plot(np.sin(np.linspace(-np.pi,np.pi,100))*0.35+0.5,
                       np.cos(np.linspace(-np.pi,np.pi,100))*0.35+0.5,
                       color = "dimgrey", lw = 1.5, ls = "-", zorder = 10)
ax1.scatter([0.6,0.5,0.35], [0.38,0.5,0.53], s = 10, color = "forestgreen")
ax1.set_xlim(-0.1,1.1)
ax1.set_ylim(-0.1,1.1)

ax2.set_ylabel("10000", fontsize = 10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.plot(np.sin(np.linspace(-np.pi,np.pi,100))*0.35+0.5,
                       np.cos(np.linspace(-np.pi,np.pi,100))*0.35+0.5,
                       color = "dimgrey", lw = 1.5, ls = "-", zorder = 10)
ax2.scatter([0.6,0.45,0.35,0.47,0.62, 0.4, 0.53, 0.73, 0.33, 0.5], 
            [0.55,0.3,0.6,0.47,0.67, 0.7, 0.77, 0.37, 0.33, 0.58]
            , s = 10, color = "forestgreen")
ax2.set_xlim(-0.1,1.1)
ax2.set_ylim(-0.1,1.1)

fig.tight_layout()
#%%
fig, (ax1) = plt.subplots(1,1,figsize = (3,2.7), dpi = 400)

ax1.plot([],[], color = "k", ls = "-.")
ax1.plot([],[], color = "k", ls = "--")
legend = ax1.legend(("90$^{th}$/50$^{th}$","50$^{th}$/10$^{th}$"), frameon = False,
            handlelength = 1.5, prop={'size': 9}, loc = "upper center")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Ratios',prop={'size': 9})


for i in range(2):
    ax1.plot(Q_range,(Q_percentiles_DS[i+1,:]/Q_percentiles_DS[i,:]), color = "cornflowerblue", ls = linestyles[i+1])
    ax1.plot(Q_range,(Q_percentiles_VS[i+1,:]/Q_percentiles_VS[i,:]), color = "indianred", ls = linestyles[i+1])

ax1.set_ylim(1,2.5)
ax1.set_xlim(0,10000)
ax1.set_xlabel("Q (DA molecules)")
ax1.set_title("Ratio between percentiles", fontsize = 10)
ax1.set_ylabel("Relative difference")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Release probability

Pr_range = np.linspace(0.02,0.6,59)
Pr_percentiles_DS = np.zeros((4,len(Pr_range)))
Pr_percentiles_VS = np.zeros((4,len(Pr_range)))

for i, Pr_val in enumerate(Pr_range):
    print(i)
    
    # DS
    simulation, space_init, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25, p_r = Pr_val, f_rate = 4, n_neurons = 150, Hz = 0.01)
            
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 4.5*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Pr_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])
    
    # VS
    simulation, space_init, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = Pr_val, f_rate = 4, n_neurons = 150, Hz = 0.01)
            
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 1.55*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Pr_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])



#%% Plot Pr effect on percentiles
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (7.5,2.5), dpi = 400)
ax1.set_title("Effect of R$_\%$ on DA levels", fontsize = 10)
ax1.set_ylabel("nM")
ax1.set_xlabel("Release probability (%)")
ax1.set_ylim(0,1500)
ax1.set_xlim(1,60)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.5, prop={'size': 8}, loc = "upper left")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":","--","-"]
# DS and VS lines
for i in range(3):
    ax1.plot(Pr_range*100,Pr_percentiles_DS[i,:]*10**9, color = "cornflowerblue", ls = linestyles[i])
    ax1.plot(Pr_range*100,Pr_percentiles_VS[i,:]*10**9, color = "indianred", ls = linestyles[i])
    # if i == 3:
    #     ax1.plot(Q_range-110,Q_percentiles_VS[i,:]*10**9, color = "darkred", ls = linestyles[i])
    # else:
    #     ax1.plot(Q_range,Q_percentiles_VS[i,:]*10**9, color = "darkred", ls = linestyles[i])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Release probability (%)")
ax2.set_title("VS and DS difference", fontsize = 10)
ax2.set_ylabel("Relative difference")
ax2.set_ylim(0,8)
ax2.set_xlim(1,60)

# "Fake" legends
ax2.plot([],[], color = "k", ls = "-")
ax2.plot([],[], color = "k", ls = "--")
ax2.plot([],[], color = "k", ls = ":")



linestyles = [":","--","-"]
# DS and VS relative line
for i in range(3):
    # ax2.plot(Q_range[:],(Q_percentiles_VS[i,:]-Q_percentiles_DS[i,:])*10**9, color = "k", ls = linestyles[i])
    ax2.plot(Pr_range[:]*100,(Pr_percentiles_VS[i,:]/Pr_percentiles_DS[i,:]), color = "k", ls = linestyles[i])

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3.set_title("99.5$^{th}$/50$^{th}$ ratio", fontsize = 10)
ax3.set_xlabel("Firing rate (Hz)")
ax3.set_ylabel("Ratio")
ax3.set_ylim(0,15)
ax3.set_xlim(1,60)
ax3.plot(Pr_range[:]*100,Pr_percentiles_DS[2,:]/Pr_percentiles_DS[1,:], color = "cornflowerblue")
ax3.plot(Pr_range[:]*100,Pr_percentiles_VS[2,:]/Pr_percentiles_VS[1,:], color = "indianred")

legend = ax3.legend(("DS", "VS"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Region',prop={'size': 8})


ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.tight_layout()


#%% Active release site fraction
# 0.11 terminals per um3 in DS
# https://www.sciencedirect.com/science/article/pii/0306452286902721?via%3Dihub
# (Quantification of the dopamine innervation in adult rat neostriatum)

Density_range = np.linspace(0.05,1,20)
Density_percentiles_DS = np.zeros((3,len(Density_range)))
Density_percentiles_VS = np.zeros((3,len(Density_range)))

simulation_DS, space_init_DS, firing_DS, release_sites_DS, var_list_DS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 9, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 9*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, D_val in enumerate(Density_range):
    print(i)
    # DS
    # Pick random subset of "D_val" size
    subset_DS = np.random.choice(np.linspace(0,firing_DS.shape[1]-1,firing_DS.shape[1]).astype(int),
                              size = int(firing_DS.shape[1]*D_val), replace = False)
    
    full_sim = sim_dynamics_3D(simulation_DS, space_init_DS, release_sites_DS[:,subset_DS], firing_DS[:,subset_DS], var_list_DS, 
                     Q = 3000, uptake_rate = 4.5*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Density_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    
    # VS
    # Pick random subset of "D_val" size
    subset_VS = np.random.choice(np.linspace(0,firing_VS.shape[1]-1,firing_VS.shape[1]).astype(int),
                              size = int(firing_VS.shape[1]*D_val), replace = False)
    
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS[:,subset_VS], firing_VS[:,subset_VS], var_list_VS, 
                     Q = 3000, uptake_rate = 1.55*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Density_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])

#%% Plot active release site effect on percentiles

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5,2.5), dpi = 400)
ax1.set_title("Active terminals", fontsize = 10)
ax1.set_ylabel("[DA] (M)")
ax1.set_xlabel("Percentage active")
ax1.set_yscale("log")
ax1.set_ylim(10**-10,10**-6)
ax1.set_xlim(0,100)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
# ax1.plot([],[], color = "k", ls = "-.")
ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            # handlelength = 1.5, prop={'size': 8}, loc = "lower right",)
            handlelength = 1.2, prop={'size': 8}, bbox_to_anchor=[1.05, 0.5], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})
fig.text(0.359+0.042,0.52,"DS", color = "cornflowerblue", fontsize = 8)
fig.text(0.435,0.52,"/", color = "k", fontsize = 8)
fig.text(0.445,0.52,"VS", color = "indianred", fontsize = 8)

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":","--","-"]
# DS and VS lines
for i in range(3):
    ax1.plot(Density_range*100,Density_percentiles_DS[i,:], color = "cornflowerblue", ls = linestyles[i])
    ax1.plot(Density_range*100,Density_percentiles_VS[i,:], color = "indianred", ls = linestyles[i])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Percentage active")
ax2.set_title("VS and DS difference", fontsize = 10)
ax2.set_ylabel("Relative difference")
ax2.set_ylim(0,8)
ax2.set_xlim(0,100)

# "Fake" legends
ax2.plot([],[], color = "k", ls = "-")
# ax2.plot([],[], color = "k", ls = "-.")
ax2.plot([],[], color = "k", ls = "--")
ax2.plot([],[], color = "k", ls = ":")
# legend = ax2.legend(("90$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
#                     handlelength = 1.2, prop={'size': 9})
#             # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
# legend.set_title('Dopamine\npercentiles',prop={'size': 9})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = ":")
# ax1.text(130*10**-9, 150, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "right", va = "center", color = "dimgrey")

# Conventional Km indicator
# ax2.vlines(210*10**-9,0,100, color = "dimgrey", lw = 0.8, ls = "-")
# ax2.text(350*10**-9, 100, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")


# DS and VS relative line
for i in range(3):
    # ax2.plot(Density_range[:],(Density_percentiles_VS[i,:]-Density_percentiles_DS[i,:])*10**9, color = "k", ls = linestyles[i])
    ax2.plot(Density_range*100,(Density_percentiles_VS[i,:]/Density_percentiles_DS[i,:]), color = "k", ls = linestyles[i])

legend = ax2.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.5, prop={'size': 8}, loc = "upper right")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

#%% Plot schemtic for active release sites

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.35,2.5), dpi = 400)
x = np.array([0.00179974, 0.01487147, 0.20527461, 0.23358012, 0.24426879,
       0.31504937, 0.38786932, 0.41819855, 0.43096693, 0.47512727,
       0.50667423, 0.55777785, 0.60687839, 0.64363345, 0.6934015 ,
       0.72484311, 0.85548847, 0.92080667, 0.9577633 , 0.98243126])
y = np.array([0.54981051, 0.71762819, 0.94544669, 0.04534824, 0.63905389,
       0.43699442, 0.64297696, 0.85857944, 0.41079574, 0.99825848,
       0.01134036, 0.34727426, 0.21816365, 0.47661489, 0.97639573,
       0.1033522 , 0.6, 0.73075464, 0.11881775, 0.3168923])

ax1.set_title("Active fraction\nof release sites", fontsize = 10)
ax1.set_ylabel("15 %", fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.scatter(x, y, s = 15, lw = 1, edgecolor = "k", color = "k")
ax1.scatter(x[[1,7,12]], y[[1,7,12]],
            s = 15, lw = 1, edgecolor = "forestgreen", color = "forestgreen")
ax1.set_xlim(-0.1,1.1)
ax1.set_ylim(-0.1,1.1)

ax2.set_ylabel("60 %", fontsize = 10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.scatter(x, y, s = 15, lw = 1, edgecolor = "k", color = "k")
ax2.scatter(x[[1,3,4,6,8,12,13,14,15,17,18,19]], y[[1,3,4,6,8,12,13,14,15,17,18,19]],
            s = 15, lw = 1, edgecolor = "forestgreen", color = "forestgreen")
ax2.set_xlim(-0.1,1.1)
ax2.set_ylim(-0.1,1.1)

fig.tight_layout()
#%% Firing rate

Fr_range = np.linspace(2,20,19)
Fr_percentiles_DS = np.zeros((4,len(Fr_range)))
Fr_percentiles_VS = np.zeros((4,len(Fr_range)))

for i, Fr_val in enumerate(Fr_range):
    print(i)
    
    # DS
    simulation, space_init, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = Fr_val, n_neurons = 150, Hz = 0.01)
            
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 4.5*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Fr_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])
    
    # VS
    simulation, space_init, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = Fr_val, n_neurons = 150, Hz = 0.01)
            
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 1.55*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Fr_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5,90])
    
#%% Plot f_rate effect on percentiles
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (7.5,2.5), dpi = 400)
ax1.set_title("Effect of firing rate on DA levels", fontsize = 10)
ax1.set_ylabel("nM")
ax1.set_xlabel("Firing rate (Hz)")
ax1.set_ylim(0,500)
ax1.set_xlim(1,20)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("99.5$^{th}$","50$^{th}$","10$^{th}$"), frameon = False,
            handlelength = 1.5, prop={'size': 8}, loc = "upper left")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Percentiles',prop={'size': 8})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":","--","-"]
# DS and VS lines
for i in range(3):
    ax1.plot(Fr_range,Fr_percentiles_DS[i,:]*10**9, color = "cornflowerblue", ls = linestyles[i])
    ax1.plot(Fr_range,Fr_percentiles_VS[i,:]*10**9, color = "indianred", ls = linestyles[i])
    # if i == 3:
    #     ax1.plot(Q_range-110,Q_percentiles_VS[i,:]*10**9, color = "darkred", ls = linestyles[i])
    # else:
    #     ax1.plot(Q_range,Q_percentiles_VS[i,:]*10**9, color = "darkred", ls = linestyles[i])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Firing rate (Hz)")
ax2.set_title("VS and DS difference", fontsize = 10)
ax2.set_ylabel("Relative difference")
ax2.set_ylim(0,8)
ax2.set_xlim(1,20)

# "Fake" legends
ax2.plot([],[], color = "k", ls = "-")
ax2.plot([],[], color = "k", ls = "--")
ax2.plot([],[], color = "k", ls = ":")

linestyles = [":","--","-"]
# DS and VS relative line
for i in range(3):
    # ax2.plot(Q_range[:],(Q_percentiles_VS[i,:]-Q_percentiles_DS[i,:])*10**9, color = "k", ls = linestyles[i])
    ax2.plot(Fr_range[:],(Fr_percentiles_VS[i,:]/Fr_percentiles_DS[i,:]), color = "k", ls = linestyles[i])

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax3.set_title("99.5$^{th}$/50$^{th}$ ratio", fontsize = 10)
ax3.set_xlabel("Firing rate (Hz)")
ax3.set_ylabel("Ratio")
ax3.set_ylim(0,15)
ax3.set_xlim(1,20)
ax3.plot(Fr_range[:],Fr_percentiles_DS[2,:]/Fr_percentiles_DS[1,:], color = "cornflowerblue")
ax3.plot(Fr_range[:],Fr_percentiles_VS[2,:]/Fr_percentiles_VS[1,:], color = "indianred")

legend = ax3.legend(("DS", "VS"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Region',prop={'size': 8})


ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.tight_layout()