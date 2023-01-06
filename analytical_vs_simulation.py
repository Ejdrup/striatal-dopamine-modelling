#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:50:51 2022

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def do_timestep_3D(u0, uptake_rate):
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
    # u = u - dt*(uptake_rate*u)/(Km + u)
    
    u0 = u.copy()
    return u0, u


def diffuse_3D_range(ms, space0, vmax):
    for i in tqdm(range(int(time/dt)-1)):
        
        _, space0[i+1,:,:,:] = do_timestep_3D(space0[i,:,:,:], 
                                      vmax)
    return space0

def point_source_3D(UCf,Ds,t,r,vmax,km):
    
    C = (UCf/(0.21*(4*Ds*t*np.pi)**(3/2)))*np.exp(-r**2/(4*Ds*t))*np.exp(-(vmax/km)*t)
    
    return C

#%% Size definitions 
time = 0.1 # in sec

# field size, um
w = h = depth = 20
# intervals in x-, y- directions, um
dx = dy = dz = 0.25
dx_dy = dx
# Diffusivity of DA in striatum, um2.s-1
D = 763
gamma = 1.54
Ds = D/(gamma**2)

nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz),

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

space0 = np.zeros((int(time/dt), nx, ny, nz))

# Q of 3000 to a concentration in dx**3 volume
Q = 9800
area = (dx*10**-6)**3 # in m^3
area_L = area*1000 # in liters
Na = 6.022*10**23 # Avrogadros number
start_conc = Q/Na/area_L*10**9 # start conc of single voxel in nM

# set middle voxel to start_conc and correct for ECF
space0[0, int(w/2/dx),int(h/2/dy), int(depth/2/dz)] = start_conc*(1/0.21) 

# Q of 3000 to a concentration in a single point
U = (4/3*np.pi*(25*10**-3)**3)*1000 # Volume in uL
Cf = 0.025375*10**6 # Fill concentration in uM at Q = 1000
Q_factor = 9.8 # Adjust to Q = 3000
UCf = U*Cf*Q_factor

#%% Run diffusion simulation and analytical result

sim_result = diffuse_3D_range(int(time*1000),space0, 0)

#%%
radius_range = np.linspace(dx,w/2,int(w/2/dx))
time_range = np.linspace(dt,time,int(time/dt))

analytical_result_t1 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.005, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_t2 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.01, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_t3 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.02, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r2 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 2,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r3 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 3,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r5 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 5,
                            vmax = 0, 
                            km = 0.210)

# #%% Plot across range

# plt.plot(radius_range,analytical_result)
# plt.plot(radius_range,sim_result[-1,int(w/2/dx),int(h/2/dy), int(depth/2/dz):]*(1/0.21))

#%% Plot across time
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (4,3), dpi = 400)

ax1.set_title("Analytical solution", fontsize = 10)
ax1.plot(time_range*1000,analytical_result_r2,
          color = "black", ls = "-")
ax1.plot(time_range*1000,analytical_result_r3,
         color = "black", ls = "--")
ax1.plot(time_range*1000,analytical_result_r5,
          color = "black", ls = ":")

ax1.set_ylabel("[DA] (nM)")
ax1.set_ylim(0,800)
ax1.set_yticks([0,200,400,600,800])
ax1.set_xlabel("ms")
ax1.set_xlim(0,50)
ax1.set_xticks([0,25,50])
ax1.legend(("2 \u00B5m", "3 \u00B5m", "5 \u00B5m"), title = "Radius     ", frameon = False,
           loc = "upper right", handlelength = 1.3)

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)


ax2.set_title("Diffusion simulation", fontsize = 10)
ax2.plot(time_range*1000,sim_result[:,int(w/2/dx),int(h/2/dy)+int(2/0.25), int(depth/2/dz)]*(1/0.21),
          color = "darkgreen", ls = "-")
ax2.plot(time_range*1000,sim_result[:,int(w/2/dx),int(h/2/dy)+int(3/0.25), int(depth/2/dz)]*(1/0.21),
         color = "darkgreen", ls = "--")
ax2.plot(time_range*1000,sim_result[:,int(w/2/dx),int(h/2/dy)+int(5/0.25), int(depth/2/dz)]*(1/0.21),
         color = "darkgreen", ls = ":")

ax2.set_ylim(0,800)
ax2.set_yticks([0,200,400,600,800])
ax2.set_xlabel("ms")
ax2.set_xlim(0,50)
ax2.set_xticks([0,25,50])
ax2.legend(("2 \u00B5m", "3 \u00B5m", "5 \u00B5m"), title = "Radius     ", frameon = False,
           loc = "upper right", handlelength = 1.3)

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

fig.tight_layout()

#%% Plot across radius

# fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.2,3), dpi = 400)
fig = plt.figure(figsize = (4.5,2.5), dpi = 400)
gs = GridSpec(2,3, height_ratios=[1,1], width_ratios=[1.5,1,1])

ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[:,2])

ax11.set_title("5 ms", fontsize = 10)
im = ax11.imshow(sim_result[int(0.005/dt),:,:,40], vmin = 0, vmax = 600, cmap = "magma")
ax11.set_xlim(0,80)
ax11.set_ylim(0,80)
ax11.set_xticks([])
ax11.set_yticks([])
# ax11.set_xlabel("10 \u00B5m")
ax11.set_ylabel("10 \u00B5m")
ax11.spines["left"].set_visible(False)
ax11.spines["bottom"].set_visible(False)
ax11.spines["right"].set_visible(False)
ax11.spines["top"].set_visible(False)

ax12.set_title("10 ms", fontsize = 10)
ax12.imshow(sim_result[int(0.01/dt),:,:,40], vmin = 0, vmax = 600, cmap = "magma")
ax12.set_xlim(0,80)
ax12.set_ylim(0,80)
ax12.set_xticks([])
ax12.set_yticks([])
ax12.set_xlabel("10 \u00B5m")
ax12.set_ylabel("10 \u00B5m")
ax12.spines["left"].set_visible(False)
ax12.spines["bottom"].set_visible(False)
ax12.spines["right"].set_visible(False)
ax12.spines["top"].set_visible(False)


# Analytical plot
ax2.set_title("Analytical", fontsize = 10)
ax2.plot(radius_range,analytical_result_t1,
          color = "black", ls = "-")
ax2.plot(radius_range,analytical_result_t2,
         color = "black", ls = "--")
ax2.plot(radius_range,analytical_result_t3,
          color = "black", ls = ":")

ax2.set_ylabel("[DA] (\u00B5M)", labelpad = 0)
ax2.set_ylim(0,1000)
ax2.set_yticks([0,200,400,600,800,1000])

ax2.set_xlabel("radius (\u00B5m)")
ax2.set_xlim(0,10)
legend = ax2.legend(("5 ms", "10 ms", "20 ms"), frameon = False,
            loc = "upper right", handlelength = 1.7, bbox_to_anchor = [1.12,1.01], fontsize = 8)
legend.set_title('Time  ',prop={'size':8})
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Simulation plot
ax3.set_title("Simulation", fontsize = 10)
ax3.plot(radius_range,sim_result[int(0.005/dt),40:,40,40],
          color = "darkgreen", ls = "-")
ax3.plot(radius_range,sim_result[int(0.01/dt),40:,40,40],
          color = "darkgreen", ls = "--")
ax3.plot(radius_range,sim_result[int(0.02/dt),40:,40,40],
          color = "darkgreen", ls = ":")

# ax3.set_ylabel("[DA] (\u00B5M)")
ax3.set_ylim(0,1000)
# ax3.set_yticks([0,200,400,600,800,1000])
ax3.set_yticks([])

legend = ax3.legend(("5 ms", "10 ms", "20 ms"), frameon = False,
            loc = "upper right", handlelength = 1.7, bbox_to_anchor = [1.12,1.01], fontsize = 8)
legend.set_title('Time  ',prop={'size':8})
ax3.set_xlabel("radius (\u00B5m)")
ax3.set_xlim(0,10)
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.spines["left"].set_visible(False)

fig.tight_layout()

cbar_ax = fig.add_axes([0.278, 0.265, 0.015, 0.555])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_yticks([])
# cbar_ax.set_xlim(0,1)
cbar_ax.set_title('[DA]', fontsize = 8)