import numpy as np
import pandas as pd
import time
import copy
from numpy import sin, cos, max, pi, exp, sqrt, abs, nanmax, min, nanmin, inf, nan, trapz
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt
from tqdm import tqdm


# Record start time
start_time = time.time()


N = 20  
M = 20  
A = 1  

Pt = 1.0  
f = 26e9  
c = 3.00e8  
wavelength = c / f  
k = 2 * pi / wavelength

dx = wavelength / 2
dy = wavelength / 2

D = M * dx
L_1 = 2 * D ** 2 / wavelength

F_D = 0.47 # 1 feed
L = D * F_D

# The distance between the transmitter and the center of the RIS
d1 = L  
theta_t = pi
phi_t = 0 
# The distance between the receiver and the center of the RIS
d2 = 100  
theta_r = np.radians(0)
phi_r = np.radians(0)


def power_radiation_pattern_t(theta, phi):
    # return cos(theta) ** 62 * np.logical_and(0 <= theta, theta <= pi / 2) * \
    #                              np.logical_and(0 <= phi, phi < 2 * pi)
    return cos(theta) ** 4 \
                * np.logical_and(0 <= theta, theta <= pi / 2)  \
                *   np.logical_and(0 <= phi, phi < 2 * pi)

    
def power_radiation_pattern_cell(theta, phi):
    # return cos(theta) ** 3 * np.logical_and(0 <= theta, theta <= pi / 2) * \
    #                              np.logical_and(0 <= phi, phi < 2 * pi)
    return 4 * pi * dx * dy * cos(theta) / wavelength **2 \
                * np.logical_and(0 <= theta, theta <= pi / 2)  \
                * np.logical_and(0 <= phi, phi < 2 * pi)


# # Calculate gains of transmitter, receiver and unit cell
# gt = lambda theta, phi: power_radiation_pattern_t(theta, phi) * sin(theta)
# # gr = lambda theta, phi: power_radiation_pattern_r(theta, phi) * sin(theta)
# g = lambda theta, phi: power_radiation_pattern_cell(theta, phi) * sin(theta)

# integral, _ = dblquad(gt, 0, 2 * pi, 0, pi)
# Gt = 4 * pi / integral
# # Gt = 125.8925
# Gr = Gt

# # integral, _ = dblquad(gr, 0, 2 * pi, 0, pi)
# # Gr = 4 * pi / integral

# integral, _ = dblquad(g, 0, 2 * pi, 0, pi)
# G = 4 * pi / integral


# Transmitter and receiver coordinates
x_t, y_t, z_t = d1 * sin(theta_t) * cos(phi_t), d1 * sin(theta_t) * sin(phi_t), d1 * cos(theta_t)
x_r, y_r, z_r = d2 * sin(theta_r) * cos(phi_r), d2 * sin(theta_r) * sin(phi_r), d2 * cos(theta_r)


x_u = np.zeros((N, M))
y_u = np.zeros((N, M))
z_u = np.zeros((N, M))


best = np.array([[0, 0, 0, pi, 0, 0, 0, pi, pi, 0, pi, pi, pi, pi, 0, pi, 0, 0, 0, pi],
                 [0, pi, pi, 0, 0, 0, 0, pi, 0, 0, 0, pi, 0, pi, pi, pi, 0, 0, 0, 0],
                 [0, pi, pi, pi, pi, 0, 0, pi, pi, pi, pi, pi, 0, 0, pi, 0, pi, 0, pi, 0],
                 [0, pi, pi, 0, 0, 0, pi, pi, pi, 0, pi, pi, pi, pi, 0, 0, pi, pi, 0, 0],
                 [pi, pi, 0, 0, pi, pi, pi, 0, 0, pi, 0, 0, 0, pi, pi, 0, 0, pi, pi, 0],
                 [pi, 0, 0, 0, pi, pi, 0, 0, pi, pi, pi, pi, 0, 0, 0, pi, 0, pi, pi, pi],
                 [pi, pi, 0, pi, pi, 0, pi, pi, pi, pi, pi, pi, pi, pi, 0, pi, 0, pi, 0, pi],
                 [0, 0, 0, pi, pi, 0, pi, pi, 0, 0, 0, 0, pi, pi, 0, 0, pi, 0, 0, 0],
                 [0, 0, 0, pi, 0, pi, pi, 0, 0, 0, 0, 0, 0, pi, pi, 0, pi, 0, pi, 0],
                 [0, 0, 0, pi, 0, pi, 0, 0, 0, 0, 0, 0, 0, 0, pi, 0, 0, 0, 0, pi],
                 [pi, 0, pi, pi, 0, pi, pi, 0, 0, 0, 0, 0, 0, pi, pi, pi, pi, pi, 0, 0],
                 [0, 0, 0, pi, 0, pi, pi, 0, 0, 0, 0, 0, 0, pi, pi, 0, pi, pi, 0, pi],
                 [0, 0, 0, pi, 0, 0, pi, pi, 0, 0, 0, 0, pi, pi, pi, 0, 0, 0, pi, pi],
                 [pi, 0, 0, pi, pi, 0, pi, pi, pi, pi, pi, pi, pi, pi, 0, pi, pi, pi, 0, 0],
                 [0, 0, 0, 0, pi, pi, 0, 0, pi, pi, pi, pi, 0, 0, pi, pi, 0, 0, pi, pi],
                 [0, 0, pi, 0, 0, pi, pi, 0, 0, 0, 0, 0, 0, pi, pi, pi, 0, 0, 0, pi],
                 [0, 0, 0, pi, 0, 0, pi, pi, pi, pi, pi, pi, pi, pi, 0, 0, 0, 0, 0, pi],
                 [0, 0, pi, pi, pi, 0, 0, 0, pi, pi, pi, pi, pi, 0, 0, 0, pi, 0, 0, 0],
                 [pi, pi, pi, 0, 0, 0, pi, pi, 0, 0, 0, 0, 0, 0, 0, pi, pi, pi, 0, pi],
                 [pi, 0, 0, pi, pi, 0, pi, pi, 0, 0, pi, 0, pi, pi, 0, 0, pi, pi, pi, pi]])

best2 = np.array([[0, pi, 0, 0, 0, 0, pi, pi, pi, pi, pi, 0, pi, 0, 0, pi, pi, pi, pi, pi],
                  [pi, pi, pi, pi, pi, 0, pi, pi, 0, 0, pi, pi, 0, pi, pi, 0, pi, 0, pi, 0],
                  [pi, pi, 0, 0, 0, pi, pi, 0, 0, 0, pi, pi, pi, 0, 0, 0, 0, pi, 0, pi],
                  [pi, 0, 0, 0, pi, pi, pi, 0, 0, 0, 0, 0, pi, 0, pi, pi, pi, 0, 0, 0],
		          [pi, pi, pi, pi, 0, pi, 0, 0, pi, pi, pi, 0, pi, pi, 0, pi, 0, 0, pi, 0],
                  [pi, pi, 0, 0, 0, pi, pi, 0, 0, 0, pi, pi, 0, pi, 0, pi, 0, pi, 0, pi],
                  [0, pi, 0, pi, pi, pi, pi, pi, pi, 0, 0, pi, 0, 0, pi, 0, pi, 0, 0, pi],
		          [0, 0, pi, pi, 0, 0, pi, pi, pi, pi, 0, 0, pi, 0, pi, 0, pi, 0, pi, pi],
                  [0, pi, 0, 0, 0, 0, 0, 0, pi, pi, pi, 0, pi, pi, 0, pi, 0, 0, pi, 0],
                  [pi, pi, pi, 0, 0, 0, 0, 0, 0, pi, pi, 0, 0, pi, 0, pi, 0, 0, 0, 0],
		          [0, pi, 0, 0, 0, 0, 0, 0, 0, 0, pi, 0, 0, pi, 0, pi, 0, pi, pi, 0],
                  [0, pi, pi, pi, pi, 0, 0, 0, 0, 0, pi, pi, 0, pi, 0, pi, 0, pi, 0, 0],
                  [pi, 0, 0, 0, 0, pi, 0, 0, 0, 0, pi, 0, 0, pi, 0, pi, pi, pi, 0, 0],
		          [0, pi, pi, pi, 0, 0, 0, 0, 0, pi, pi, 0, 0, pi, 0, pi, 0, pi, pi, 0],
                  [pi, pi, pi, 0, pi, 0, 0, 0, 0, pi, pi, 0, pi, pi, 0, pi, 0, 0, pi, 0],
                  [pi, 0, 0, pi, 0, pi, 0, pi, pi, pi, 0, 0, pi, 0, 0, 0, 0, 0, pi, 0],
		          [0, pi, 0, pi, 0, pi, pi, pi, pi, pi, 0, 0, pi, 0, pi, pi, 0, 0, 0, pi],
                  [0, 0, 0, 0, pi, 0, 0, 0, pi, 0, 0, pi, pi, 0, 0, pi, 0, pi, 0, pi],
                  [pi, pi, 0, 0, 0, 0, pi, pi, pi, pi, 0, pi, pi, 0, pi, pi, pi, 0, 0, pi],
  		          [pi, pi, pi, pi, pi, pi, pi, pi, 0, 0, pi, 0, pi, 0, pi, 0, pi, 0, pi, 0]])

# phi_Gamma = np.flipud(best2) 


for n in range(0, N):
    for m in range(0, M):
        x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
        y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

# Compute the parameters along the coordinate axes
# The distance between the transmitter and the unit cell
r_t = np.sqrt((x_t - x_u) ** 2 + (y_t - y_u) ** 2 + z_t ** 2)  
# The distance between the receiver and the unit cell
r_r = np.sqrt((x_r - x_u) ** 2 + (y_r - y_u) ** 2 + z_r ** 2)

# The phase shift calculated from CUI
r_total = r_t + r_r
r_max = max(r_total) - r_total
phi_Gamma = k * r_max


# Make sure the phase shift between 0 and 2pi
phi_Gamma = np.mod(phi_Gamma, 2 * pi)

for n in range(0, N):
    for m in range(0, M):
        # 1-bit: 2 positions
        if phi_Gamma[n, m] < pi / 2 or phi_Gamma[n, m] > 3 * pi / 2:
            phi_Gamma[n, m] = pi
        else:
            phi_Gamma[n, m] = 0


Gamma = A * exp(1j * phi_Gamma)


# The elevation angle from the unit cell to the transmitter
theta_t_nm = np.arccos(abs(z_t / r_t))  
phi_t_nm = 0
# The elevation angle from the unit cell to the receiver
# theta_r_nm = np.arccos(abs(z_r / r_r))  
# phi_r_nm = 0
# The elevation angle from the transmitter to the unit cell
r_f = sqrt(x_t**2 + y_t**2 + z_t**2)
theta_tx_nm = np.arccos((r_f ** 2 + r_t ** 2 - (x_u ** 2 + y_u ** 2)) / (2 * r_f * r_t))  
# theta_tx_nm = np.arctan((np.sqrt(((z_t ** 2) * (y_u ** 2)) + ((z_t ** 2) * (x_u ** 2))) + \
#                                         (y_u * (x_t - x_u) - x_u * (y_t - y_u)) ** 2) / \
#                                         (r_t ** 2 - ((x_u * x_t) + (y_u * y_t))))
# phi_tx_nm = 0


# # Received Signal Power
# Pr = Pt * ((Gt * Gr * G * dx * dy * wavelength**2) / (64 * pi**3)) * (abs(total)**2)


# print(phi_Gamma)
plt.figure()
plt.imshow(phi_Gamma, origin='lower', cmap='viridis')    
plt.axis('equal') 

# Set integer tick labels for both x and y axes
plt.xticks(range(0, len(phi_Gamma[0]), 2))
plt.yticks(range(0, len(phi_Gamma), 2))
plt.colorbar()
plt.show()

# print(f"Gt:{10 * np.log10(Gt)} dB")
# print(f"Gr:{10 * np.log10(Gr)} dB")
# print(f"G:{10 * np.log10(G)} dB")


# print(f"Received Signal Power (Pr): {Pr} W")
# print(f"Received Signal Power (Pr): {10 * np.log10(Pr * 1e3)} dBm")


# Calculate radiation pattern
N_x = N*10
N_y = M*10

r_theta_vect = np.linspace(0, pi/2, N_x)
r_phi_vect = np.linspace(0, 2*pi, N_y)

x_r1 = np.zeros((len(r_theta_vect), len(r_phi_vect)))
y_r1 = np.zeros((len(r_theta_vect), len(r_phi_vect)))
z_r1 = np.zeros((len(r_theta_vect), len(r_phi_vect)))
theta_r_nm1 = np.zeros((N, M))
r_r1 = np.zeros((N, M))
u_ch = np.zeros((N, M))
E_r_2 = np.zeros((N, M), dtype = 'complex_')
E_r1 = np.zeros((len(r_theta_vect), len(r_phi_vect)), dtype = 'complex_')
rmn_rf = r_t
bla = np.array(power_radiation_pattern_t(theta_t_nm, 0) * \
                  power_radiation_pattern_cell(theta_t_nm, 0) * \
                #   power_radiation_pattern_cell(theta_r_nm, 0) * \
                  Gamma * exp(-1j * k * rmn_rf), dtype = 'complex_')

# Simulation loop
# for u in range(0, len(r_r_vect)):
for v in tqdm(range(0, len(r_theta_vect)), desc="Theta Loop"):
    for w in tqdm(range(0, len(r_phi_vect)), desc="Phi Loop", leave=False): 
        # x_r1[v, w], y_r1[v, w], z_r1[v, w] = d2 * np.sin(r_theta_vect[v]) * np.cos(r_phi_vect[w]), \
        #                                      d2 * np.sin(r_theta_vect[v]) * np.sin(r_phi_vect[w]), \
        #                                      d2 * np.cos(r_theta_vect[v])
        
        for m in range(0, M):
            for n in range(0, N):
                # r_r1[n, m] = np.sqrt((x_r1[v, w] - x_u[n, m]) ** 2 + (y_r1[v, w] - y_u[n, m]) ** 2 + z_r1[v, w] ** 2)
                # theta_r_nm1[n, m] = np.arccos(np.abs(z_r1[v, w] / r_r1[n, m])) 

                u_ch[n, m] = x_u[n, m] * sin(r_theta_vect[v]) * cos(r_phi_vect[w]) + \
                             y_u[n, m] * sin(r_theta_vect[v]) * sin(r_phi_vect[w])

                E_r_2[n, m]= bla[n, m] * exp(1j * k * u_ch[n, m]) * power_radiation_pattern_cell(r_theta_vect[v], 0)
                
        total = np.sum(E_r_2)

        # The radiation pattern calculated from GA
        E_r1[v, w] = total


# Normalize
E_o = copy.deepcopy(E_r1)
E_r1 = abs(E_r1)
E_r = E_r1 / nanmax(E_r1)
# E_r = E_r1
values = 20 * np.log10(E_r) 


# Prepare the data to calculate the directity
E_2 = copy.deepcopy(E_r)
E_2 = E_2**2
E_2 = E_2 * sin(r_theta_vect)
E_int = trapz(E_2, r_theta_vect)
E_int = trapz(E_int, r_phi_vect)
# print('int',E_int)
E_0 = E_r[np.where(values==0)]
print("Max value position: ", np.argwhere(values==0))
D_0 = abs(4*pi*E_0**2 / E_int)
print("Directity: ", 10*np.log10(D_0))


# Prepare data for visualization
# thetaaxisde = np.arange(theta_inf, theta_sup + pas_theta, pas_theta)
# phiaxisde = np.arange(phi_inf, phi_sup + pas_phi, pas_phi)
# thetaaxis = thetaaxisde * pi / 180
# phiaxis = phiaxisde * pi / 180

# Theta_angde_vec = thetaaxisde
# Phi_angde_vec = phiaxisde
# Theta_ang_vec = thetaaxis
# Phi_ang_vec = phiaxis

# vec_Theta = np.tile(Theta_angde_vec, len(Phi_ang_vec))
# vec_Phi = np.repeat(Phi_angde_vec, len(Theta_ang_vec))
# zzz = np.column_stack([10 * np.log10(abs(E_r).reshape(-1)), vec_Theta, vec_Phi])

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(zzz[:, 1], zzz[:, 2], zzz[:, 0], c=zzz[:, 0], cmap='viridis')
# ax.set_xlabel('Theta (degrees)')
# ax.set_ylabel('Phi (degrees)')
# ax.set_zlabel('Power (dB)')
# plt.show()

# plt.figure()
# plt.plot(zzz[:, 1], zzz[:, 0], marker='o', linestyle='-', color='b')
# plt.xlabel('Theta (degrees)')
# plt.ylabel('Power (dB)')
# plt.title('2D Radiation Pattern')
# plt.grid(True)
# plt.show()




# Create a rectangular heatmap
# pos = Phi_ang_vec
# ind = Theta_angde_vec

values1 = copy.deepcopy(values)
for v in range(0, len(r_theta_vect)):
    for w in range(0, len(r_phi_vect)): 
        if values1[v, w] < -100:
            values1[v, w] = None
        if values1[v, w] < -20.5:
            values1[v, w] = -20.5

## Calculate the interpolation function
# func = interp2d(pos, ind, values, kind='cubic')

# Plotting data points
# tnew = np.linspace(0, 2*pi, 200)  # theta
# rnew = np.linspace(0, 90, 100)  # r
# vnew = func(tnew, rnew)
# tnew, rnew = np.meshgrid(tnew, rnew)
THETA = r_phi_vect
# R = r_theta_vect
R = sin(r_theta_vect)


plt.figure(figsize=(30, 30))
ax = plt.subplot(2, 2, 1, projection='polar')
ax.set_rmax(nanmax(R))
plt.pcolor(THETA, R, values1, cmap='jet')
# plt.pcolor(tnew, rnew, vnew, cmap='jet')
plt.grid(c='black')
plt.colorbar()
plt.title('Normalized radiation patterns', fontsize=20)

# Set coordinate label annotations
phi_label_str = [r'$\varphi _{r}=0$     ', r'  $\varphi _{r}=\frac{\pi}{4}$', \
                 r'$\varphi _{r}=\frac{\pi}{2}$', r'$\varphi _{r}=\frac{3\pi}{4}$      ', \
                 r'$\varphi _{r}=\pi$     ', r'$\varphi _{r}=\frac{5\pi}{4}$       ', \
                 r'$\varphi _{r}=\frac{3\pi}{2}$', r'     $\varphi _{r}=\frac{7\pi}{4}$']
ax.set_xticklabels(phi_label_str)
# ax.set_yticks([pi/6, pi/3, nanmax(R)]) 
ax.set_yticks([sin(pi/6), sin(pi/3), nanmax(R)])  
theta_label_str = [r'$\theta _{r}=\frac{\pi}{6}$', r'$\theta _{r}=\frac{\pi}{3}$', ' ']
ax.set_yticklabels(theta_label_str)

# Set coordinate scale font size
plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)

# plt.savefig("pic.png", dpi=300)
plt.show()


# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s")

