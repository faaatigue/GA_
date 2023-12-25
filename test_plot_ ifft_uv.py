import numpy as np
import pandas as pd
import time
import copy
from numpy import sin, cos, max, pi, exp, sqrt, abs, nanmax, min, nanmin, inf, nan, trapz
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

Theta_hp = np.radians(10)
Theta_fn = np.radians(13)
Theta_c = np.radians(3)
SLL = -20


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

phi_Gamma = np.array([[0,         pi, pi, 0,         pi, pi,
  pi, pi, 0,         0,         pi, pi,
  0,         pi, pi, pi, pi, pi,
  pi, 0],
 [0,         pi, 0,         pi, pi, 0,
  0,         pi, 0,         0,         pi, 0,
  0,         pi, pi, pi, 0,         0,
  0,         pi],
 [pi, 0,         pi, pi, pi, 0,
  0,         pi, 0,         pi, 0,         pi,
  0,         pi, pi, 0,         pi, 0,
  pi, pi],
 [0,         pi, pi, 0,         pi, 0,
  0,         0,         0,         0,         0,         0,
  0,         pi, 0,         0,         0,         pi,
  pi, 0],
 [0,         pi, pi, pi, 0,         pi,
  0,         0,         pi, pi, 0,         pi,
  0,         0,         pi, 0,         0,         pi,
  pi, pi],
 [pi, 0,         0,         pi, 0,         0,
  pi, pi, pi, pi, pi, pi,
  pi, 0,         pi, 0,         pi, pi,
  0,         pi],
 [0,         pi, 0,         0,         0,         0,
  pi, pi, pi, 0,         pi, pi,
  pi, pi, 0,         0,         pi, 0,
  0,         pi],
 [pi, 0,         pi, 0,         0,         pi,
  pi, 0,         0,         0,         0,         0,
  pi, pi, pi, pi, 0,         pi,
  pi, pi],
 [0,         pi, 0,         0,         0,         pi,
  pi, 0,         0,         0,         0,         0,
  0,         pi, pi, 0,         0,         0,
  0,         0],
 [pi, 0,         0,         0,         pi, pi,
  0,         0,         0,         0,         0,         0,
  0,         pi, pi, pi, 0,         0,
  pi, pi],
 [pi, pi, pi, 0,         0,         pi,
  pi, 0,         0,         0,         0,         0,
  0,         pi, pi, pi, 0,         0,
  0,         0],
 [0,         pi, 0,         0,         pi, pi,
  pi, 0,         0,         0,         0,         0,
  0,         pi, pi, 0,         0,         0,
  0,         0],
 [pi, 0,         0,         0,         0,         pi,
  pi, pi, 0,         0,         0,         0,
  pi, pi, pi, 0,         0,         pi,
  pi, pi],
 [0,         pi, pi, 0,         0,         0,
  pi, pi, pi, pi, 0,         pi,
  pi, pi, pi, 0,         0,         0,
  0,         0],
 [0,         0,         0,         pi, 0,         0,
  0,         pi, pi, pi, pi, pi,
  pi, 0,         0,         pi, pi, 0,
  0,         0],
 [pi, pi, pi, pi, pi, 0,
  0,         0,         0,         pi, 0,         pi,
  0,         0,         0,         pi, pi, pi,
  pi, 0],
 [0,         pi, pi, 0,         pi, pi,
  pi, 0,         0,         0,         0,         0,
  0,         0,         pi, pi, 0,         0,
  0,         0],
 [pi, pi, 0,         0,         0,         pi,
  pi, pi, 0,         pi, 0,         pi,
  pi, pi, 0,         pi, pi, 0,
  pi, 0],
 [0,         pi, 0,         pi, 0,         pi,
  0,         pi, 0,         pi, 0,         pi,
  pi, pi, 0,         pi, pi, 0,
  pi, pi],
 [pi, pi, pi, pi, pi, 0,
  pi, pi, 0,         0,         pi, 0,
  pi, pi, 0,         0,         pi, pi,
  pi, pi]])

for n in range(0, N):
    for m in range(0, M):
        x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
        y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

# Compute the parameters along the coordinate axes
# The distance between the transmitter and the unit cell
r_t = np.sqrt((x_t - x_u) ** 2 + (y_t - y_u) ** 2 + z_t ** 2)  
# The distance between the receiver and the unit cell
r_r = np.sqrt((x_r - x_u) ** 2 + (y_r - y_u) ** 2 + z_r ** 2)

# # The phase shift calculated from CUI
# r_total = r_t + r_r
# r_max = max(r_total) - r_total
# phi_Gamma = k * r_max


# # Make sure the phase shift between 0 and 2pi
# phi_Gamma = np.mod(phi_Gamma, 2 * pi)

# for n in range(0, N):
#     for m in range(0, M):
#         # 1-bit: 2 positions
#         if phi_Gamma[n, m] < pi / 2 or phi_Gamma[n, m] > 3 * pi / 2:
#             phi_Gamma[n, m] = pi
#         else:
#             phi_Gamma[n, m] = 0


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
plt.xticks(range(0, M, 2))
plt.yticks(range(0, N, 2))
plt.colorbar()
plt.show()

# print(f"Gt:{10 * np.log10(Gt)} dB")
# print(f"Gr:{10 * np.log10(Gr)} dB")
# print(f"G:{10 * np.log10(G)} dB")


# print(f"Received Signal Power (Pr): {Pr} W")
# print(f"Received Signal Power (Pr): {10 * np.log10(Pr * 1e3)} dBm")



# Calculate radiation pattern
N_x = N*100
N_y = M*100
p_inf, p_sup = -(N_x - 1) / 2, (N_x - 1) / 2
q_inf, q_sup = -(N_y - 1) / 2, (N_y - 1) / 2
# p_inf, p_sup = 1, N_x
# q_inf, q_sup = 1, N_y

r_p_vect = np.arange(p_inf, p_sup + 1, 1)
r_q_vect = np.arange(q_inf, q_sup + 1, 1)


rmn_rf = r_t
# rmn_rf = sqrt(((m + 1) * dx) ** 2 + ((n + 1) * dy) ** 2 + d1 ** 2)    
      

# bla = sqrt(power_radiation_pattern_t(theta_t_nm, 0) *  \
#          power_radiation_pattern_cell(theta_t_nm, 0))    \
#          * Gamma * exp(-1j * k * rmn_rf) / rmn_rf

bla = (power_radiation_pattern_t(theta_tx_nm, 0) *      \
         power_radiation_pattern_cell(theta_tx_nm, 0))    \
         * Gamma * exp(-1j * k * rmn_rf) / rmn_rf


matrice = np.fft.ifft2(bla, (N_x, N_y))
matrice = np.fft.fftshift(matrice)
        

u, v = 2 * pi * r_p_vect / (N_x * dx * k), \
       2 * pi * r_q_vect / (N_y * dy * k)

u_, v_ = np.meshgrid(u,v)

s = 1 - u_**2 - v_**2

E_1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        
for p in tqdm(range(0, len(r_p_vect)), desc="E Loop"):
    for q in tqdm(range(0, len(r_q_vect)), desc=" ", leave=False): 

        if s[p, q] < 0:
            E_1[p, q] = nan
            u_[p, q], v_[p, q] = nan, nan

        else: 
            # E_1[p, q] = sqrt(sqrt(s[p, q])) * M * N * \
            #             matrice[p, q]
            E_1[p, q] = sqrt(s[p, q]) * M * N * \
                        matrice[p, q]


# Normalize
E_o = copy.deepcopy(E_1)
E_1 = abs(E_1)
E = E_1 / nanmax(E_1)
# E = E_1
values = 20 * np.log10(E) 
# values = abs(E)


oversmall = 0
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if values[p, q] < -100:
            values[p, q] = nan
            oversmall = oversmall + 1

print(oversmall)

# values[np.isneginf(values)] = nan


# Calculate objective function
u_0, v_0 = sin(theta_r) * cos(phi_r), sin(theta_r) * sin(phi_r)


# First objective function and Lower mask (HPBW)
# E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp)**2
E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp/2)**2
E_L1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_hp/2)**2
E_L1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_hp/2) * cos(theta_r))**2
E_L1 = E_L1_1 + E_L1_2

M_L = np.zeros((len(r_p_vect), len(r_q_vect)))
F_L = []

values_min = np.nanmin(values)
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_L[p, q] > 0:
            M_L[p, q] = values_min
        elif s[p, q] < 0:
            M_L[p, q] = nan
        else:
            M_L[p, q] = -3
            if values[p, q] < M_L[p, q]:
                F_L.append((values[p, q] - M_L[p, q])**2)


if len(F_L) == 0:
    F_L.append(0)

F_L_sum = sum(F_L) / (len(F_L)**2)


# Second objective function and Upper mask (SLL)
# E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn)**2
E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn/2)**2
E_U1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_fn/2)**2
E_U1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_fn/2) * cos(theta_r))**2
E_U1 = E_U1_1 + E_U1_2

M_U = np.zeros((len(r_p_vect), len(r_q_vect)))
F_U = []

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_U[p, q] > 0:
            M_U[p, q] = SLL
            if values[p, q] > M_U[p, q]:
                F_U.append((values[p, q] - M_U[p, q])**2)
        elif s[p, q] < 0:
            M_U[p, q] = nan        
        else:
            M_U[p, q] = 0


if len(F_U) == 0:
    F_U.append(0)

F_U_sum = sum(F_U) / len(F_U)**2

F_obj = F_L_sum + F_U_sum

print(F_obj)


# Make sure plot correctly
values1 = copy.deepcopy(values)

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if values[p, q] < -100:
            values[p, q] = None
            values1[p, q] = None

        if values1[p, q] < -20.5:
            values1[p, q] = -20.5


# Plot Lower mask and Upper mask together
E_C = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_c)**2
E_C1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_c)**2
E_C1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_c) * cos(theta_r))**2
E_C1 = E_C1_1 + E_C1_2

M_LU = copy.deepcopy(M_U)
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_L[p, q] <= 0:
            M_LU[p, q] = 20
        if E_C[p, q] <= 0:
            M_LU[p, q] = 30

plt.imshow(M_LU, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Lower mask and Upper mask using First way')
plt.show()

M_LU1 = np.zeros((len(r_p_vect), len(r_q_vect)))
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_U1[p, q] > 1:
            M_LU1[p, q] = SLL
        elif s[p, q] < 0:
            M_LU1[p, q] = nan        
        else:
            M_LU1[p, q] = 0

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_L1[p, q] <= 1:
            M_LU1[p, q] = 20
        if E_C1[p, q] <= 1:
            M_LU1[p, q] = 30

plt.imshow(M_LU1, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Lower mask and Upper mask using Second way')
plt.show()


# # Plot the difference of two masks
# M_L1 = copy.deepcopy(M_L)
# for p in range(0, len(r_p_vect)):
#     for q in range(0, len(r_q_vect)):
#         if E_L1[p, q] <= 1:
#             M_L1[p, q] = 20

# plt.imshow(M_L1, cmap='jet', origin='lower')
# plt.axis('equal') 
# plt.colorbar(label='')
# plt.axis('off')
# plt.title('The difference of two Lower masks')
# plt.show()

# M_U1 = copy.deepcopy(M_U)
# for p in range(0, len(r_p_vect)):
#     for q in range(0, len(r_q_vect)):
#         if E_U1[p, q] <= 1:
#             M_U1[p, q] = 20

# plt.imshow(M_U1, cmap='jet', origin='lower')
# plt.axis('equal') 
# plt.colorbar(label='')
# plt.axis('off')
# plt.title('The difference of two Upper masks')
# plt.show()


# Plotting data points
THETA = np.zeros((len(r_p_vect), len(r_q_vect)))
R = np.zeros((len(r_p_vect), len(r_q_vect)))

for p in tqdm(range(0, len(r_p_vect)), desc="Plot Loop"):
    for q in tqdm(range(0, len(r_q_vect)), desc=" ", leave=False): 
        THETA[p, q] = np.arctan2(v_[p, q], u_[p, q])
        R[p, q] = sqrt(u_[p, q]**2 + v_[p, q]**2)

# Calculate the theta and phase for each points
theta = np.arcsin(R)
phi = copy.deepcopy(THETA)
phi[:1000, :] = phi[:1000, :] + 2*pi

# plt.figure(figsize=(30, 30))
# ax = plt.subplot(2, 2, 1, projection='polar')
# ax.set_rmax(nanmax(R))
# plt.pcolor(THETA, R, values1, cmap='jet')
# plt.grid(c='black')
# plt.colorbar()
# plt.title('Normalized radiation patterns', fontsize=20)

# # Set coordinate label annotations
# phi_label_str = [r'$\varphi _{r}=0$     ', r'  $\varphi _{r}=\frac{\pi}{4}$', \
#                  r'$\varphi _{r}=\frac{\pi}{2}$', r'$\varphi _{r}=\frac{3\pi}{4}$      ', \
#                  r'$\varphi _{r}=\pi$     ', r'$\varphi _{r}=\frac{5\pi}{4}$       ', \
#                  r'$\varphi _{r}=\frac{3\pi}{2}$', r'     $\varphi _{r}=\frac{7\pi}{4}$']
# ax.set_xticklabels(phi_label_str)
# ax.set_yticks([sin(pi/6), sin(pi/3), nanmax(R)])  
# theta_label_str = [r'$\theta _{r}=\frac{\pi}{6}$', r'$\theta _{r}=\frac{\pi}{3}$     ', ' ']
# ax.set_yticklabels(theta_label_str)


# # Set coordinate scale font size
# plt.xticks(fontsize=15, rotation=90)
# plt.yticks(fontsize=15)

# # plt.savefig("pic.png", dpi=300)
# plt.show()


# Create a rectangular heatmap
# plt.imshow(abs(matrice), cmap='viridis', origin='upper')
plt.imshow(values1, cmap='jet', origin='lower')
plt.axis('equal') 

# Add labels and a color bar
plt.colorbar(label='')

plt.axis('off')

# Add a title
plt.title('Normalized radiation patterns')

# Display the plot
plt.show()


# Plot Area over -3dB
values2 = copy.deepcopy(values1)

values2[np.where(np.isclose(values2, 0, atol=1e-2))] = 0
values2[np.where((values2 >= -3) & (values2 < 0))] = -1.5

plt.imshow(values2, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Over -3dB Area')
plt.show()


# Display the masks 
E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp / 2)**2
E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn / 2)**2
E_C = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_c)**2

values3 = copy.deepcopy(values1)
values3[np.where(np.isclose(values3, 0, atol=1e-2))] = -20
values3[np.where((values3 >= -3) & (values3 < 0))] = -1.5
values3[np.where(np.isclose(E_L, 0, atol=1e-3))] = -12.5
values3[np.where(np.isclose(E_U, 0, atol=1e-3))] = 0
# values3[np.where(np.isclose(E_C, 0, atol=1e-4))] = -20

plt.imshow(values3, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Display the masks on the radiation patterns')
plt.show()


# Plot 2d Radiation Pattern
# x_asix = np.linspace(0, 58.5, 100)
# plt.figure()
# plt.plot(values[100,:], marker='o', linestyle='-', color='b')

# plt.title('2D Radiation Pattern')
# plt.grid(True)
# plt.show()


# Prepare the data to calculate the directity
E_2 = copy.deepcopy(E)
s_1 = copy.deepcopy(s).astype(complex)
E_2 = E_2**2
E_2 = E_2 / np.sqrt(s_1)
E_2[np.isnan(E_2)] = 0
E_int = trapz(E_2, u)
E_int = trapz(E_int, v)
# print('int',E_int)
E_0 = E[np.where(values==0)]
print("Max value position: ", np.argwhere(values==0))
D_0 = abs(4*pi*E_0**2 / E_int)
print("Directity: ", 10*np.log10(D_0))





# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s")


# Calculate HPBW manually
u0 = 0
v0 = 0
u_theta1 = u0
u_theta2 = u0
v_theta1 = v[1576]
v_theta2 = v[1705]
u_phi1 = u[909]
u_phi2 = u[1081]
v_phi1 = v0
v_phi2 = v0

# theta_0 = np.arcsin(sqrt(u0**2 + v0**2))
theta_0 = theta[1000, 1636]
phi_0 = phi[1000, 1636]
print("Theta (Maximum point): ", np.degrees(theta_0))
print("Phi (Maximum point): ", np.degrees(phi_0))

# Theta 
a1 = np.degrees(np.arccos((2-(((u_theta1 - u_theta2)**2 + (v_theta1 - v_theta2)**2))/cos(theta_0)**2) / 2))
a2 = np.degrees(abs(np.arcsin(sqrt(u_theta1**2 + v_theta1**2)) - np.arcsin(sqrt(u_theta2**2 + v_theta2**2))))
print("Theta (HPBW): ", a1, a2)

# Phi
b = np.degrees(np.arccos((2-((u_phi1 - u_phi2)**2 + (v_phi1 - v_phi2)**2)) / 2))
print("Phi (HPBW): ", b)


# Calculate HPBW manually
u_theta1 = u[913]
u_theta2 = u[753]
v_theta1 = v[1020]
v_theta2 = v[1056]
u_phi1 = u[801]
u_phi2 = u[834]
v_phi1 = v[965]
v_phi2 = v[1114]

# theta_0 = np.arcsin(sqrt(u0**2 + v0**2))
theta_0 = theta[818, 1041]
phi_0 = phi[818, 1041]
print("Theta (Maximum point): ", np.degrees(theta_0))
print("Phi (Maximum point): ", np.degrees(phi_0))

# Theta 
a1 = np.degrees(np.arccos((2-(((u_theta1 - u_theta2)**2 + (v_theta1 - v_theta2)**2))/cos(theta_0)**2) / 2))
a2 = np.degrees(abs(np.arcsin(sqrt(u_theta1**2 + v_theta1**2)) - np.arcsin(sqrt(u_theta2**2 + v_theta2**2))))
print("Theta (HPBW): ", a1, a2)

# Phi
b = np.degrees(np.arccos((2-((u_phi1 - u_phi2)**2 + (v_phi1 - v_phi2)**2)) / 2))
print("Phi (HPBW): ", b)

