# test final plot
phi_Gamma = best_matrix


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

F_D = 0.47
L = wavelength * 10 * F_D

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





# Transmitter and receiver coordinates
x_t, y_t, z_t = d1 * sin(theta_t) * cos(phi_t), d1 * sin(theta_t) * sin(phi_t), d1 * cos(theta_t)
x_r, y_r, z_r = d2 * sin(theta_r) * cos(phi_r), d2 * sin(theta_r) * sin(phi_r), d2 * cos(theta_r)


x_u = np.zeros((N, M))
y_u = np.zeros((N, M))
z_u = np.zeros((N, M))
r_t = np.zeros((N, M))
r_r = np.zeros((N, M))
r_total = np.zeros((N, M))
r_max = np.zeros((N, M))
theta_t_nm = np.zeros((N, M))
theta_r_nm = np.zeros((N, M))
theta_tx_nm = np.zeros((N, M))
theta_rx_nm = np.zeros((N, M))
F_combine = np.zeros((N, M))

Gamma = np.zeros((N, M), dtype = 'complex_')
bla = np.zeros((N, M), dtype = 'complex_')


for n in range(0, N):
    for m in range(0, M):
        x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
        y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

        # Compute the parameters along the coordinate axes
        # The distance between the transmitter and the unit cell
        r_t[n, m] = np.sqrt((x_t - x_u[n, m]) ** 2 + (y_t - y_u[n, m]) ** 2 + z_t ** 2)  
        # The distance between the receiver and the unit cell
        r_r[n, m] = np.sqrt((x_r - x_u[n, m]) ** 2 + (y_r - y_u[n, m]) ** 2 + z_r ** 2)

            
        Gamma[n, m] = A * exp(1j * phi_Gamma[n, m])
        
        # The elevation angle from the unit cell to the transmitter
        theta_t_nm[n, m] = np.arccos(abs(z_t / r_t[n, m]))  
        phi_t_nm = 0
        # The elevation angle from the unit cell to the receiver
        # theta_r_nm[n, m] = np.arccos(abs(z_r / r_r[n, m]))  
        # phi_r_nm = 0
        # The elevation angle from the transmitter to the unit cell
        # theta_tx_nm[n, m] = np.arccos((d1 ** 2 + r_t[n, m] ** 2 - (x_u[n, m] ** 2 + y_u[n, m] ** 2)) /  \
        #                               (2 * d1 * r_t[n, m]))  
        theta_tx_nm[n, m] = np.arctan((np.sqrt(((z_t ** 2) * (y_u[n, m] ** 2)) + ((z_t ** 2) * (x_u[n, m] ** 2))) + \
                                                (y_u[n, m] * (x_t - x_u[n, m]) - x_u[n, m] * (y_t - y_u[n, m])) ** 2) / \
                                                (r_t[n, m] ** 2 - ((x_u[n, m] * x_t) + (y_u[n, m] * y_t))))
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



N_x = N*10
N_y = M*10
p_inf, p_sup = -(N_x - 1) / 2, (N_x - 1) / 2
q_inf, q_sup = -(N_y - 1) / 2, (N_y - 1) / 2
# p_inf, p_sup = 1, N_x
# q_inf, q_sup = 1, N_y
pas_pq = 1

# Generate receiver coordinates
r_theta_vect = np.linspace(0, pi/2, N_x)
r_phi_vect = np.linspace(0, 2*pi, N_y)


r_p_vect = np.arange(p_inf, p_sup + pas_pq, pas_pq)
r_q_vect = np.arange(q_inf, q_sup + pas_pq, pas_pq)


# u = np.zeros((len(r_p_vect), len(r_q_vect)))
# v = np.zeros((len(r_p_vect), len(r_q_vect)))

rmn_rf = np.zeros((N, M))
s = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
P_r1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')


for n in range(0, N):
    for m in range(0, M):

        rmn_rf[n, m] = np.sqrt(x_u[n, m] ** 2 + y_u[n, m] ** 2 + d1 ** 2) 
        # rmn_rf[n, m] = sqrt(((m + 1) * dx) ** 2 + ((n + 1) * dy) ** 2 + d1 ** 2)       

bla_2 = (power_radiation_pattern_t(theta_tx_nm, 0) * \
         power_radiation_pattern_cell(theta_tx_nm, 0) / rmn_rf) * \
         Gamma*  exp(-1j * k * rmn_rf) 


matrice = np.fft.ifft2(bla_2, (N_x, N_y))
matrice = np.fft.fftshift(matrice)
        

u, v = 2 * pi * r_p_vect / (N_x * dx * k), \
       2 * pi * r_q_vect / (N_y * dy * k)

u_, v_ = np.meshgrid(u,v)

        
for p in tqdm(range(0, len(r_p_vect)), desc="E Loop"):
    for q in tqdm(range(0, len(r_q_vect)), desc=" ", leave=False): 
        
        s[p, q] = 1 - u_[p, q]**2 - v_[p, q]**2

        # The radiation pattern calculated from GA
        P_r1[p, q] = sqrt(s[p, q]) * M * N * \
                     matrice[p, q]
        

        # if s[p, q] < 0:
        #     P_r1[p, q] = None
        #     u_[p, q], v_[p, q] = None, None


# Normalize received power
P_r = P_r1 / nanmax(P_r1)
# P_r = P_r1
values = 10 * np.log10(abs(P_r))
# values = abs(P_r)

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if values[p, q] < -60:
            values[p, q] = None




u_0, v_0 = sin(theta_r) * cos(phi_r), sin(theta_r) * sin(phi_r)
Theta_hp = np.radians(20)
Theta_fn = 2*Theta_hp
SLL = -30

# First objective function and Lower mask
E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp)**2

M_L = np.zeros((len(r_p_vect), len(r_q_vect)))
F_L = []

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_L[p, q] > 1:
            M_L[p, q] = np.nanmin(values)
        else:
            M_L[p, q] = -3

        if values[p, q] < M_L[p, q]:
            F_L.append((values[p, q] - M_L[p, q])**2)



F_L_sum = sum(F_L) / (len(F_L)**2)

# Second objective function and Upper mask
E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn)**2

M_U = np.zeros((len(r_p_vect), len(r_q_vect)))
F_U = []

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if E_U[p, q] > 1:
            M_U[p, q] = SLL
        else:
            M_U[p, q] = 0

        if values[p, q] > M_U[p, q]:
            F_U.append((values[p, q] - M_U[p, q])**2)


F_U_sum = sum(F_U) / len(F_U)**2

F_obj = F_L_sum + F_U_sum

print(F_obj)



# Make sure plot correctly
# for p in range(0, len(r_p_vect)):
#     for q in range(0, len(r_q_vect)): 
#         if s[p, q] < 0:
#             values[p, q] = None
#             u_[p, q], v_[p, q] = None, None

        # if values[p, q] < -100:
        #     values[p, q] = None


# # Plotting data points
# THETA = np.zeros((len(r_p_vect), len(r_q_vect)))
# R = np.zeros((len(r_p_vect), len(r_q_vect)))

# for p in tqdm(range(0, len(r_p_vect)), desc="Plot Loop"):
#     for q in tqdm(range(0, len(r_q_vect)), desc=" ", leave=False): 
#         THETA[p, q] = np.arctan2(v_[p, q], u_[p, q])
#         R[p, q] = sqrt(u_[p, q]**2 + v_[p, q]**2)


# plt.figure(figsize=(30, 30))
# ax = plt.subplot(2, 2, 1, projection='polar')
# ax.set_rmax(nanmax(R))
# plt.pcolor(THETA, R, values, cmap='jet')
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
plt.imshow(values, cmap='jet', origin='lower')
plt.axis('equal') 

# Add labels and a color bar
plt.colorbar(label='')

plt.axis('off')

# Add a title
plt.title('')

# Display the plot
plt.show()




# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s")









