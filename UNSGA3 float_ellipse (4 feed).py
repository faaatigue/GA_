import os
import time
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool
from numpy import sin, cos, pi, exp, sqrt, abs, nanmax, nan, trapz
from pymoo.optimize import minimize
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.binx import BX, BinomialCrossover
from pymoo.operators.mutation.gauss import GM, GaussianMutation
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter





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

F_D = 0.26 # 4 feed
L = D * F_D
# d_D = 0.54
# d = D * d_D

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
SLL = -25


def power_radiation_pattern_t(theta, phi):
    return cos(theta) ** 4 \
                * np.logical_and(0 <= theta, theta <= pi / 2)  \
                *   np.logical_and(0 <= phi, phi < 2 * pi)

    
def power_radiation_pattern_cell(theta, phi):
    return 4 * pi * dx * dy * cos(theta) / wavelength **2 \
                * np.logical_and(0 <= theta, theta <= pi / 2)  \
                * np.logical_and(0 <= phi, phi < 2 * pi)


# Transmitter and receiver coordinates
x_t, y_t, z_t = d1 * sin(theta_t) * cos(phi_t), d1 * sin(theta_t) * sin(phi_t), d1 * cos(theta_t)

x_t1, y_t1, z_t1 =  (M/4)*dx,  (N/4)*dy, -d1
x_t2, y_t2, z_t2 =  (M/4)*dx, -(N/4)*dy, -d1
x_t3, y_t3, z_t3 = -(M/4)*dx,  (N/4)*dy, -d1
x_t4, y_t4, z_t4 = -(M/4)*dx, -(N/4)*dy, -d1

x_r, y_r, z_r = d2 * sin(theta_r) * cos(phi_r), d2 * sin(theta_r) * sin(phi_r), d2 * cos(theta_r)


x_u = np.zeros((N, M))
y_u = np.zeros((N, M))
z_u = np.zeros((N, M))


for n in range(0, N):
    for m in range(0, M):
        x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
        y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

# Compute the parameters along the coordinate axes
# The distance between the transmitter and the unit cell
# r_t = np.sqrt((x_t - x_u) ** 2 + (y_t - y_u) ** 2 + z_t ** 2)  
r_t1 = np.sqrt((x_t1 - x_u) ** 2 + (y_t1 - y_u) ** 2 + z_t ** 2)  
r_t2 = np.sqrt((x_t2 - x_u) ** 2 + (y_t2 - y_u) ** 2 + z_t ** 2)  
r_t3 = np.sqrt((x_t3 - x_u) ** 2 + (y_t3 - y_u) ** 2 + z_t ** 2)  
r_t4 = np.sqrt((x_t4 - x_u) ** 2 + (y_t4 - y_u) ** 2 + z_t ** 2)  
a_t1, a_t2, a_t3, a_t4 = 1/r_t1, 1/r_t2, 1/r_t3, 1/r_t4
# The distance between the receiver and the unit cell
r_r = np.sqrt((x_r - x_u) ** 2 + (y_r - y_u) ** 2 + z_r ** 2)


# The elevation angle from the unit cell to the transmitter
theta_t1_nm = np.arccos(abs(z_t / r_t1))  
theta_t2_nm = np.arccos(abs(z_t / r_t2))  
theta_t3_nm = np.arccos(abs(z_t / r_t3))  
theta_t4_nm = np.arccos(abs(z_t / r_t4))  
# The elevation angle from the transmitter to the unit cell
r_f1 = sqrt(x_t1**2 + y_t1**2 + z_t**2)
r_f2 = sqrt(x_t2**2 + y_t2**2 + z_t**2)
r_f3 = sqrt(x_t3**2 + y_t3**2 + z_t**2)
r_f4 = sqrt(x_t4**2 + y_t4**2 + z_t**2)
theta_tx1_nm = np.arccos((r_f1 ** 2 + r_t1 ** 2 - (x_u ** 2 + y_u ** 2)) / (2 * r_f1 * r_t1))  
theta_tx2_nm = np.arccos((r_f2 ** 2 + r_t2 ** 2 - (x_u ** 2 + y_u ** 2)) / (2 * r_f2 * r_t2)) 
theta_tx3_nm = np.arccos((r_f3 ** 2 + r_t3 ** 2 - (x_u ** 2 + y_u ** 2)) / (2 * r_f3 * r_t3)) 
theta_tx4_nm = np.arccos((r_f4 ** 2 + r_t4 ** 2 - (x_u ** 2 + y_u ** 2)) / (2 * r_f4 * r_t4)) 


# Calculate radiation pattern
N_x = N*10
N_y = M*10
p_inf, p_sup = -(N_x - 1) / 2, (N_x - 1) / 2
q_inf, q_sup = -(N_y - 1) / 2, (N_y - 1) / 2
# p_inf, p_sup = 1, N_x
# q_inf, q_sup = 1, N_y

r_p_vect = np.arange(p_inf, p_sup + 1, 1)
r_q_vect = np.arange(q_inf, q_sup + 1, 1)


rmn_rf = np.sqrt(x_u ** 2 + y_u ** 2 + d1 ** 2) 
# rmn_rf = sqrt(((m + 1) * dx) ** 2 + ((n + 1) * dy) ** 2 + d1 ** 2)    
            

u, v = 2 * pi * r_p_vect / (N_x * dx * k), \
       2 * pi * r_q_vect / (N_y * dy * k)

u_, v_ = np.meshgrid(u,v)

s = 1 - u_**2 - v_**2


# The parameters settings for objective function
u_0, v_0 = sin(theta_r) * cos(phi_r), sin(theta_r) * sin(phi_r)

# Lower mask (HPBW)
E_L1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_hp/2)**2
E_L1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_hp/2) * cos(theta_r))**2
E_L = E_L1_1 + E_L1_2

# Upper mask (SLL)
E_U1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_fn/2)**2
E_U1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_fn/2) * cos(theta_r))**2
E_U = E_U1_1 + E_U1_2


class BinaryMatrixProblem(ElementwiseProblem):

    def __init__(self, n_rows, n_cols, **kwargs):
        super(BinaryMatrixProblem, self).__init__(
            n_var=n_rows * n_cols,  # Number of variables
            n_obj=1,  # Number of objectives
            n_constr=0,  # Number of constraints
            xl=0,  # Variable lower bound (usually 0 for binary problems)
            xu=pi,  # Variable upper bound (usually 1 for binary problems)
            vtype=float,
            **kwargs
        )
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _evaluate(self, x, out, *args, **kwargs):
        # x = x.astype(float)
        for i, value in enumerate(x):
            if value >= pi/2:
                x[i] = 0
            else:
                x[i] = pi
        # print(x)

        # Reshape the 1D array to a 2D matrix
        matrix = np.reshape(x, (self.n_rows, self.n_cols))
        # print(matrix)

        Gamma = np.zeros((N, M), dtype = 'complex_')

        for n in range(0, N):
            for m in range(0, M):
                Gamma[n, m] = A * exp(1j * matrix[n, m])
        
        bla_1 = (power_radiation_pattern_t(theta_tx1_nm, 0) *      \
                power_radiation_pattern_cell(theta_t1_nm, 0))    \
                * Gamma * exp(-1j * k * r_t1) / r_t1
        bla_2 = (power_radiation_pattern_t(theta_tx2_nm, 0) *      \
                power_radiation_pattern_cell(theta_t2_nm, 0))    \
                * Gamma * exp(-1j * k * r_t2) / r_t2
        bla_3 = (power_radiation_pattern_t(theta_tx3_nm, 0) *      \
                power_radiation_pattern_cell(theta_t3_nm, 0))    \
                * Gamma * exp(-1j * k * r_t3) / r_t3
        bla_4 = (power_radiation_pattern_t(theta_tx4_nm, 0) *      \
                power_radiation_pattern_cell(theta_t4_nm, 0))    \
                * Gamma * exp(-1j * k * r_t4) / r_t4


        matrice_1 = np.fft.ifft2(bla_1, (N_x, N_y))
        matrice_1 = np.fft.fftshift(matrice_1)
        matrice_2 = np.fft.ifft2(bla_2, (N_x, N_y))
        matrice_2 = np.fft.fftshift(matrice_2)
        matrice_3 = np.fft.ifft2(bla_3, (N_x, N_y))
        matrice_3 = np.fft.fftshift(matrice_3)
        matrice_4 = np.fft.ifft2(bla_4, (N_x, N_y))
        matrice_4 = np.fft.fftshift(matrice_4)


        E_1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        E_2 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        E_3 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        E_4 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        
        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 

                if s[p, q] < 0:
                    E_1[p, q], E_2[p, q], E_3[p, q], E_4[p, q] = nan, nan, nan, nan
                    u_[p, q], v_[p, q] = nan, nan

                else: 
                    E_1[p, q] = sqrt(s[p, q]) * M * N * \
                                matrice_1[p, q]
                    E_2[p, q] = sqrt(s[p, q]) * M * N * \
                                matrice_2[p, q]                
                    E_3[p, q] = sqrt(s[p, q]) * M * N * \
                                matrice_3[p, q]
                    E_4[p, q] = sqrt(s[p, q]) * M * N * \
                                matrice_4[p, q]

        E_total = E_1 + E_2 + E_3 + E_4

        # Normalize
        E_total = abs(E_total)
        E = E_total / nanmax(E_total)
        values = 20 * np.log10(E) 
        # values = abs(E)

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if values[p, q] < -100:
                    values[p, q] = nan

        # Calculate objective function
        # First objective function and Lower mask (HPBW)
        F_L = []

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if E_L[p, q] <= 1 and values[p, q] < -3 and s[p, q] >= 0:
                    F_L.append((values[p, q] - (-3))**2)

        if len(F_L) == 0:
            F_L.append(0)

        F_L_sum = sum(F_L) / len(F_L)**2


        # Second objective function and Upper mask (SLL)
        F_U = []

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if E_U[p, q] > 1 and values[p, q] > SLL and s[p, q] >= 0:
                    F_U.append((values[p, q] - SLL)**2)                 

        if len(F_U) == 0:
            F_U.append(0)

        F_U_sum = sum(F_U) / len(F_U)**2

        F_obj = F_L_sum + F_U_sum


        # Calculate the fitness function 
        fitness = F_obj
        # print(fitness)
        out["F"] = fitness
        # out["F"] = [F_L_sum, F_U_sum]



# Parameters for optimization
n_rows = N
n_cols = M

# initialize the thread pool and create the runner
pool = ThreadPool()
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = BinaryMatrixProblem(n_rows, n_cols, elementwise_runner=runner)


sampling = LHS()
crossover = BinomialCrossover(prob=0.9)  
mutation = GaussianMutation(prob=0.1)
pop_size = 1000
ref_dirs = get_reference_directions("energy", 1, 1000)
termination = ("n_gen", 400)


# Configure the algorithm
algorithm = UNSGA3(
    pop_size=pop_size,  # Population size
    ref_dirs=ref_dirs,  # Number of reference directions
    # selecttion=selecttion,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
    elitism=6,  # Number of individuals to be considered as elites
)


# Perform optimization
res = minimize(
    problem,
    algorithm,
    termination=termination,
    seed=1,
    save_history=True,
    verbose=True,
    disp=2,
    # pop_initializer=pop_custom,
)


# print('Threads:', res.exec_time)

pool.close()


# Output the best fitness value for each generation
best_fitness_each_gen = [gen.opt.get("F") for gen in res.history]
# for gen_num, fitness in enumerate(best_fitness_each_gen):
#     print(f"Generation {gen_num + 1}: {fitness}")

# Plot the best fitness values
best_fitness_each_gen = np.array(best_fitness_each_gen)
best_fitness_each_gen = best_fitness_each_gen.reshape(-1)
plt.plot(range(1, len(best_fitness_each_gen) + 1), best_fitness_each_gen, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Evolution of Best Fitness Value over Generations')
plt.grid(True)
plt.savefig('out1/0-Evolution of Fitness.png', dpi=300)
plt.show()

# Output optimization results
best_fitness = res.F
# print("Best Fitness:", best_fitness)

best_solution = res.X.astype(float)
for i, value in enumerate(best_solution):
    if value >= pi/2:
        best_solution[i] = 0
    else:
        best_solution[i] = pi
best_matrix = best_solution.reshape((n_rows, n_cols))
# print("Best Solution (Flattened Binary Matrix):", best_matrix)



# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

# print(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s")



# Test optimization plot
phi_Gamma = best_matrix

# Plot phase grid
plt.figure()
plt.imshow(phi_Gamma, origin='lower', cmap='viridis')    
plt.axis('equal') 

# Set integer tick labels for both x and y axes
plt.xticks(range(0, M, 2))
plt.yticks(range(0, N, 2))
plt.colorbar()
plt.savefig('out1/1-Phase matrix.png', dpi=300)
plt.show()

# Recalculate radiation pattern
Gamma = np.zeros((N, M), dtype = 'complex_')

for n in range(0, N):
    for m in range(0, M):
        Gamma[n, m] = A * exp(1j * phi_Gamma[n, m])

bla_1 = (power_radiation_pattern_t(theta_tx1_nm, 0) *      \
           power_radiation_pattern_cell(theta_t1_nm, 0))    \
          * Gamma * exp(-1j * k * r_t1) / r_t1
bla_2 = (power_radiation_pattern_t(theta_tx2_nm, 0) *      \
           power_radiation_pattern_cell(theta_t2_nm, 0))    \
          * Gamma * exp(-1j * k * r_t2) / r_t2
bla_3 = (power_radiation_pattern_t(theta_tx3_nm, 0) *      \
           power_radiation_pattern_cell(theta_t3_nm, 0))    \
          * Gamma * exp(-1j * k * r_t3) / r_t3
bla_4 = (power_radiation_pattern_t(theta_tx4_nm, 0) *      \
           power_radiation_pattern_cell(theta_t4_nm, 0))    \
          * Gamma * exp(-1j * k * r_t4) / r_t4


N_x = N*100
N_y = M*100
p_inf, p_sup = -(N_x - 1) / 2, (N_x - 1) / 2
q_inf, q_sup = -(N_y - 1) / 2, (N_y - 1) / 2
# p_inf, p_sup = 1, N_x
# q_inf, q_sup = 1, N_y

r_p_vect = np.arange(p_inf, p_sup + 1, 1)
r_q_vect = np.arange(q_inf, q_sup + 1, 1)
            

u, v = 2 * pi * r_p_vect / (N_x * dx * k), \
       2 * pi * r_q_vect / (N_y * dy * k)

u_, v_ = np.meshgrid(u,v)

s = 1 - u_**2 - v_**2

matrice_1 = np.fft.ifft2(bla_1, (N_x, N_y))
matrice_1 = np.fft.fftshift(matrice_1)
matrice_2 = np.fft.ifft2(bla_2, (N_x, N_y))
matrice_2 = np.fft.fftshift(matrice_2)
matrice_3 = np.fft.ifft2(bla_3, (N_x, N_y))
matrice_3 = np.fft.fftshift(matrice_3)
matrice_4 = np.fft.ifft2(bla_4, (N_x, N_y))
matrice_4 = np.fft.fftshift(matrice_4)

E_1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
E_2 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
E_3 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
E_4 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 

        if s[p, q] < 0:
            E_1[p, q], E_2[p, q], E_3[p, q], E_4[p, q] = nan, nan, nan, nan
            u_[p, q], v_[p, q] = nan, nan

        else: 
            E_1[p, q] = sqrt(s[p, q]) * M * N * \
                        matrice_1[p, q]
            E_2[p, q] = sqrt(s[p, q]) * M * N * \
                        matrice_2[p, q]                
            E_3[p, q] = sqrt(s[p, q]) * M * N * \
                        matrice_3[p, q]
            E_4[p, q] = sqrt(s[p, q]) * M * N * \
                        matrice_4[p, q]

E_total = E_1 + E_2 + E_3 + E_4
            

# Normalize
E_total = abs(E_total)
E = E_total / nanmax(E_total)
values = 20 * np.log10(E) 
# values = abs(E)


# Make sure plot correctly
values1 = copy.deepcopy(values)

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        if values[p, q] < -100:
            values[p, q] = nan
            values1[p, q] = nan

        if values1[p, q] < -20.5:
            values1[p, q] = -20.5


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
plt.savefig('out1/2-Radiation patterns.png', dpi=300)
plt.show()


# Plot Area over -3dB
values2 = copy.deepcopy(values1)

values2[np.where(np.isclose(values2, 0, atol=1e-1))] = 0
values2[np.where((values2 >= -3) & (values2 < 0))] = -1.5

plt.imshow(values2, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Over -3dB Area')
plt.savefig('out1/3-3dB Area.png', dpi=300)
plt.show()


# Display the masks 
E_L1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_hp/2)**2
E_L1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_hp/2) * cos(theta_r))**2
E_L = E_L1_1 + E_L1_2

E_U1_1 = ((u_-u_0)*cos(phi_r + pi/2) + (v_-v_0)*sin(phi_r + pi/2))**2 / sin(Theta_fn/2)**2
E_U1_2 = ((u_-u_0)*sin(phi_r + pi/2) - (v_-v_0)*cos(phi_r + pi/2))**2 / (sin(Theta_fn/2) * cos(theta_r))**2
E_U = E_U1_1 + E_U1_2

values3 = copy.deepcopy(values1)
values3[np.where(np.isclose(values3, 0, atol=1e-2))] = -20
values3[np.where((values3 >= -3) & (values3 < 0))] = -1.5
values3[np.where(np.isclose(E_L, 1, atol=1e-2, rtol=1e-2))] = -12.5
values3[np.where(np.isclose(E_U, 1, atol=1e-2, rtol=1e-2))] = 0

plt.imshow(values3, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Display the masks on the radiation patterns')
plt.savefig('out1/4-Masks.png', dpi=300)
plt.show()


# SLL
Theta_s = Theta_fn / 2 + np.radians(4)
E_S = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_s)**2

values4 = copy.deepcopy(values1)

values4[np.where(E_S<=0)] = nan

plt.imshow(values4, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Check SLL Area')
plt.savefig('out1/5-Check SLL.png', dpi=300)
plt.show()


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
# print("Max value position: ", np.argwhere(values==0))
D_0 = abs(4*pi*E_0**2 / E_int)
# print("Directity: ", 10*np.log10(D_0))
# print("SLL: ", nanmax(values4))


# Create a DataFrame
df = pd.DataFrame(values, index=u, columns=v)

# File path
csv_file_path = 'out1/UNSGA3 float_ellipse (4 feed).csv'

# Write the DataFrame to a CSV file (including the index)
df.to_csv(csv_file_path, index=True, header=True)


# Output to txt
with open("out1/out.txt", "w") as f:
    for gen_num, fitness in enumerate(best_fitness_each_gen):
        f.write(f"Generation {gen_num + 1}: {fitness}\n")
    f.write(os.linesep * 2)
    f.write(f"theta_r: {theta_r}\n")
    f.write(f"phi_r: {phi_r}\n")
    f.write(os.linesep)
    f.write(f"Theta_hp: {Theta_hp}\n")
    f.write(f"Theta_fn: {Theta_fn}\n")
    f.write(f"SLL: {SLL}\n")
    f.write(os.linesep * 2)
    f.write(f"sampling: {sampling}\n")
    f.write(f"crossover: {crossover}\n")
    f.write(f"mutation: {mutation}\n")
    f.write(os.linesep)
    f.write(f"pop_size: {pop_size}\n")
    f.write(f"ref_dirs: {ref_dirs}\n")
    f.write(f"termination: {termination}\n")
    f.write(os.linesep * 2)
    f.write(f"Best Fitness: {best_fitness}\n")
    f.write(f"Best Solution (Flattened Binary Matrix): {best_matrix}\n")
    f.write(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s\n")
    f.write(os.linesep * 2)
    f.write("Max value position: {}\n".format(np.argwhere(values == 0)))
    f.write("Directity: {}\n".format(10 * np.log10(D_0)))
    f.write("SLL: {}\n".format(nanmax(values4)))

with open("out1/out.txt", "r") as f:
    data = f.read()

print(data)
