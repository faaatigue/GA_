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


# RIS parameters
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
L = D * F_D

# The distance between the transmitter and the center of the RIS
d1 = L  
theta_t = pi
phi_t = 0 
# The distance between the receiver and the center of the RIS
d2 = 100  
theta_r = np.radians(0)
phi_r = np.radians(0)

Theta_hp = np.radians(20)
Theta_fn = np.radians(25)
Theta_c = np.radians(5)
SLL = -20


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
x_r, y_r, z_r = d2 * sin(theta_r) * cos(phi_r), d2 * sin(theta_r) * sin(phi_r), d2 * cos(theta_r)


x_u = np.zeros((N, M))
y_u = np.zeros((N, M))
z_u = np.zeros((N, M))
r_t = np.zeros((N, M))
r_r = np.zeros((N, M))
theta_t_nm = np.zeros((N, M))
theta_tx_nm = np.zeros((N, M))


for n in range(0, N):
    for m in range(0, M):
        x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
        y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

        # Compute the parameters along the coordinate axes
        # The distance between the transmitter and the unit cell
        r_t[n, m] = np.sqrt((x_t - x_u[n, m]) ** 2 + (y_t - y_u[n, m]) ** 2 + z_t ** 2)  
        # The distance between the receiver and the unit cell
        r_r[n, m] = np.sqrt((x_r - x_u[n, m]) ** 2 + (y_r - y_u[n, m]) ** 2 + z_r ** 2)
        
        # The elevation angle from the unit cell to the transmitter
        theta_t_nm[n, m] = np.arccos(abs(z_t / r_t[n, m]))   
        theta_tx_nm[n, m] = np.arctan((np.sqrt(((z_t ** 2) * (y_u[n, m] ** 2)) + ((z_t ** 2) * (x_u[n, m] ** 2))) + \
                                                (y_u[n, m] * (x_t - x_u[n, m]) - x_u[n, m] * (y_t - y_u[n, m])) ** 2) / \
                                                (r_t[n, m] ** 2 - ((x_u[n, m] * x_t) + (y_u[n, m] * y_t))))


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
E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp / 2)**2

# Upper mask (SLL)
E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn / 2)**2

# Add one constraint
E_C = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_c / 2)**2


class BinaryMatrixProblem(ElementwiseProblem):

    def __init__(self, n_rows, n_cols, **kwargs):
        super(BinaryMatrixProblem, self).__init__(
            n_var=n_rows * n_cols,  # Number of variables
            n_obj=1,  # Number of objectives
            n_constr=1,  # Number of constraints
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
        
        bla = (power_radiation_pattern_t(theta_t_nm, 0) * \
                 power_radiation_pattern_cell(theta_t_nm, 0) / rmn_rf) * \
                 Gamma*  exp(-1j * k * rmn_rf) 

        matrice = np.fft.ifft2(bla, (N_x, N_y))
        matrice = np.fft.fftshift(matrice)

        E_1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')
        
        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 

                if s[p, q] < 0:
                    E_1[p, q] = nan
                    u_[p, q], v_[p, q] = nan, nan

                else: 
                    E_1[p, q] = sqrt(s[p, q]) * M * N * \
                                 matrice[p, q]
                    
        
        # Normalize
        E_1 = abs(E_1)
        E = E_1 / nanmax(E_1)
        # E = E_1
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
                if E_L[p, q] <= 0 and values[p, q] < -3 and s[p, q] >= 0:
                    F_L.append((values[p, q] - (-3))**2)

        if len(F_L) == 0:
            F_L.append(0)

        F_L_sum = sum(F_L) / len(F_L)**2


        # Second objective function and Upper mask (SLL)
        F_U = []

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if E_U[p, q] > 0 and values[p, q] > SLL and s[p, q] >= 0:
                    F_U.append((values[p, q] - SLL)**2)                      

        if len(F_U) == 0:
            F_U.append(0)

        F_U_sum = sum(F_U) / len(F_U)**2

        F_obj = F_L_sum + F_U_sum


        # Calculate the first constraint
        F_C = len(np.argwhere((E_C<=0) & (values==0)))

        # Calculate the first constraint
        # F_C2 = len(np.argwhere((E_U>0) & (values>=-3)))

        # Calculate the fitness function 
        fitness = F_obj
        # print(fitness)
        out["F"] = fitness
        # out["F"] = [F_L_sum, F_U_sum]
        out["G"] = 1 - F_C



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


# Configure the algorithm
algorithm = UNSGA3(
    pop_size=1000,  # Population size
    ref_dirs=get_reference_directions("energy", 1, 1000),  # Number of reference directions
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
    termination=("n_gen", 400),
    seed=1,
    save_history=True,
    verbose=True,
    disp=2,
    # pop_initializer=pop_custom,
)


print('Threads:', res.exec_time)

pool.close()


# Output the best fitness value for each generation
best_fitness_each_gen = [gen.opt.get("F") for gen in res.history]
for gen_num, fitness in enumerate(best_fitness_each_gen):
    print(f"Generation {gen_num + 1}: {fitness}")

# Plot the best fitness values
best_fitness_each_gen = np.array(best_fitness_each_gen)
best_fitness_each_gen = best_fitness_each_gen.reshape(-1)
plt.plot(range(1, len(best_fitness_each_gen) + 1), best_fitness_each_gen, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Evolution of Best Fitness Value over Generations')
plt.grid(True)
plt.show()

# Output optimization results
best_fitness = res.F
print("Best Fitness:", best_fitness)

best_solution = res.X.astype(float)
for i, value in enumerate(best_solution):
    if value >= pi/2:
        best_solution[i] = 0
    else:
        best_solution[i] = pi
best_matrix = best_solution.reshape((n_rows, n_cols))
print("Best Solution (Flattened Binary Matrix):", best_matrix)



# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Script execution time: {int(hours)}h {int(minutes)}min {int(seconds)}s")



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
plt.show()

# Recalculate radiation pattern
Gamma = np.zeros((N, M), dtype = 'complex_')

for n in range(0, N):
    for m in range(0, M):
        Gamma[n, m] = A * exp(1j * phi_Gamma[n, m])

bla = (power_radiation_pattern_t(theta_t_nm, 0) * \
            power_radiation_pattern_cell(theta_t_nm, 0) / rmn_rf) * \
            Gamma*  exp(-1j * k * rmn_rf) 

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

matrice = np.fft.ifft2(bla, (N_x, N_y))
matrice = np.fft.fftshift(matrice)

E_1 = np.zeros((len(r_p_vect), len(r_q_vect)), dtype = 'complex_')

for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 

        if s[p, q] < 0:
            E_1[p, q] = nan
            u_[p, q], v_[p, q] = nan, nan

        else: 
            E_1[p, q] = sqrt(s[p, q]) * M * N * \
                            matrice[p, q]
            

# Normalize
E_1 = abs(E_1)
E = E_1 / nanmax(E_1)
# E = E_1
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
plt.show()


# Display the masks 
E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp / 2)**2
E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn / 2)**2
E_C = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_c / 2)**2

values3 = copy.deepcopy(values1)
values3[np.where(np.isclose(values3, 0, atol=1e-2))] = -20
values3[np.where((values3 >= -3) & (values3 < 0))] = -1.5
values3[np.where(np.isclose(E_L, 0, atol=1e-4))] = -12.5
values3[np.where(np.isclose(E_U, 0, atol=1e-3))] = 0
values3[np.where(np.isclose(E_C, 0, atol=1e-4))] = -20

plt.imshow(values3, cmap='jet', origin='lower')
plt.axis('equal') 
plt.colorbar(label='')
plt.axis('off')
plt.title('Display the masks on the radiation patterns')
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
plt.show()


# # Plot 2d Radiation Pattern
# x_asix = np.linspace(-85, 85, N_x)
# plt.figure()
# plt.plot(x_asix, values[:, 100], marker='o', linestyle='-', color='b')

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
print("SLL: ", nanmax(values4))


# Create a DataFrame
df = pd.DataFrame(values, index=u, columns=v)

# File path
csv_file_path = 'UNSGA3 float_circle (1-Center constraint).csv'

# Write the DataFrame to a CSV file (including the index)
df.to_csv(csv_file_path, index=True, header=True)