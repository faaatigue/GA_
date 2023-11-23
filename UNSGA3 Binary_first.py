import numpy as np
from pymoo.core.problem import ElementwiseProblem
from numpy import sin, cos, max, pi, exp, sqrt, abs, nanmax
from matplotlib import pyplot as plt
import time
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
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
theta_r_nm = np.zeros((N, M))
theta_tx_nm = np.zeros((N, M))
theta_rx_nm = np.zeros((N, M))
# phi_Gamma = np.zeros((N, M))
Gamma = np.zeros((N, M), dtype = 'complex_')
bla = np.zeros((N, M), dtype = 'complex_')



class BinaryMatrixProblem(ElementwiseProblem):

    def __init__(self, n_rows, n_cols):
        super(BinaryMatrixProblem, self).__init__(
            n_var=n_rows * n_cols,  # Number of variables
            n_obj=1,  # Number of objectives
            n_constr=0,  # Number of constraints
            xl=0,  # Variable lower bound (usually 0 for binary problems)
            xu=1,  # Variable upper bound (usually 1 for binary problems)
            vtype=int,
        )
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(float)
        for i, value in enumerate(x):
            if value == 1:
                x[i] = pi
        # print(x)

        # Reshape the 1D array to a 2D matrix
        matrix = np.reshape(x, (self.n_rows, self.n_cols))
        # print(matrix)

        for n in range(0, N):
            for m in range(0, M):
                x_u[n, m] = (m - ((M / 2) - 1 ) - 0.5) * dx
                y_u[n, m] = (n - ((N / 2) - 1 ) - 0.5) * dy

                # Compute the parameters along the coordinate axes
                # The distance between the transmitter and the unit cell
                r_t[n, m] = np.sqrt((x_t - x_u[n, m]) ** 2 + (y_t - y_u[n, m]) ** 2 + z_t ** 2)  
                # The distance between the receiver and the unit cell
                r_r[n, m] = np.sqrt((x_r - x_u[n, m]) ** 2 + (y_r - y_u[n, m]) ** 2 + z_r ** 2)

                    
                Gamma[n, m] = A * exp(1j * matrix[n, m])
                
                # The elevation angle from the unit cell to the transmitter
                theta_tx_nm[n, m] = np.arctan((np.sqrt(((z_t ** 2) * (y_u[n, m] ** 2)) + ((z_t ** 2) * (x_u[n, m] ** 2))) + \
                                                        (y_u[n, m] * (x_t - x_u[n, m]) - x_u[n, m] * (y_t - y_u[n, m])) ** 2) / \
                                                        (r_t[n, m] ** 2 - ((x_u[n, m] * x_t) + (y_u[n, m] * y_t))))


        N_x = N*10
        N_y = M*10
        p_inf, p_sup = -(N_x - 1) / 2, (N_x - 1) / 2
        q_inf, q_sup = -(N_y - 1) / 2, (N_y - 1) / 2
        # p_inf, p_sup = 1, N_x
        # q_inf, q_sup = 1, N_y
        pas_pq = 1


        r_p_vect = np.arange(p_inf, p_sup + pas_pq, pas_pq)
        r_q_vect = np.arange(q_inf, q_sup + pas_pq, pas_pq)


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

                
        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                
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
        # # values = abs(P_r)

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if values[p, q] < -100:
                    values[p, q] = None



        u_0, v_0 = sin(theta_r) * cos(phi_r), sin(theta_r) * sin(phi_r)
        Theta_hp = np.radians(20)
        Theta_fn = 2 * Theta_hp
        SLL = -30

        # First objective function and Lower mask
        E_L = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_hp)**2

        M_L = np.zeros((len(r_p_vect), len(r_q_vect)))
        F_1 = []

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if E_L[p, q] > 0:
                    M_L[p, q] = np.nanmin(values)
                else:
                    M_L[p, q] = -3

                if values[p, q] < M_L[p, q]:
                    F_1.append((values[p, q] - M_L[p, q])**2)

        F_1_sum = sum(F_1) / len(F_1)**2

        # Second objective function and Upper mask
        E_U = (u_ - u_0)**2 + (v_ - v_0)**2 - sin(Theta_fn)**2

        M_U = np.zeros((len(r_p_vect), len(r_q_vect)))
        F_2 = []

        for p in range(0, len(r_p_vect)):
            for q in range(0, len(r_q_vect)): 
                if E_U[p, q] > 0:
                    M_U[p, q] = SLL
                else:
                    M_U[p, q] = 0

                if values[p, q] > M_U[p, q]:
                    F_2.append((values[p, q] - M_U[p, q])**2)

        F_2_sum = sum(F_2) / len(F_2)**2

        F_obj = F_1_sum + F_2_sum

        # Calculate the fitness function 
        fitness = F_obj
        print(fitness)
        out["F"] = fitness



# Parameters for optimization
n_rows = N
n_cols = M
problem = BinaryMatrixProblem(n_rows, n_cols)


sampling = BinaryRandomSampling()
crossover=TwoPointCrossover(prob=0.9)
mutation=BitflipMutation(prob=0.1)

# Configure the algorithm
algorithm = UNSGA3(
    pop_size=36,  # Population size
    ref_dirs=get_reference_directions("energy", 1, 36),  # Number of reference directions
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
    termination=("n_gen", 1000),
    seed=1,
    save_history=True,
    verbose=True,
    disp=2,
)



# Output the best fitness value for each generation
best_fitness_each_gen = [gen.opt.get("F") for gen in res.history]
for gen_num, fitness in enumerate(best_fitness_each_gen):
    print(f"Generation {gen_num + 1}: {fitness}")



# Output optimization results
best_fitness = res.F
print("Best Fitness:", best_fitness)

best_solution = res.X.astype(float)
for i, value in enumerate(best_solution):
    if value == 1:
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



# test final plot
phi_Gamma = best_matrix


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
        theta_tx_nm[n, m] = np.arctan((np.sqrt(((z_t ** 2) * (y_u[n, m] ** 2)) + ((z_t ** 2) * (x_u[n, m] ** 2))) + \
                                                (y_u[n, m] * (x_t - x_u[n, m]) - x_u[n, m] * (y_t - y_u[n, m])) ** 2) / \
                                                (r_t[n, m] ** 2 - ((x_u[n, m] * x_t) + (y_u[n, m] * y_t))))



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

        
for p in range(0, len(r_p_vect)):
    for q in range(0, len(r_q_vect)): 
        
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
        if values[p, q] < -100:
            values[p, q] = None



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

