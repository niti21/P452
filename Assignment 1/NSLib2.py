import numpy as np
import math
import NSLib2 as l
import matplotlib.pyplot as plt

from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def fixed_point(x_0, tol=1e-6, iteration_max=100):
    x_n1 = x_0
    i = 0
    
    for iteration in range(iteration_max):
        x_n2 = math.exp(-x_n1)
        
        if abs(x_n2 - x_n1) < tol:
            sol = x_n2
            i=i+1
        x_n1 = x_n2
    
    print( 'No. of iterations =', i)
    return sol




def simpsons(f, a, b, n):
    h = (b - a) / n
    sol = f(a) + f(b)

    for i in range(1, n):
        x_i = a + i * h
        factor = 4 if i % 2 == 1 else 2
        sol += factor * f(x_i)

    sol *= h / 3
    return sol

#Alternate SIMPSONS- same results

def Simpsons(f, l, u, n):
 
    # Calculating the value of h
    h = ( u - l )/n
 
    # List for storing value of x and f(x)
    x = list()
    fx = list()
    # Calculating values of x and f(x)
    i = 0
    while i<= n:
        x.append(l + i * h)
        fx.append(f(x[i]))
        i += 1
 
    # Calculating result
    res = 0
    i = 0
    while i<= n:
        if i == 0 or i == n:
            res+= fx[i]
        elif i % 2 != 0:
            res+= 4 * fx[i]
        else:
            res+= 2 * fx[i]
        i+= 1
    res = res * (h / 3)
    return res


def gaussian_quadrature(f, a, b, degree):
    nodes, weights =  np.polynomial.legendre.leggauss(degree)
    result = 0.5 * (b - a) * sum(w * f(0.5 * (b - a) * x + 0.5 * (a + b)) for x, w in zip(nodes, weights))
    return result



########################################################################################   
# Check later

def legendre_gauss_nodes_weights(degree, tolerance=1e-15, max_iterations=100):
    # Initial guesses for the roots of the Legendre polynomial
    nodes = np.cos(np.pi * (4 * np.arange(1, degree + 1) - 1) / (4 * degree + 2))

    # Iterate to refine the roots
    for _ in range(max_iterations):
        nodes_prev = np.copy(nodes)
        for i in range(degree):
            x_i = nodes[i]
            nodes[i] = x_i - (x_i * np.cos(degree * np.arccos(x_i)) - np.cos((degree - 1) * np.arccos(x_i))) / (
                degree * np.sin(degree * np.arccos(x_i)))

        # Check for convergence
        if np.allclose(nodes, nodes_prev, atol=tolerance):
            break

    # Calculate weights
    weights = 2 / ((1 - nodes**2) * (degree * np.sin(degree * np.arccos(nodes)))**2)

    return nodes, weights
def gaussian_quad_cust(f, a, b, degree):
    nodes, weights =  l.legendre_gauss_nodes_weights(degree)
    result = 0.5 * (b - a) * sum(w * f(0.5 * (b - a) * x + 0.5 * (a + b)) for x, w in zip(nodes, weights))
    return result


#########################################################################################


def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def solve_rk4(f, x0, y0, h, xf):
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < xf:
        y = rk4_step(f, x, y, h)
        x += h
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

#ALternate RK4
def RK( f, x0, y0, xf, st):
    
    x = [x0]
    y = [y0]

    n = int((xf-x0)/st)     # no. of steps
    for i in range(n):
        x.append(x[i] + st)
        k1 = st * dydx(x[i], y[i])
        k2 = st * dydx(x[i] + st/2, y[i] + k1/2)
        k3 = st * dydx(x[i] + st/2, y[i] + k2/2)
        k4 = st * dydx(x[i] + st, y[i] + k3)
      
        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
       
    return x, y




def crank_nicolson(initial_condition, alpha, x_max, t_max, num_x, num_t):
    delta_x = x_max / (num_x - 1)
    delta_t = t_max / (num_t - 1)
    
    x_values = np.linspace(0, x_max, num_x)
    t_values = np.linspace(0, t_max, num_t)
    
    u = np.zeros((num_x, num_t))
    
    # Set initial condition
    u[:, 0] = initial_condition(x_values)
    
    for n in range(0, num_t - 1):
        A = np.eye(num_x)
        b = np.zeros(num_x)
        
        for i in range(1, num_x - 1):
            A[i, i-1] = -alpha
            A[i, i] = 1 + 2 * alpha
            A[i, i+1] = -alpha
            
            b[i] = alpha * u[i-1, n] + (1 - 2 * alpha) * u[i, n] + alpha * u[i+1, n]
        
        # Boundary conditions
        b[0] = 0
        b[-1] = 0
        
        # Solve the system of equations using NumPy's solve
        u[:, n+1] = solve(A, b)
    
    return x_values, t_values, u




def initialize_grid(nx, ny):
    return np.zeros((nx, ny))

def poisson_equation(u, x, y, nx, ny, dx, dy, num_iterations):
    for _ in range(num_iterations):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - dx * dy * x[i, j] * np.exp(y[i, j]))

                
def set_boundary_conditions(u, x, y):
    # Set boundary conditions
    u[:, 0] = x[:, 0]  # u(x, 0) = x
    u[:, -1] = x[:, -1] * np.exp(1)  # u(x, 1) = xe
    u[0, :] = y[0, :]  # u(0, y) = y
    u[-1, :] = 2 * np.exp(y[-1, :])  # u(2, y) = 2e^y

def save_solution_table(u, x, y, nx, ny, filename):
    with open(filename, 'w') as file:
        file.write("x\t\ty\t\tu(x,y)\n")
        for i in range(nx):
            for j in range(ny):
                file.write(f"{x[i, j]:.2f}\t{y[i, j]:.2f}\t{u[i, j]:.6f}\n")

def plot_3d(x, y, u):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, cmap='magma', edgecolor='k', linewidth=0.5)
    
    # Add color gradient legend
    mappable = cm.ScalarMappable(cmap='magma')
    mappable.set_array(u)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label('u(x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    plt.title('3D Plot of the Solution to Poisson\'s Equation')
    plt.show()


#######################################################################################################################################################################################
