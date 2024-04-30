import numpy as np
import math
import NSLib2 as l
import matplotlib.pyplot as plt
import copy
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import sparse
from scipy.fftpack import *





def make_matrix(N, M):
    I = [[0 for x in range(M)] for y in range(N)]
    return I

def matrix_read(B):
    #read the matrix text files
    a = open(B)
    A = []
    #A matrix
    for i in a:
        A.append([float(j) for j in i.split()])
    return (A)

def scaler_matrix_multiplication(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = c * A[i][j]
    return cA
    

def scaler_matrix_division(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = A[i][j]/c
    return cA


def matrix_multiplication(A, B):
    AB =  [[0.0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[i])):
            add = 0
            for k in range(len(A[i])):
                multiply = (A[i][k] * B[k][j])
                add = add + multiply
            AB[i][j] = add
    return (AB)





def matrix_addition(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] + B[i][j]
    return C

def matrix_substraction(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] - B[i][j]
    return C



def transpose(A):
    #if a 1D array, convert to a 2D array = matrix
    if not isinstance(A[0],list):
        A = [A]
 
    #Get dimensions
    r = len(A)
    c = len(A[0])

    #AT is zeros matrix with transposed dimensions
    AT = make_matrix(c, r)

    #Copy values from A to it's transpose AT
    for i in range(r):
        for j in range(c):
            AT[j][i] = A[i][j]

    return AT

def inner_product(A,B):

    AT = transpose(A)

    C = matrix_multiplication(AT, B)

    return C[0][0]


def power_normalize(A):
    max = -1000000
    #print("d", A)
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    #print("max", max)
    normA = scaler_matrix_division(max,A)
    return normA

def power_method(A, it):
	n = A.shape[0]
	
	# Step 1: Initialize a random vector
	v = np.random.rand(n)
	
	# Step 2: Power method iterations
	for _ in range(it):
		# Multiply v by the matrix
		Av = np.dot(A, v)
		
		# Normalize Av
		v = Av / np.linalg.norm(Av)
		
	# Step 3: Calculate the eigenvalue
	eigenvalue = np.dot(v, np.dot(A, v)) / np.dot(v, v)
	eigenvector = v
	
	return eigenvalue, eigenvector

def acceptance_rejection(target_pdf, initial_pdf=None, n_samples=100000, range_start=0, range_end=1):

    if initial_pdf is None:
        initial_pdf = lambda x: 1  # Uniform distribution by default

    # Find c by evaluating the ratio at the highest value within the range
    x_values = np.linspace(range_start, range_end, 1000)
    
    max_ratio = max(target_pdf(x) / initial_pdf(x) for x in x_values)
    print("c =", max_ratio)
    
    samples = []
    success_count = 0

    for _ in range(n_samples):
        # Sample from the initial distribution within the specified range
        x = np.random.uniform(range_start, range_end)

        # Sample from uniform distribution for acceptance
        u = np.random.rand()

        # Acceptance condition
        if u <= target_pdf(x) / (max_ratio * initial_pdf(x)):
            samples.append(x)
            success_count += 1

    success_probability = success_count / n_samples
    average_iterations = 1 / success_probability if success_probability != 0 else float('inf')

    return samples, success_probability, average_iterations


########################################################################################################################################
# To use this same one to find 1st and 2nd derivative
def derivative(z):
    n = len(z)
    for i in range(0,n):
        z[i] = z[i]*(n-i-1)
    z.pop()
    print ('derivative: ',cf)
    return z

# Function for finding derivative of a function at given x for h

def derivate(f,x):
    h=10**-8
    fd=(f(x+h)-f(x))/h # Derivative algorithm
    return fd

# Function for finding double derivative of a function at given x for h

def double_derivate(f,x):
    h=10**-8
    fdd=(f(x+h)+f(x-h)-2*f(x))/(2*h) # Double derivative algorithm
    return fdd

def matTol(x,y,tol):
    count=0
    if len(x)==len(y):
        for i in range(0,len(x)):
            if (abs(x[i]-y[i]))/abs(y[i]) < tol:
                count =count+1
            else:
                return False
    if count==len(x):
        return True

########################################################################################  
#########################################################################################

# Crout's method of LU decomposition
def crout(a):
    U = [[0 for i in range(len(a))] for j in range(len(a))]
    L = [[0 for i in range(len(a))] for j in range(len(a))]

    for i in range(len(a)):
        L[i][i] = 1

    for j in range(len(a)):
        for i in range(len(a)):
            total = 0
            for k in range(i):
                total += L[i][k] * U[k][j]

            if i == j:
                U[i][j] = a[i][j] - total

            elif i > j:
                L[i][j] = (a[i][j] - total)/U[j][j]

            else :
                U[i][j] = a[i][j] - total

    return U, L





# Forward-backward substitution function which returns the solution x = [x1, x2, x3, x4]
def forward_backward(U, L, b):
    y = [0 for i in range(len(b))]

    for i in range(len(b)):
        total = 0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = b[i] - total

    x = [0 for i in range(len(b))]

    for i in reversed(range(len(b))):
        total = 0
        for j in range(i+1, len(b)):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total)/U[i][i]

    return x


def lud(A):
    #Define matrices
    lt = [[0 for x in range(len(A))]
             for y in range(len(A))]
    ut = [[0 for x in range(len(A))]         
             for y in range(len(A))]
 
    for i in range(len(A)):                 
        #lower and upper matrix decomposition
        for k in range(i, len(A)):
            sum1 = 0
            for j in range(i):
                sum1 += (lt[i][j] * ut[j][k])       
            ut[i][k] = round((A[i][k] - sum1),4)  
            
        #Making diagonal terms 1 for solving equation.
        for k in range(i, len(A)):
            if (i == k):
                lt[i][i] = 1                                         
            else: 
                sum1 = 0                                                    
                for j in range(i):
                    sum1 += (lt[k][j] * ut[j][i])
                lt[k][i] = round(((A[k][i] - sum1)/ut[i][i]),4)
    print()
    print('Lower triangle:')
    print (lt)
    print()
    print('Upper triangle:')
    print (ut)          
    return lt,ut

# CHOLESKY 

def is_symmetric(A):
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] != A[j][i]:
                return False
    return True

def forward_sub(L, B):
    n = len(L)
    Y = [0] * n
    for i in range(n):
        temp = 0
        for j in range(i):
            temp += L[i][j] * Y[j]
        Y[i] = round((B[i] - temp) / L[i][i], 4)
    return Y

def backward_sub(U, Y):
    n = len(U)
    X = [0] * n
    for i in range(n - 1, -1, -1):
        temp = 0
        for j in range(i + 1, n):
            temp += U[i][j] * X[j]
        X[i] = round((Y[i] - temp) / U[i][i], 4)
    return X


############################################################           
    
def Transpose2(z):
    x = []
    for i in range(0,len(z)):
        y = []
        for j in range(0,len(z)):
            y.append(z[j][i])
        x.append(y)
    return x
 

def cholesky(z):
    if is_symmetric(z):                                       
        i=0  #checking symitric matrix
        while i <len(z):
            j=0
            sum1=0
            while j<i:
                sum1 = sum1 + z[j][i]*z[j][i]
                j = j + 1
            z[i][i]=(z[i][i]-sum1)**(0.5)                       
            j=i+1
            while j<len(z):
                k=0
                sum1=0
                while k<i:
                    sum1 = sum1 + z[i][k]*z[k][j]
                    k = k + 1
                z[j][i]=(z[j][i]-sum1)/z[i][i]                  
                z[i][j]=z[j][i]
                j=j+1
            i=i+1
        i=0
        while i <len(z):                                        
            j=i+1
            while j<len(z): #making all the elements above the diagonal 0
                z[i][j]=0   #To get a trangular matrix
                j=j+1
            i=i+1 
    return z

############################################################           


def pivot(A,B):
    n = len(A)
    count = 0   # keeps a track of number of exchanges (odd number of exchanges adds a phase of -1 to determinant)
    for i in range(n-1):
        if A[i][i] == 0:
            for j in range(i+1,n):
                if A[j][i] > A[i][i]:
                    A[i],A[j] = A[j],A[i]
                    count += 1
                    B[i],B[j] = B[j],B[i]
                    
    # return count if needed
    return A,B

    
#function for gauss-jordan elimination
def gauss_jordan(A,B):
    n = len(A)
    
    #partial pivoting
    A,B=pivot(A, B)             
    for i in range(n):
        p = A[i][i]
        #Making diagonal terms 1
        for j in range(i,n):
            A[i][j] = round((A[i][j]/p),4)
        B[i] = round((B[i]/p),4) 

        for k in range(n):              
            #Making Column zero except diagonal
            if (k == i) or (A[k][i] == 0):
                continue
            else:
                f = round(A[k][i],4)
                for l in range(i,n):
                    A[k][l] =round(( A[k][l] - f*A[i][l]),4)
                B[k] = round((B[k] - f*B[i]),4)
    print('Gauss jordan solution: ')
    return B

#For guess matrix
def gsm(a,b,x):
    for j in range(0, len(a)):               
        d = b[j]
        
        for i in range(0, len(a)):     
            if(j != i):
                d = d - a[j][i] * x[i]
        x[j] = round((d / a[j][j]),4)
    return x

#For gauss seidel (n = no. of iterations)
def GaussSeidel(a,x,b,n,tol): 
    y=[]
    for i in range(0,len(x)):
        y.append(x[i])
    
    for i in range(1,n):
        t=0
        for t in range(0,len(x)):
            y[t] = x[t]
    
        x = gsm(a,b,x)
    
        print(i,':',x)
    
        z = matTol(x,y,tol)
        if z == True:
            print()
            print('Gauss seidel result: ',x, '; no. of iterations: ',i)
            break
        else:
            continue

#Function for jacobi
def jacobi(a,b,it,tol):
    print(" Jacobi Calculations: \n")
    
    x=[1]*len(a)
    x1=[1]*len(a)
    
    
    for i in range(0,it):
        t=0
        for t in range(0,len(x)):
            x[t] = x1[t]
        
        x1 = gsm(a,b,x1)
    
        print(i+1,".",x1)
    
        z = matTol(x1,x,tol)
        if z == True:
            print('Jacobi result: \n',x1,"\n after",i+1,"iterations.")
            break
        else:
            continue


########################################################################    
    
def bracket(f, a, b):
    t = 0        # keeps track of number of iterations
    beta = 1.5   # step value
    if a > b:
        print("'a' should be less than 'b'!!!")
        exit()
    else:
        pr = f(a)*f(b)
        if pr < 0:
            return a, b
        else:
            while pr > 0:
                t += 1
                if abs(f(a)) < abs(f(b)):
                    a = a - beta*(b-a)
                    pr = f(a) * f(b)

                elif abs(f(a)) > abs(f(b)):
                    b = b + beta*(b-a) 
                    pr = f(a) * f(b)

            if t > 15:
                print("Try another range.")
                exit()

            return a, b
        

def bracketRF(f,a,b,tol):
    t = 0
    while (f(a)*f(b)>=0):
        if abs(f(a))<abs(f(b)):
           
            b = b + 0.1
        else:
            a = a - 0.1
        bisection(f,a,b,tol)
        t = t+1
    # print(t)
    return a,b,t

   

#Bisection Method
def bisection(f,a,b,tol):
    if (f(a)*f(b)>=0):
        print('wrong interval')
    c = a
    #print ("%10.2E" %a, "%10.2E" %f(a))
    #print ("%10.2E" %b, "%10.2E" %f(b))
    while ((b-a)>= tol):
        for i in range(15):
            c=(a+b)/2
            if (f(c)==0):
                break
            if (f(c)*f(a)<0):
                b=c
            else:
                a=c
    #print ("%10.2E" %c, "%10.2E" %(f(b)))
    return c

#Regula Falsi Method

def regulaFalsi(f, a, b, tol):
    a,b,s= bracketRF(f, a, b,tol)
    ct = 0
    root_i = []
    abs_error = []
    max_iter = 200
    c = a  # initial guess for the root
    for i in range(max_iter):
        c_prev = c
        c = b - ((b-a)*f(b))/(f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        elif f(a) * f(c) > 0:
            a = c
        #print(ct,' : ',c)
        ct+=1
        root_i.append(c)
        print('Root for Regula falsi',root_i[i])
        print(ct,' : ',c)
        error = abs(root_i[i] - root_i[i-1])
        abs_error.append(error)

        if abs(c - c_prev) < tol:
           #print(c)
           #print( 'No. of iterations:',ct)
            return c,ct     
    

    
#Newton raphson method
#f = function, df = derivative of f

def newtonRaphson(f,df,q,tol):
    t=0
    h = f(q) / df(q)
    while abs(h) >= tol:
        h = f(q)/df(q)
        q = q - h
        t=t+1
     
        print("Root for newton raphson: ", q)
        print("No. of iterations:", t)
    return q,t

#Alternate

def newt_raph(f,x,tol):
    max_it = 100
    xn = x
    k = 0
    x = x-f(x)/derivate(f,x)
    while abs(x-xn)>tol and k<max_it: # checking if the accuracy is achieved
        xn = x
        x = x-f(x)/derivate(f,x)
        k += 1
    return x


def Fixed_point(func, x_0, derivative, tol=1e-6, iteration_max=100):
    x_n1 = x_0
    i = 0
    sol = 1
    for iteration in range(iteration_max):
        if derivative is not None:
            x_n2 = x_n1 - func(x_n1) / derivative(x_n1)
        else:
            x_n2 = func(x_n1)
        
        if abs(x_n2 - x_n1) < tol:
            sol = sol*x_n2
            break  # Exit the loop once the solution is within toleranc
            print(sol)
        x_n1 = x_n2
        #i += 1
        return sol
def fixed_point(func, guess, tolerance=1e-5, max_iterations=1000):
	iterations = []
	values = []
	guesses = []
	for i in range(max_iterations):
		next_guess = func(guess)
		if abs(next_guess - guess) < tolerance:
			return next_guess, iterations, values, guesses
		guess = next_guess
		iterations.append(i)
		guesses.append(guess)
		values.append(func(guess))
	return None, iterations, values, guesses

def secant_method(f, x0, x1, tol):
    t = 0
    h0 = f(x0)
    h1 = f(x1)
    
    while abs(h1) >= tol:
        x_new = x1 - h1 * (x1 - x0) / (h1 - h0)
        x0, x1 = x1, x_new
        h0, h1 = h1, f(x_new)
        t += 1

    #print("Root: ", x_new)
    #print("No. of iterations:", t)
    return x_new



#Integration

#Midpoint integration
def midpoint_method(f,a,b,N):
    h = (abs(a - b))/N
    x = []
    y = []
    # print(h)
    for i in range(1,N+1):
        x.append((2*a + (2*i-1)*h)/2)
       
        y.append(f(x[i-1]))
    # print(y)
    # print(x)
    sum1 = 0
    for j in range(0,len(y)):
        sum1 = sum1 + y[j]*h
    res = round(sum1,9)
    return res
 
#trapeziodal integration
def trapezoidal(f,a,b,N):
    h = (abs(a - b))/N
    x = []
    y = []
    # print(h)
   
    for i in range(0,N+1):
        x.append((a + i*h))
        y.append(f(x[i]))
    sum1 = 0
    for i in range(1,len(x)):
        sum1 = sum1 + (h/2)*(y[i-1]+y[i])
    res = round(sum1,9)
    return res
    
#Simpsons


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
    res = round((res * (h / 3)),9)
    return res


#Finding N

def N_simp(l,u,t):
    N = ((1/180*t)*24)**(1/4)
    return N
def N_mid(l,u,t):
    N = ((1/24*t)*2)**(1/2)
    return N
def N_trap(l,u,t):
    N = ((1/12*t)*2)**(1/2)
    return N

# GAUSSIAN QUADRATURE

def gaussian_quadrature(f, a, b, degree):
    nodes, weights =  np.polynomial.legendre.leggauss(degree)
    result = 0.5 * (b - a) * sum(w * f(0.5 * (b - a) * x + 0.5 * (a + b)) for x, w in zip(nodes, weights))
    return result

#MONTE CARLO INTEGRATION
def monte_carlo(l,u,N,f):
    
    xran = []
    xi = []
    for i in range(N):
        xran.append(random.uniform(l, u))

    sm = 0
    for i in range(N):
        sm += f(xran[i])
        xi.append(i)
    total = (u-l)/float(N) * sm        
    return total


# ODE

def perdictor_corrector(f,x0,y0,N,x1):
    h=(x1-x0)/N
    x = x0
    y = y0
    X = [x0]
    Y = [y0]
    while x<x1:
        k1 = h*f(x)
        k2 = h*f(x+h)
        y = y+(k1+k2)/2
        x = x+h
        X.append(x)
        Y.append(y)
    return X,Y


# FORWARD EULER or EXPLICIT EULER

def explicit_euler(f, y0, x0, xf, st):
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    
    n = int((xf-x0)/st)
    for i in range(n):
        x.append(x[i] + st)
    for i in range(n):
        y.append(y[i] + st * f(y[i], x[i]))

    return x, y

#Implicit euler using Secant method

def implicit_euler_secant(dydx, y0, x0, xf, st):
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    n = int((xf - x0) / st)
    for i in range(n):
        x.append(x[i] + st)
        y_new = secant_method(lambda y_new: y_new - y[i] - st * dydx(y_new, x[i+1]), y[i], y[i]+st, 1e-6)
        y.append(y_new)

    return x, y

def semi_implicit_euler(dydx, y0, x0, xf, st, tol=1e-6):
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    n = int((xf - x0) / st)
    for i in range(n):
        x.append(x[i] + st)
        y_new = secant_method(lambda y_new: y_new - y[i] - st * dydx(x[i+1], y_new), y[i], y[i]+st, tol)
        y.append(y_new)

    return x, y


# RUNGE KUTTA

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

# ALTERNATE RK4

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

# Coupled ODE SHO
def RKsho(d2ydx2, dydx, x0, y0, z0, xf, st):
    
    x = [x0]
    y = [y0]
    z = [z0]      # dy/dx

    n = int((xf-x0)/st)     # no. of steps
    for i in range(n):
        x.append(x[i] + st)
        k1 = st * dydx(x[i], y[i], z[i])
        l1 = st * d2ydx2(x[i], y[i], z[i])
        k2 = st * dydx(x[i] + st/2, y[i] + k1/2, z[i] + l1/2)
        l2 = st * d2ydx2(x[i] + st/2, y[i] + k1/2, z[i] + l1/2)
        k3 = st * dydx(x[i] + st/2, y[i] + k2/2, z[i] + l2/2)
        l3 = st * d2ydx2(x[i] + st/2, y[i] + k2/2, z[i] + l2/2)
        k4 = st * dydx(x[i] + st, y[i] + k3, z[i] + l3)
        l4 = st * d2ydx2(x[i] + st, y[i] + k3, z[i] + l3)

        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        z.append(z[i] + (l1 + 2*l2 + 2*l3 + l4)/6)

    return x, y, z


# Shooting method


def LagInterpolation(zh, zl, yh, yl, y):
    z = zl + ((zh - zl) * (y - yl)/(yh - yl))
    return z



def shoot(d2ydx2, dydx, x0, y0, xf, yf, z_g1, z_g2, st, tol):
    #x0: Lower boundary value of x; xf: Upper boundary value of x
    #y0 = y(x0) ; yf = y(xf); z = dy/dx
    x, y, z = RKsho(d2ydx2, dydx, x0, y0, z_g1, xf, st)
    yn = y[-1]
    
    if abs(yn - yf) > tol:
        if yn < yf:
            zl = z_g1
            yl = yn

            x, y, z = RKsho(d2ydx2, dydx, x0, y0, z_g2, xf, st)
            yn = y[-1]

            if yn > yf:
                zh = z_g2
                yh = yn

                # calculate zeta using Lagrange interpolation
                z = LagInterpolation(zh, zl, yh, yl, yf)

                # using this zeta to solve using RK4
                x, y, z = RKsho(d2ydx2, dydx, x0, y0, z, xf, st)
                return x, y, z

            else:
                print("Bracketing failed!")


        elif yn > yf:
            zh = z_g1
            yh = yn

            x, y, z = RKsho(d2ydx2, dydx, x0, y0, z_g2, xf, st)
            yn = y[-1]

            if yn < yf:
                zl = z_g2
                yl = yn

                # calculate zeta using Lagrange interpolation
                z = LagInterpolation(zh, zl, yh, yl, yf)

                x, y, z = RKsho(d2ydx2, dydx, x0, y0, z, xf, st)
                return x, y, z

            else:
                print("Bracketing FAILED")


    else:
        return x, y, z 

# SYMPLECTIC OR VERLET

def verlet_method(f, y0, v0, t_span, h):
    num_steps = int((t_span[1] - t_span[0]) / h) + 1
    t_values = np.linspace(t_span[0], t_span[1], num_steps)
    
    y_values = np.zeros((num_steps, len(y0)))
    y_values[0] = y0
    
    # Initial step using Euler method
    y_temp = y0 + h * f(t_values[0], y0)
    y_values[1] = y0 + h * 0.5 * (f(t_values[0], y0) + f(t_values[1], y_temp))
    
    # Verlet method iterations
    for i in range(2, num_steps):
        y_values[i] = 2 * y_values[i-1] - y_values[i-2] + h**2 * f(t_values[i-1], y_values[i-1])
    
    return t_values, y_values

#VELOCITY VERLET

def velocity_verlet(f, x0, v0, t_span, h):
    
    num_steps = int((t_span[1] - t_span[0]) / h) + 1
    t_values = np.linspace(t_span[0], t_span[1], num_steps)
    x_values = np.zeros((num_steps, len(x0)))
    v_values = np.zeros((num_steps, len(v0)))

    x_values[0] = x0
    v_values[0] = v0

    for i in range(1, num_steps):
        # Verlet algorithm
        x_values[i] = x_values[i - 1] + h * v_values[i - 1] + 0.5 * h**2 * f(x_values[i - 1], t_values[i - 1])
        v_values[i] = v_values[i - 1] + 0.5 * h * (f(x_values[i], t_values[i - 1]) + f(x_values[i - 1], t_values[i - 1]))

    return t_values, x_values, v_values

# LEAPFROG

def leapfrog_int(f, q0, p0, dt, num_steps):
   
    q_values = np.zeros(num_steps + 1)
    p_values = np.zeros(num_steps + 1)
    t_values = np.zeros(num_steps + 1)

    q_values[0] = q0
    p_values[0] = p0
    t_values[0] = 0.0

    for i in range(num_steps):
        # Leapfrog integration steps
        p_half = p_values[i] - 0.5 * dt * f(q_values[i])
        q_values[i + 1] = q_values[i] + dt * p_half
        p_values[i + 1] = p_half - 0.5 * dt * f(q_values[i + 1])
        t_values[i + 1] = t_values[i] + dt

    return q_values, p_values, t_values

# BVP

def finite_difference_bvp(f, a, b, alpha, beta, N):

    h = (b - a) / N
    x_values = np.linspace(a, b, N + 1)

    # Build the coefficient matrix and the right-hand side vector
    A = np.zeros((N - 1, N - 1))
    rhs = np.zeros(N - 1)

    for i in range(1, N):
        x_i = a + i * h
        A[i - 1, i - 1] = 2 + h**2 * f(x_i)
        rhs[i - 1] = -h**2 * f(x_i) * x_i

        if i < N - 1:
            A[i - 1, i] = -1
            A[i, i - 1] = -1

    # Adjust the system for boundary conditions
    rhs[0] -= alpha
    rhs[-1] -= beta

    # Solve the linear system
    y_interior = solve(A, rhs)

    # Combine interior and boundary values
    y_values = np.concatenate(([alpha], y_interior, [beta]))

    return x_values, y_values


# PDE
    
def pde(vnot,l_x,l_t,N_x,N_t):
    
    h_x = (l_x/N_x)
    h_t = (l_t/N_t)
    X, V0 = vnot(l_x, N_x)
    alpha  = (h_t/(h_x**2))
    print('Alpha = ',alpha)
    print()
    V1 = np.zeros(N_x+1)
   
    for j in range(0,1000):
        for i in range(0,N_x+1):
            if i == 0:
                V1[i] = (1-2*alpha)*V0[i] + alpha*V0[i+1]
            elif i == N_x:
                V1[i] = alpha*V0[i-1] + (1-2*alpha)*V0[i]
            else:
                V1[i] = alpha*V0[i-1] + (1-2*alpha)*V0[i] + alpha*V0[i+1]
        V0 = list(V1)
        if j==1 or j==2 or j==5 or j==10 or j==20 or j==50 or j==100 or j==200 or j==500 or j==1000:
            plt.plot(X, V0)
    plt.show()



def crank_nicolson(M, alpha, T, L):
    # Discretization parameters
    N = M**2
    x0, xL = 0, L
    dx = (xL - x0) / (M - 1)
    dt = T / (N - 1)

    # Coefficients for the Crank-Nicolson method
    a0 = 1 + 2 * alpha
    c0 = 1 - 2 * alpha

    # Spatial and temporal grids
    xspan = np.linspace(x0, xL, M)
    tspan = np.linspace(0, T, N)

    # Tri-diagonal matrices for the implicit step
    maindiag_a0 = a0 * np.ones((1, M))
    offdiag_a0 = (-alpha) * np.ones((1, M - 1))
    A = sparse.diags([maindiag_a0, offdiag_a0, offdiag_a0], [0, -1, 1], shape=(M, M)).toarray()
    A[0, 1] = A[M-1, M-2] = (-2) * alpha

    # Tri-diagonal matrices for the explicit step
    maindiag_c0 = c0 * np.ones((1, M))
    offdiag_c0 = alpha * np.ones((1, M - 1))
    Arhs = sparse.diags([maindiag_c0, offdiag_c0, offdiag_c0], [0, -1, 1], shape=(M, M)).toarray()
    Arhs[0, 1] = Arhs[M-1, M-2] = 2 * alpha

    # Matrix U for solution
    U = np.zeros((M, N))

    # Initial condition function
    def u_init(x):
        return 4 * x - x**2 / 2

    # Set initial condition
    U[:, 0] = u_init(xspan)

    # Left boundary condition
    f = np.zeros(N)
    U[0, :] = f

    # Right boundary condition
    g = np.zeros(N)
    U[-1, :] = g

    # Time-stepping using Crank-Nicolson method
    for k in range(1, N):
        # Construct the right-hand side
        b1 = np.asarray([4 * alpha * dx * f[k], 4 * alpha * dx * g[k]])
        b1 = np.insert(b1, 1, np.zeros(M - 2))
        b2 = np.matmul(Arhs, U[0:M, k-1])
        b = b1 + b2

        # Solve the linear system using the implicit step
        U[0:M, k] = np.linalg.solve(A, b)

    return U, tspan, xspan



def save_solution_table(U, tspan, xspan, filename):
    with open(filename, 'w') as file:
        file.write("t\t\t")
        for x in xspan:
            file.write(f"{x:.2f}\t")
        file.write("\n---------------------------------------------------------\n")

        for i, t in enumerate(tspan):
            file.write(f"{t:.2f}\t|\t")
            for j in range(U.shape[0]):
                file.write(f"{U[j, i]:.6f}\t")
            file.write("\n")

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
    u[0, :] = 0  # u(0, y) = 0
    u[-1, :] = 2 * np.exp(y[-1, :])  # u(2, y) = 2e^y

def save_output_table(u, x, y, nx, ny, filename):
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
    

    
#QUESTION 3 ASSIGNMENT 2    
    
def calculate_norm(r):
    """
    Calculates the Euclidean norm (L2 norm) of a vector r and returns the Euclidean norm of the vector r.
    """
    norm_squared = sum(element ** 2 for element in r)
    norm = norm_squared ** 0.5
    return norm


def conjugate_gradient(A, b, x0, tol=10**(-4), max_iter=100):
   
    x = np.array(x0, dtype=float)  # Initial guess
    r = b - np.dot(A, x)  # Initial residual
    p = r.copy()  # Initial search direction
    r_r = np.dot(r, r)
   
    
    for k in range(max_iter):
        Ap = np.dot(A, p)
        alpha = r_r / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        
        r_r_next = np.dot(r, r)
        beta = r_r_next / r_r
        p = r + beta * p
        r_r = r_r_next
        
        if calculate_norm(r) < tol:
            break
        
    return x
  
    
    
#QUESTION 4 ASSIGNMENT 2

def matrix_A_ij(x):
    """
    Calculate the matrix-vector product Ax for the given vector x.
    Returns The result of the matrix-vector product Ax.
    """
    m = 0.2
    N = len(x)
    delta = 1.0 / N
    result = np.zeros_like(x)
    
    for i in range(N):
        result[i] += (delta + m) * x[i]
        result[i] -= 2 * delta * x[i]
        result[i] += delta * x[(i + 1) % N]  # Periodic boundary condition
        result[i] += delta * x[(i - 1) % N]  # Periodic boundary condition
        result[i] += m ** 2 * delta * x[i] 
    #print(result)
    return result

def conjugate_fly(matrix_A_ij, b, x0, tol=10**(-6), max_iter=100):
    """
    Conjugate Gradient method for solving linear systems Ax = b.
    Returns: The approximate solution vector x and the List of residue norms at each iteration step.
    """
    it = 0
    x = x0.copy()  # Initial guess
    r = b - matrix_A_ij(x)  # Initial residual
    p = r.copy()  # Initial search direction
    residue_norms = [l.calculate_norm(r)]  # List to store residue norms

    for k in range(max_iter):
        Ap = matrix_A_ij(p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        
        beta = np.dot(r, r) / np.dot(r - alpha * Ap, r - alpha * Ap)
        p = r - alpha * Ap + beta * p

        residue_norm = l.calculate_norm(r)
        residue_norms.append(residue_norm)
        it= it+1
        if residue_norm < tol:
            break

    return x, residue_norms, it


def conjugate_inv(matrix_A_ij, b, x0, tol=10**(-6), max_iter=100):
    N = len(b)
    inverse_columns = []
    
    for i in range(N):
        # Create the right-hand side vector for solving Ax = e_i
        ei = np.zeros(N)
        ei[i] = 1
        
        # Solve the equation Ax = e_i using Conjugate Gradient method
        x, _, _ = conjugate_fly(matrix_A_ij, ei, x0, tol, max_iter)
        
        # Append the solution (column of the inverse matrix) to the list
        inverse_columns.append(x)
    
    # Stack the columns of the inverse matrix horizontally to form the complete inverse matrix
    A_inv = np.column_stack(np.round(inverse_columns,4))
    return A_inv

#################################################################################
#Assignment 4

#Largest eigenvalue of a matrix using power iteration method

def Power_Method(A, it=1000, tol=1e-6):
    n = A.shape[0]
    x0 = np.random.rand(n)  #Initial guess

    for _ in range(it):
        x1 = np.dot(A, x0)
        ev = np.linalg.norm(x1)
        x1 = x1 / ev  # Normalizing eigenvector
        if np.linalg.norm(x0 - x1) < tol:
            break
        x = x1

    return ev, x


#QR factorization using Gram-Scmidt orthogonalization

def QR_Gram_Schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

#Find all Eigenvalues

def eigen_value(A, it = 50000):
    A_k = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(it):
        Q, R = QR_Gram_Schmidt(A_k)
        A_k = R @ Q
        ev = []
    for i in range (0, A_k.shape[0]):
        ev.append(A_k[i, i])
    ev = np.array(ev)
    return ev, A_k, QQ

#Polynomial fit

def pol_fit(x_val, y_val, basis):
    avg = 0
    #print(x_val.shape)
    for x in (y_val):
        avg += x
    avg = avg/(x_val.shape)
    lhs = basis(x_val) @ basis(x_val).T
    rhs = basis(x_val) @ y_val.T
    par = np.linalg.inv(lhs)@rhs
    return par, np.linalg.cond(lhs)


#####################################################################################

#ASSIGNMENT 4


# Define the multiplicative linear congruential generator function
def cong_lcg(a, m, c=0, x = 10):
	while True:
		x = (a * x + c) % m
		yield x / m
#a is the seed, m is the period parameter       

def lcgplot(a,m,c,x):
    rnum = []
    it = []
    for i in range(0,100):
        y = (a*x + c) % m
        rnum.append(y/m)
        x=y
        it.append(i)  
    plot=plt.scatter(it, rnum, label=f'a={a}, m={m}') 
    return plot


# Monte Carlo integration
def Monte_carlo_int(f, a, b, n, gen):
	sum = 0.0
	for _ in range(n):
		x = a + (b - a) * next(gen)  # Scale the random number to the interval [a, b]
		sum += f(x)
	return (b - a) * sum / n


# Monte Carlo integration with importance sampling
def mc_imp_sampling(f, p, inverse_cdf_p, n):
    samples = inverse_cdf_p(np.random.uniform(0, 1, n))
    weights = f(samples) / p(samples)
    return np.mean(weights), np.var(weights)
