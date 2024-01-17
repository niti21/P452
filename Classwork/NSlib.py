get_ipython().run_line_magic('matplotlib', 'inline')
import math
import matplotlib.pyplot as plot
import random
import copy
import numpy as np

def lcg(x):
    a = 1103515245
    c = 12345
    m = 32768
    
    y = ((a*x + c) % m)
    return y

def random(x):
    a = 1103515245
    c = 12345
    m = 32768
    
    y = ((a*x + c) % m)/m
    return y

def randomAB(r,A,B):
    
    a = 1103515245
    c = 12345
    m = 32768

    x = ((a*r+c)%m)/m
    
    y= A + (B-A)*x
        
    return y


# In GENERAL
def readlist(file):
    b3 = open('file', 'r')
    b= b3.read().split(',')
    for i in range(0,len(b)):
        b[i]=float(b[i])
    #print(b)
    return b

def readMatrix(file):
    with open(file, 'r') as f:
        a = [[float(num) for num in line.split(',')] for line in f]

    return a

# Prints matrix as written on paper
def printMatrix(a, n):
    for i in range(n):
        print(a[i])
        
        
def ceil(x):
    return int(x) + 1


        
#To check for tolerance
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

# Partial pivoting
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

#Swapping individual rows
def swap(A,i,j):
    A[i],A[j]=A[j],A[i]
    return A
   
# MATRIX MULTIPLICATION
def mat_mul(A,B):
    if len(A[0]) == len(B):
        a = list(A)
        b = list(B)
        R = []
        res = [[0 for x in range(len(b[0]))]
             for y in range(len(a))]
        for i in range(0,len(a)):
            s1 = 0
            for j in range(0, len(b[0])):
                for k in range(0,len(b)):
                    s1 += a[i][k]*b[k][j]
                res[i][j] = s1
        for r in res:
            R.append(r)
        return R
    else:
        print('Incompatible Matrices!')

# DOT PRODUCT
def dot(A,B):
    res = 0
    for i in range(0,len(A)):
        for j in range(0, len(B[0])):
            res += A[i][j]*B[i][j]
    return res

# Determinant of a matrix
def det(a):
    b = [0 for i in range(len(a))]
    a, b = pivot(a, b)
    U = crout(a)[0]
    det = 1
    for i in range(len(a)):
        det *= U[i][i]

    # if even number of row exchanges, determinant remains the same, else is multiplied by -1
    if count % 2 == 0:
        return det
    else:
        return -det

    # Function to round off all elements of a matrix

def roundMtrix(M,x):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j]=round(M[i][j],x)
    return M



# Forms Table
def write_table(file, col1, col2, col3):
    table = [[ 0 for i in range(3)] for j in range(len(col1))]
    f = open(file, 'w')
    for i in range(len(col1)):
        for j in range(3):
            table[i][:] = [col1[i], col2[i], col3[i]]

        f.writelines(str(table[i])[1:-1] + "\n")
    f.close()

    return table

# Outputs the data into a table format
def print_table(table, head1, head2, head3):
    data = table.copy()
    col1 = [head1]       # stores first column
    col2 = [head2]   # stores second column
    col3 = [head3]      # stores third column
    for i in range(len(data)):
        col1.append(data[i][0])
        col2.append(data[i][1])
        col3.append(data[i][2])

    for i in range(len(data)+1):
        print(col1[i], col2[i], col3[i])

    return 0

def readData(file):
    x = []   # holds x vals
    y = []   # holds y vals
    ct = 0   # variable to check if value is in first column (x) or 2nd column (y)
    with open(file, 'r') as f:
        for line in f:
            for num in line.split('\t'):
                if ct % 2 == 0:
                    x.append(float(num))
                else:
                    y.append(float(num))

                ct += 1

    return x, y


def lin_solver(A, b):
    pivot(A, b)
    U, L = crout(A)
    x = forward_backward(U, L, b)
    return 



#Checks if matrix is symmetric
def symmetric(A):                                            
    i=0
    cnt=0
    while i<len(A):
        j=0
        while j<len(A):
            if A[i][j]==A[j][i]:
                cnt+=1    
            j+=1
        i+=1
    if cnt==len(A)*len(A):
        print('Matrix is Symmetric!')
        return True
    else:
        print('Not Symmetric!')
        return False
    
def is_symetric(A):
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if A[i][j] != A[j][i]:
                return False
    return True


#Transpose of a square matrix
def transpose(a):
    i=0
    while i<len(a):
        j=i+1
        while j<len(a):
            a[i][j],a[j][i] = a[j][i],a[i][j]
            j+=1
        i+=1
    return a

#For a diagonally dominant matrix.(to find the maximum element in a column and swap it with the diagonal element.)
def DiagDom(A,X):
    i=0
    n=len(A)
    for i in range(n):               
        maxima=A[0][i]              
        for j in range(n):
            if A[j][i]>maxima:
                maxima=A[j][i]
                swap(A, i, j)
                swap(X, i, j)
                if A[j][j]==0:
                    swap(A, i, j)
                    swap(X, i, j)
    return A,X


def lcgshow(a,c,m,x):
    rnum = []
    for i in range(0,500):
        y = (a*x + c) % m
        rnum.append(y)
        x=y
        i= i+1  
        plt=plot.scatter(i, y) 
    return rnum

########################################################################################

def lcgplot(a,c,m,x):
    rnum = []
    for i in range(0,500):
        y = (a*x + c) % m
        rnum.append(y)
        x=y
        i= i+1  
        plt=plot.scatter(i, y) 
    return plt

def displaylcg(x):
    print(lcgshow(1103515245, 12345, 32768, x))
    lcgplot(1103515245, 12345, 32768, x)

########################################################################################      


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

def determinant(mat,n):
    det=1
    for i in range(n):
        det*=-1*mat[i][i]
    return det

# Function for partial pivot for LU decomposition

def partial_pivot_LU (mat, vec, n):
    for i in range (n-1):
        if mat[i][i] ==0:
            for j in range (i+1,n):
                # checks for max absolute value and swaps rows 
                # of both the input matrix and the vector as well
                if abs(mat[j][i]) > abs(mat[i][i]):
                    mat[i], mat[j] = mat[j], mat[i]
                    vec[i], vec[j] = vec[j], vec[i]
    return mat, vec

def get_identity(n):
    I=[[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        I[i][i]=1
    return I


# LU decomposition using Doolittle's condition L[i][i]=1
# without making separate L and U matrices

def LU_doolittle(mat,n):
    for i in range(n):
        for j in range(n):
            if i>0 and i<=j: # changing values of upper triangular matrix
                sum=0
                for k in range(i):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=mat[i][j]-sum
            if i>j: # changing values of lower triangular matrix
                sum=0
                for k in range(j):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=(mat[i][j]-sum)/mat[j][j]
    return mat



# Function to find the solution matrix provided a vector using 
# forward and backward substitution respectively

def for_back_subs_doolittle(mat,n,vect):
    # initialization
    y=[0 for i in range(n)]
    x=[0 for i in range(n)]
    # forward substitution
    y[0]=vect[0]
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=mat[i][j]*y[j]
        y[i]=vect[i]-sum
    
    # backward substitution
    
    x[n-1]=y[n-1]/mat[n-1][n-1]
    for i in range(n-1,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=mat[i][j]*x[j]
        x[i]=(y[i]-sum)/mat[i][i]
    del(y)
    return x

def inverse_by_lu_decomposition (matrix, n):

    identity=get_identity(n)
    x=[]
    det = determinant(matrix,n)
    det1 = round(det,4)
    if det1 == 0:
        print ('Inverse does not exist as determinant is 0' )
    else:
        print('Determinant: ', det1, ' ; Inverse exists!')
    
    #deepcopy() is used so that the original matrix doesn't change on changing the copied entities. We reuire the original multiple times here
    
    matrix_0 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_0, identity[0], n)
    
    matrix_0 = LU_doolittle(matrix_0, n)
    x0 = for_back_subs_doolittle(matrix_0, n, identity[0])
    
    x.append(copy.deepcopy(x0))


    matrix_1 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_1, identity[1], n)
    
    matrix_1 = LU_doolittle(matrix_1, n)
    x1 = for_back_subs_doolittle(matrix_1, n, identity[1])
    
    x.append(copy.deepcopy(x1))

    matrix_2 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_2, identity[2], n)
    

    matrix_2 = LU_doolittle(matrix_2, n)
    x2 = for_back_subs_doolittle(matrix_2, n, identity[2])
    x.append(copy.deepcopy(x2))

    matrix_3 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_3, identity[3], n)
    
    matrix_3 = LU_doolittle(matrix_3, n)
    x3 = for_back_subs_doolittle(matrix_3, n, identity[3])
    
    x.append(copy.deepcopy(x3))
    
    matrix_4 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_4, identity[4], n)
    
    matrix_4 = LU_doolittle(matrix_4, n)
    x4 = for_back_subs_doolittle(matrix_4, n, identity[4])
    
    x.append(copy.deepcopy(x4))
    
    # The x matrix to be transposed to get the inverse in desired form
    inv=transpose(x)
    Inverse=roundMtrix(inv,3)
    print('The inverse matrix: ')
    return (Inverse)





def forwardsub(A,B):
    i=0
    Y=[]
    for k in range(len(A)):
        Y.append(0)
    while i<len(A):
        j=0
        temp=0
        while j<i:
            temp+=A[i][j]*Y[j]
            j+=1
        Y[i]=round(((B[i]-temp)/A[i][i]),4)
        i+=1
    print()
    return Y

def backwardsub(A,Y):
    i=len(A)-1
    X=[]
    for l in range(len(A)):
        X.append(0)
    while i>=0:
        j=i+1
        temp=0
        while j<len(A):
            temp+=A[i][j]*X[j]
            j+=1
        X[i]=round(((Y[i]-temp)/A[i][i]),4)
        i-=1
    print()
    return X


#To decomposes A into Lower and Upper which are transpose of each other if A is symmetric
def cholesky(A):
    #Check for symmetric matrix
    if symmetric(A):                                         
        i=0
        while i <len(A):
            j=0
            temp=0
            while j<i:
                temp+=round((A[j][i]*A[j][i]),4)
                j+=1
            A[i][i]=round(((A[i][i]-temp)**(0.5)),4)                
            j=i+1
            while j<len(A):
                k=0
                temp=0
                #Recurrence relations
                while k<i:
                    temp+= round((A[i][k]*A[k][j]),4)
                    k+=1
                A[j][i]=round(((A[j][i]-temp)/A[i][i] ),4)          
                A[i][j]=A[j][i]
                j+=1
            i+=1
        i=0
        while i <len(A):                                   
            j=i+1
            while j<len(A):
                A[i][j]=0
                j+=1
            i+=1
        
        print()
        print("Cholesky Decomposed matrix: ")
        print(A)
        return A


#function for gauss-jordan elimination
def gauss_jordan(A,B):
    n = len(A)
    
    #partial pivoting
    A,B=pivot(A, B)             
    for i in range(n):
        p = A[i][i]
        #Making diagonal terms 1
        for j in range(i,n):
            A[i][j] = A[i][j]/p
        B[i] = B[i]/p 

        for k in range(n):              
            #Making Column zero except diagonal
            if (k == i) or (A[k][i] == 0):
                continue
            else:
                f = A[k][i]
                for l in range(i,n):
                    A[k][l] = A[k][l] - f*A[i][l]
                B[k] = B[k] - f*B[i]
    #print('Gauss jordan solution: ')
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

#########################################

#Deflation
def deflate(c,x):
    root = x
    k = len(c)
    for i in range (0,k-1):
        c[i+1] = c[i+1]+ root*c[i]  
        print (c) 
    return c

#Recurssion deflation
def def2(c):
    for a in range(k-1,2,-1):
        if c[a]==0:
            x = c[a-1]
        d = deflate(c,x)
        print(x)
    return d

#######################################

#Alternative to use the same fn to deflate again
def deflation(c,a):
    if len(c) != 1:
        c[1] = c[1] + c[0]*a
        for i in range(2,len(c)):
            c[i] = c[i] + a*c[i - 1]
        c.pop()
    else:
        print("cannot deflate")
    return c

#########################################

# To make a copy of given function
def copy_list(c):
    cn = []
    for i in range(0,len(c)):
        a = c[i]
        cn.append(a)
    return cn 

# Bracketing
def mybracket(f,a,b,func):
    t=0
    while (f(a)*f(b)>=0):
        t=t+1
        if (abs(f(a)) < abs(f(b))):
            a=a-0.1
        if (abs(f(a)) > abs(f(b))):
            b= b+0.1
        print()
        call = func
    print()
    print('Steps: ',t)
    return call

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

#Bisection Method
def bisection(f,a,b,tol):
    if (f(a)*f(b)>=0):
        print('wrong interval')
    c = a
    print ("%10.2E" %a, "%10.2E" %f(a))
    print ("%10.2E" %b, "%10.2E" %f(b))
    while ((b-a)>= tol):
        for i in range(15):
            c=(a+b)/2
            if (f(c)==0):
                break
            if (f(c)*f(a)<0):
                b=c
            else:
                a=c
    print ("%10.2E" %c, "%10.2E" %(f(b)))
    return c

#Regula Falsi Method
def regulaFalsi(f,a,b,tol):
    if f(a) * f(b) >= 0:
        print("Wrong interval")
        return -1
    o=0 
    c = a 
    print (a, f(a))
    print (b, f(b)) 
    for i in range(n):
        #point touching x axis
        c = (b - ((b - a) * f(b))/ (f(b) - f(a))) 
        # Find root
        if f(c) == tol:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    print("Roots : " , '%10.2E' %c, 'at: ', '%10.2E' %f(c) )
    
    
###################################################################

# For normal use define function as f(z)

# To use a function in form of list
def fn(c,z):
    sum1 =0
    for i in range(0,len(c)):
        sum1 = sum1 + c[i]*(z**(len(c)-i-1))
    return sum1

#To find derivative 
def deriv1(c1):
    l = len(c1)
    for i in range (0,l):
        c1[i]= c1[i]*i
        cf = c1
    
    return cf

#to find 2nd derivative
def deriv2(c2):
    for i in range (1,l+2):
        c2[i]= c2[i]*(i-1)
    return c2

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

# Function for finding double derivative of a function at given x fo h

def double_derivate(f,x):
    h=10**-8
    fdd=(f(x+h)+f(x-h)-2*f(x))/(2*h) # Double derivative algorithm
    return fdd


###############################################################################
    

#Laguerre Method
#f = function, df = derivative of f, d2f = 2nd dereivative of f
def laguerreMethod(f,df,d2f, x0, n, e):
    xk = x0
    t=0
    while abs(f(xk)) > e:
        G = df(xk) / f(xk)
        H = (G ** 2) - d2f(xk) / f(xk)
        root = math.sqrt((n - 1) * (n * H - G ** 2))
        d = max(abs(G + root), abs(G - root))
        a = n / d
        xk = xk - a
        t=t+1
    print("Root: ", xk)
    print("No. of iterations:", t)
    return xk

# Polynomial root solver using Laguerre's method
def poly_lag(c, z, tol=1e-6):
    roots = []
    index = -1  # holds index position of newly added root
    while(len(c) > 1):
        roots.append(laguerre(c, z, tol))
        index += 1
        coeffs = deflation(c, roots[index])

    return roots


###########################################
def laguerre_2(c,z,degree,tol):
    d = 0
    t = 7
     
    k1 = copy_list(c)
    k2 = copy_list(c)
    k2 = derivative(k2)
   
    while d < degree:
        if abs(fn(c,z))<tol:
            print(fn(c,z))
            print(z, "is a root")
        else:
            k1 = derivative(k1)
            k2 = derivative(k2)
            # print(c)
            # print(k1)
            # print(k2)
            count = 0
            while abs(fn(c,z))>tol:
                t = z
               
                sum1 = 0
                for i in range(0,len(k1)):
                    sum1 = sum1 + k1[i]*(z**(len(k1)-i-1))
                S = sum1/fn(c,z)
               
                sum2 = 0
                for i in range(0,len(k2)):
                    sum2 = sum2 + k2[i]*(z**(len(k2)-1-i))
                p = S**2 - (sum2/fn(c,z))
               
                n = len(c)-1
                if S < 0:
                    a = n/(S - ((n-1)*(n*p - S**2))**0.5)
                else:
                    a = n/(S + ((n-1)*(n*p - S**2))**0.5)
                z = z - a
                count = count + 1
        print(z,"is a root")
        # print(count,'is the number of iterations used to get',z)
        c = deflation(c,z)
        k1 = copy_list(c)
        k2 = copy_list(c)
        k2 = derivative(k2)
        d=d+1


#Newton raphson method
#f = function, df = derivative of f
#prolly wrong
def newtonRaphson(f,df,q,tol):
    t=0
    h = f(q) / df(q)
    while abs(h) >= tol:
        h = f(q)/df(q)
        q = q - h
        t=t+1
     
    print("Root: ", q)
    print("No. of iterations:", t)
    return q

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


#Data Fit

#Interpolation
def interpolate(a,b,x):
    s = 0
    l = len(a)
    for i in range(l):
        p = 1
        for k in range(l):
            if k != i:
                p *= ((x - a[k])/(a[i]-a[k]))
        s+=p*b[i]
    print (x,':', s)
    print(l,a,b)
    return s
    
#Least Sq fit
def augmentMat(z,x):
    for i in range(0,len(z)):
        z[i].append(x[i])
        #print(z[i])
    return z

def polySF(X,Y,degree):
    plot.plot(X,Y)
    plot.show()  
    V= []
    for i in range(0,degree):
        a = []
        for j in range(0,degree):
            a.append(0)
        V.append(a)
    z = []
    c = []
    for j in range(0,2*len(V)-1):
        sum1= 0
        sum2 = 0
        for i in range(0,len(X)):
            sum1 =sum1+(X[i]**j)
            sum2 = sum2 + Y[i]*(X[i]**(j))
        z.append(sum1)
        c.append(sum2)
    c.pop()
    c.pop()
    for i in range(0,len(V)):
        for j in range(0,len(V)):
            V[i][j] = z[i+j]
    augmentMat(V,c)
    gauss_jordan(V,c)
     
    return V,c


####################################################################
#Pearson R
def persons_ratio(x,y):
    #for sigma =1
    n=len(x)
    sx,sy,sxy,sx2,sy2=0,0,0,0,0
    for i in range(n):
        sx+=x[i]
        sy+=y[i]
        sxy+=x[i]*y[i]
        sx2+=x[i]**2
        sy2+=y[i]**2
    R=(n*sxy - sx*sy)/(((n*sx2 - sx*2)**0.5)*((n*sy2 - sy*2)**0.5))
    return R

#ALTERNATE

def pearson_r(xvals, yvals):
    '''xvals, yvals: data points given as a list (separately) as input
    '''
    n = len(xvals)   # number of datapoints

    xbar = sum(xvals)/n
    ybar = sum(yvals)/n

    sxx = 0  # sigma**2 * n
    for i in xvals:
        sxx += (i - xbar)**2

    sxy = 0  # covariance * n
    for i in range(n):
        sxy += (xvals[i] - xbar) * (yvals[i] - ybar)

    syy = 0
    for j in yvals:
        syy += (j - ybar)**2

    r2 = sxy**2 / (sxx * syy)
    r = r2**(0.5)

    return r


#Linear Fit
def linear_fit(X,Y):
    # plt.scatter((X),Y)
    # plt.show()  
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_xy = 0
    sum_y2 = 0
    N = len(X)
    for i in range(0,len(X)):
        sum_x = sum_x + (X[i]/sigma[i])
        sum_x2 = sum_x2 + ((X[i]**2)/sigma[i])
        sum_y = sum_y + (Y[i]/sigma[i])
        sum_xy = sum_xy + ((X[i]*Y[i])/sigma[i])
        sum_y2 = sum_y2 + ((Y[i]**2)/sigma[i])
    a1 = (N*sum_xy-sum_x*sum_y)/(N*sum_x2-(sum_x**2))
    a2 = (sum_y - a1*sum_x)/(N)
   
    delta_x = N*sum_x2-(sum_x**2)
    delta_y = N*sum_y2-(sum_y**2)
    r = ((N*sum_xy - sum_x*sum_y)**2)/(delta_x*delta_y)
    return a1,a2,r




###################################################################################################
# POST MID SEM
###################################################################################################
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

def perdictor_corrector(x0,y0,N,x1,f):
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


# FORWARD EULER

def for_euler(dydx, y0, x0, xf, st):
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    
    n = int((xf-x0)/st)
    for i in range(n):
        x.append(x[i] + st)
    for i in range(n):
        y.append(y[i] + st * dydx(y[i], x[i]))

    return x, y

# RK4
def RK( dydx, x0, y0, xf, st):
    
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


# COUPLED ODE Lorentz
def RKLorentz(dxdt, dydt, dzdt, x0, y0, z0, t0, tf, st):
    
    x = [x0]
    y = [y0]
    z = [z0]      
    t = [t0]
    n = int((tf-t0)/st)     # no. of steps
    for i in range(n):
        t.append(t[i] + st)
        k1 = st * dxdt(x[i], y[i], z[i], t[i])
        l1 = st * dydt(x[i], y[i], z[i], t[i])
        m1 = st * dzdt(x[i], y[i], z[i], t[i])
        
        k2 = st * dxdt( x[i] + k1/2, y[i] + l1/2, z[i] + m1/2, t[i] + st/2,)
        l2 = st * dydt( x[i] + k1/2, y[i] + l1/2, z[i] + m1/2, t[i] + st/2,)
        m2 = st * dzdt( x[i] + k1/2, y[i] + l1/2, z[i] + m1/2, t[i] + st/2,)
        
        k3 = st * dxdt( x[i] + k2/2, y[i] + l2/2, z[i] + m2/2, t[i] + st/2,)
        l3 = st * dydt( x[i] + k2/2, y[i] + l2/2, z[i] + m2/2, t[i] + st/2,)
        m3 = st * dzdt( x[i] + k2/2, y[i] + l2/2, z[i] + m2/2, t[i] + st/2,)
        
        k4 = st * dxdt( x[i] + k3, y[i] + l3, z[i] + m3, t[i] + st,)
        l4 = st * dydt( x[i] + k3, y[i] + l3, z[i] + m3, t[i] + st,)
        m4 = st * dzdt( x[i] + k3, y[i] + l3, z[i] + m3, t[i] + st,)
        

        x.append(x[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        y.append(y[i] + (l1 + 2*l2 + 2*l3 + l4)/6)
        z.append(z[i] + (m1 + 2*m2 + 2*m3 + m4)/6)

    return t, x,y,z



# PDE finite diff

#Lagrange Interpolation
def LagInterpolation(zh, zl, yh, yl, y):
    z = zl + ((zh - zl) * (y - yl)/(yh - yl))
    return z


# SHOOTING METHOD

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
            plot.plot(X, V0)
    plot.show()

    
# Dominant Eigen value and it Eigen vector
def Evalue(A, x):
    k = 1
    xk1 = x[:]
    xk2 = mat_mul(A,xk1)
    e1 = 0
    e2 = dot(xk2,x)/dot(xk1,x)
    t = 0
    while abs(e1-e2) > 10**(-3) and k < 50:
        e1 = e2
        xk1 = xk2
        xk2 = mat_mul(A,xk1)
        e2 = dot(xk2,x)/dot(xk1,x)
        k = k + 1
        t = t + 1
    ev = []
    n = 0
    
    for i in range(len(xk2)):
        n = n + xk2[i][0]**2
    n = math.sqrt(n)
    for i in range(len(xk2)):
        ev.append(xk2[i][0]/n)
        ev[i] = (ev[i]*10**3)/10**3
    e = (e2*10**3)/10**3
    return e,ev,t

# RADIOACTIVE DECAY
def DecayABC(Na,Nb,Nc,Ta,Tb,T,dT):
    lA = math.log(2)/Ta
    lB = math.log(2)/Tb
   
    r = 10
    t = 0
   
    NA=[Na]
    NB=[0]
    NC=[0]
    NT=[0]
    i=0
    while i in range(0,int(T/dT)) and Na >=0:
       
        for j in range(0,Na):
            if r <= lA*dT:
                Na = Na - 1
                Nb = Nb + 1
       
            r = l.random(r)
        for j in range(0,Nb):
            if r <= lB*dT:
                Nb = Nb - 1
                Nc = Nc + 1
       
            r = l.random(r)
       
        NA.append(Na)
        NB.append(Nb)
        NC.append(Nc)
        t +=dT
        NT.append(t)
        i=i+1
   
    return NA,NB,NC,NT


def prbRL(nL,nR,T,dT):
    Nmax=nL+nR
    NL=[nL]
    NR=[nR]
    NT=[0]
    r = random(T)
    t=0
   
    while t <= int(T) and nL >=0 and nL <=Nmax and  nR <=Nmax and nR >=0:
        r = random(r)
        probtoR = NL[len(NL)-1]/Nmax
        if r <= probtoR:
            nL = nL -1
            nR = nR +1
            t+=dT
            NL.append(nL)
            NR.append(nR)
            NT.append(t)
           
            r=random(r)
       
        probtoL = 1 - (NL[len(NL)-1]/Nmax)
        r = random(r)
        if r <= probtoL:
            nR = nR -1
            nL = nL +1
            t+=dT
            NL.append(nL)
            NR.append(nR)
            NT.append(t)
           
            r=random(r)
       
       
    return NL,NR,NT

