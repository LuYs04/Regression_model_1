import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from random import randint

exp_count = int(input("Experiments' count: "))
x_len = int(input("Count of x-es: "))

# input X
def inputX(n, x_len):
    x = np.zeros((n, x_len))
    for i in range(n):
        for j in range(x_len):
            #x[i][j] = eval(input("x = "))
            x[i][j] = randint(1, 7)
    return (x)

#input Y
def inputY(n):
    y = np.zeros(n)
    for i in range(n):
        #y[i] = eval(input("y = "))
        y[i] = randint(1, 7)
    return (y)

#sum_x
def sum_x(x, i, n):
    s = 0
    for k in range(n):
        s += x[k, i]
    return (s)    
        
#sum_xx
def sum_xx(x, i, j, n):
    s = 0
    for k in range(n):
        s += x[k, i]*x[k, j]
    return (s)

# getting A
def gettingA(x_len, n, x):
    A = np.zeros((x_len + 1, x_len + 1))
    for i in range(x_len + 1):
        for j in range(x_len + 1):
            if i == 0 and j == 0:
                A[i][j] = n
            elif i == 0 and j != 0:
                A[i][j] = sum_x(x, j - 1, n)
            elif i != 0 and j == 0:
                A[i][j] = sum_x(x, i - 1, n)
    for i in range(1, x_len + 1):
        for j in range(1, x_len + 1):
            if i == j:
                A[i, j] = sum_xx(x, j - 1, j - 1, n)
            else:
                A[i, j] = sum_xx(x, j - 1, i - 1, n)
    return (A)

#getting Y
def gettingY(y, x, x_len, n):
    y_n = np.zeros((x_len + 1, 1))
    y_n[0] = np.sum(y)
    for i in range(1, x_len + 1):
        for k in range(n):
            y_n[i] += x[k, i - 1]*y[k]
    return (y_n)
        
#getting B
def gettingB(A, y):
    if det(A) != 0:
        g = inv(A)
    else:
        print("the determinant of A is 0 :(")
    #print("A_inv = \n", g, "\n\n")
    b = np.matmul(g, y)
    return (b)

#getting Y_m
def gettingY_m(n, x_len, b):
    y_m = np.zeros(n)
    for i in range(n):
        for k in range(x_len + 1):
            if k == 0:
                y_m[i] += b[k]
            else:
                y_m[i] += b[k]*x[i, k - 1]
    return (y_m)

#y_average
def avg_y(Y, n):
    avg = 0
    for i in Y:
        avg += i
    return (avg/n)

#ss_o
def ss_o(Y, avg_y):
    ss_o = 0
    for i in Y:
        ss_o += (i - avg_y)**2
    return (ss_o)

#ss_r
def ss_r(Y_m, avg_y):
    ss_r = 0
    for i in Y_m:
        ss_r += (i - avg_y)**2
    return (ss_r)

#ss_e
def ss_e(Y, Y_m):
    ss_e = 0
    for i in range(0, len(Y)):
        ss_e += (Y[i] - Y_m[i])**2
    return (ss_e)
    
#main
x = inputX(exp_count, x_len)
y = inputY(exp_count)
y_n = gettingY(y, x, x_len, exp_count)
A = gettingA(x_len, exp_count, x)
B = gettingB(A, y_n)
y_m = gettingY_m(exp_count, x_len, B)
ss0 = ss_o(y, avg_y(y, exp_count))
ssE = ss_e(y, y_m)
ssR = ss_r(y_m, avg_y(y, exp_count))
r_2 = ssR/ss0
msR = ssR/x_len
sigma_2 = ssE/(exp_count - x_len - 1)
F = msR/sigma_2

print("X = \n", x, "\n\n")
print("Y = \n", y, "\n\n")
print("A = \n", A, "\n\n")
print("Y_n = \n", y_n, "\n\n")
print("B = \n", B, "\n\n")
print("Y_m = \n", y_m, "\n\n")
print("y_avg = ", avg_y(y, exp_count), "\n\n")
print("ss(0) = ", ss0, "\n\n")
print("ss(E) = ", ssE, "\n\n")
print("r^2 = ", r_2, "\n\n")
print("msR = ", msR, "\n\n")
print("sigma^2 = ", sigma_2, "\n\n")
print("F = ", F, "\n\n")

if (int(ss0) == int(ssE) + int(ssR)):
    print("Yeah! You did it!\nYour model is equivalent!!")
else:
    print("O_o! Your model isn't equivalent :(")

x = np.array([[3, 6, 7, 7], [7, 4, 4, 2], [4, 3, 1, 1], [6, 3, 6, 5], [3, 1, 7, 2], [6, 1, 6, 6], [6, 2, 6, 4], [4, 4, 2, 5], [5, 2, 2, 7], [1, 7, 7, 5], [5, 1, 1, 4], [4, 1, 2, 4], [5, 7, 2, 3], [7, 5, 7, 3], [4, 3, 1, 5]])
y = np.array([4, 7, 2, 7, 6, 2, 4, 5, 4, 3, 5, 2, 3, 1, 6])



