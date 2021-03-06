from scipy.optimize import linprog
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

#CV4 #2
c = [-1,-2]
A = [[1,3],[1,1]]
b = [8,4]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#3 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}, value = {-round(res.fun)}')

#CV4 #3
c = [-2,-3]
A = [[-3,1],[4,2],[4,-1],[-1,2]]
b = [1,20,10,5]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#3 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}, value = {-round(res.fun)}')