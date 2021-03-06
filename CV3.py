from scipy.optimize import linprog
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

#CV3 #1
c = [-3,-5]
A = [[1,0],[0,2],[1,2]]
b = [4,12,18]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#1 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}, value = {-round(res.fun)}')

#CV3 #2
c = [-3,-2]
A = [[2,1],[1,2]]
b = [6,6]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#2 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}, value = {-round(res.fun)}')

