from scipy.optimize import linprog
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

#CV2 #1
c = [-5,-2]
A = [[3,2],[0,1],[2,0]]
b = [2400,800,1200]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#1 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}')

#CV2 #2
c = [-0.8,-0.3]
A = [[0,0.1],[1,0],[3,2]]
b = [200,800,12000]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#2 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}')

#CV2 #3
c = [-50,-20]
A = [[9,3],[5,4],[3,0]]
b = [400,350,110]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'#3 x1 = {round(res.x[0])}, x2 = {round(res.x[1])}')


""" xx1 = np.linspace(0,0)
yy1 = np.linspace(0,1000)

xx2 = np.linspace(0,0)
yy2 = np.linspace(0,1800)

x10 = np.linspace(0,800)
x11 = 1200 - 3/2*x10

x20 = np.linspace(0,800)
x21 = np.linspace(800,800)

x30 = np.linspace(600,600)
x31 = np.linspace(0,1200)

fig = plt.figure()
ax = fig.add_subplot(1,1,1) """