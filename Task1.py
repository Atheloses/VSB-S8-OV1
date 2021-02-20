from scipy.optimize import linprog

#Task 1
c = [-1,-2]
A = [[1,3],[2,2],[0,1]]
b = [200,300,60]
x0_bounds = (0,None)
x1_bounds = (0,None)

res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds])
#print(res)
print(f'Task 1: \nx1 = {round(res.x[0])}, x2 = {round(res.x[1])}, profit = {-round(res.fun)}')
