from scipy.optimize import linprog

#Task 2
c = [-10,-20]
A = [[-1,2],[1,1],[5,3]]
b = [15,12,45]
x0_b = (0,None)
x1_b = (0,None)

res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_b, x1_b])
#print(res)
print('Task 2: \n' +
    f'x1 = {round(res.x[0])}, ' +
    f'x2 = {round(res.x[1])}, ' +
    f'profit = {-round(res.fun)}')
