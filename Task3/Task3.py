from pulp import *
import numpy as np

#Task 3

Factories = ["1", "2", "3", "4", "5", "6"]

supply = {"1": 10, "2": 3, "3": 8, "4": 2, "5":10, "6":3}

Products = ["1", "2", "3", "4", "5", "6", "7"]

demand = {"1": 5, "2":3, "3":3, "4":5, "5":4, "6":4, "7":12}

costs = {"1":{"1":15, "2": 16, "3": 16, "4": 18, "5": 18, "6": 19, "7": 0 },
         "2":{"1":18, "2": 20, "3": 19, "4": 22, "5": 21, "6": 23, "7": 0 },
         "3":{"1":1e6, "2": 1e6, "3": 17, "4": 15, "5": 19, "6": 16, "7": 0 },
         "4":{"1":1e6, "2": 1e6, "3": 20, "4": 18, "5": 22, "6": 19, "7": 0 },
         "5":{"1":1e6, "2": 1e6, "3": 1e6, "4": 1e6, "5": 19, "6": 17, "7": 0 },
         "6":{"1":1e6, "2": 1e6, "3": 1e6, "4": 1e6, "5": 22, "6": 22, "7": 0 }
         }

prob = LpProblem("Factories and Products", LpMinimize)

Routes = [(f,p) for f in Factories for p in Products]

route_vars = LpVariable.dicts("Route", (Factories, Products), 0, None, LpInteger)

prob += lpSum([route_vars[f][p]*costs[f][p] for (f,p) in Routes]), "Sum of Transporting Costs"

# The supply maximum constraints are added to prob for each supply node (warehouse)
for f in Factories:
    prob += lpSum([route_vars[f][p] for p in Products]) <= supply[f], "Sum of Products out of Plants %s"%f

# The demand minimum constraints are added to prob for each demand node (bar)
for p in Products:
    prob += lpSum([route_vars[f][p] for f in Factories]) >= demand[p], "Sum of Products into Warehouses %s"%p

rts = []

prob.solve()
for v in prob.variables():
    if(v.varValue > 0):
        x = v.name.split("_")
        rts.append(f"x{x[2]}{x[1]} = {v.varValue}")

rts.sort()
for route in rts:
    print(route)

print('Total costs = ', value(prob.objective))
