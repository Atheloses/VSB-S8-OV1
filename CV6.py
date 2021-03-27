from pulp import *
import numpy as np


Factories = ["A", "B", "C", "D", "E"]

supply = {"A": 400, "B": 600, "C": 400, "D": 600, "E":1000}

Products = ["1", "2", "3", "4"]

demand = {"1": 700, "2":1000, "3":900, "4":400}

costs = {'A':{"1":41, "2": 55, "3": 48, "4":0 },
         'B':{"1":39, "2": 51, "3": 45, "4":0 },
         'C':{"1":42, "2": 56, "3": 50, "4":0 },
         'D':{"1":38, "2": 52, "3": 1e6, "4":0 },
         'E':{"1":39, "2": 53, "3": 1e6, "4":0 }
         }
#nebo
#costs = makeDict((Warehouses, Bars),costs)

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

prob.solve()
for v in prob.variables():
    print(v.name, "=", v.varValue)

print('Total costs = ', value(prob.objective))




Factories = ["A", "B", "C", "D"]

supply = {"A": 50, "B": 60, "C": 50, "D": 50}

Products = ["1", "2", "3", "4", "5"]

demand = {"1": 30, "2":20, "3":70, "4":30, "5":60}

costs = {'A':{"1":16, "2": 16, "3": 13, "4":22, "5": 17},
         'B':{"1":14, "2": 14, "3": 13, "4":19, "5": 15},
         'C':{"1":19, "2": 19, "3": 20, "4":23, "5": 1e6},
         'D':{"1":1e6, "2": 0, "3": 1e6, "4":0, "5": 0}
         }
#nebo
#costs = makeDict((Warehouses, Bars),costs)

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

prob.solve()
for v in prob.variables():
    print(v.name, "=", v.varValue)

print('Total costs = ', value(prob.objective))
