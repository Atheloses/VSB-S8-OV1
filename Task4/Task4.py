from operator import itemgetter
import numpy as np
from math import sqrt,pi,floor
from functools import partial


def _function_wrapper(func, args, x):
    return func(x, *args)


def _is_feasible_wrapper(func, x):
    values = func(x)
    return np.all(values <= 0)


def _cons_function_wrapper(cons, args, x):
    return np.array(cons(x, *args))


def pso(func, bounds, fcons, args=(),
        swarmsize=100, wmax=0.6, wmin=1.4, c1=0.5, c2=0.5, maxiter=100, error=1e-8):

    # Check user's inputs
    lb = bounds[0]
    ub = bounds[1]
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    # Parameters mapping
    func_to_minimize = partial(_function_wrapper, func, args)
    cons = partial(_cons_function_wrapper, fcons, args)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Generate random positions
    dimensions = len(lb)
    x = np.random.rand(swarmsize, dimensions)
    x = lb + x*(ub - lb)

    # Initialize arrays
    pbest_position = np.zeros_like(x)
    pbest_values = np.ones(swarmsize)*np.inf
    gbest_position = []
    gbest_value = np.inf
    fx = np.zeros(swarmsize)
    fs = np.zeros(swarmsize, dtype=bool)

    # Calculate fitness and feasibility
    for i in range(swarmsize):
        fx[i] = func_to_minimize(x[i, :])
        fs[i] = is_feasible(x[i, :])

    # Store feasible best positions
    i_update = np.logical_and((fx < pbest_values), fs)
    pbest_position[i_update, :] = x[i_update, :].copy()
    pbest_values[i_update] = fx[i_update]

    # Update gbest position
    i_min = np.argmin(pbest_values)
    if pbest_values[i_min] < gbest_value:
        gbest_value = pbest_values[i_min]
        gbest_position = pbest_position[i_min, :].copy()
    else:
        gbest_position = x[0, :].copy()

    # Initialize velocities
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    v = vlow + np.random.rand(swarmsize, dimensions)*(vhigh - vlow)

    # Iteration
    iteration = 1
    while iteration <= maxiter:
        rp = np.random.uniform(size=(swarmsize, dimensions))
        rg = np.random.uniform(size=(swarmsize, dimensions))

        # Update velocities
        w = wmax-(wmax-wmin)/maxiter*iteration
        v = w*v + c1*rp*(pbest_position - x) + c2*rg*(gbest_position - x)
        # Update positions
        x = x + v

        # Must be inside bounds
        maskl = x < lb
        masku = x > ub
        not_violated = x*(~np.logical_or(maskl, masku))
        x = not_violated + lb*maskl + ub*masku

        # Update objectives and constraints
        for i in range(swarmsize):
            fx[i] = func_to_minimize(x[i, :])
            fs[i] = is_feasible(x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < pbest_values), fs)
        pbest_position[i_update, :] = x[i_update, :].copy()
        pbest_values[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(pbest_values)
        if pbest_values[i_min] < gbest_value:
            p_min = pbest_position[i_min, :].copy()

            if np.abs(gbest_value - pbest_values[i_min]) <= error:
                break
            else:
                gbest_position = p_min.copy()
                gbest_value = pbest_values[i_min]

        iteration += 1

    if not is_feasible(gbest_position):
        #print("Feasible positions not found.")
        pass
    else:
        #print(f"Best feasible design found in {iteration} iterations.")
        return [gbest_position, gbest_value]


def coil_cons(x, *args):
    d, D, N = x
    N = round(N)
    Pmax, S, G, Lfree, dmin, Dmax, P, deltapm, dw = args

    C = D/d
    K = (4*C-1)/(4*C-4) + (0.615/C)
    k = (G*d**4)/(8*N*D**3)
    deltap = P/k

    return [
        ((8*K*Pmax*D)/(pi*d**3) - S),
        (Pmax/k - 1.05*(N + 2)*d),
        (dmin - d),
        ((d + D) - Dmax),
        ((3 - D/d)),
        (deltap - deltapm),
        (Pmax/k - 1.05*(N + 2)*d - Lfree),
        (dw - (Pmax - P)/k)
    ]


def coil_minimize(x, *args):
    d, D, N = x
    N = round(N)
    return (pi**2 * D * d**2 * (N + 2) / 4)

Pmax = 453.6  # Fmax
S = 13288.02  # S
E = 30e+06  # -
G = 808543.6  # G
Lfree = 35.56  # lmax
dmin = 0.508  # dmin
Dmax = 7.62  # Dmax
P = 136.08  # Fp
deltapm = 15.24  # deltapm
dw = 3.175  # deltaw

coil_args = (Pmax, S, G, Lfree, dmin, Dmax, P, deltapm, dw)
coil_bounds = [[0.01, 1, 2], [2, 30, 100]]
coil_out = []

def coil_print(x, message):
    [(d, D, N), fx] = x
    print(message, "f(x) = ", format(round(fx, 3), '.3f'), "| d = ", format(round(d, 3), '.3f'),
        "| D = ", format(round(D, 3), '.3f'), "| N = ", round(N))


'''
if __name__ == "__main__":
    for i in range(20):
        coil_out.append(
            pso(coil_minimize, coil_bounds, fcons=coil_cons,
                args=coil_args, swarmsize=20, maxiter=100, error=1e-08,
                wmax=0.6, wmin=1.4, c1=0.5, c2=1.5))

    print("Closed coil helical spring (top 5 out of 20):")
    for row in sorted(filter(None, coil_out), key=itemgetter(1))[:5]:
        coil_print(row,"")
'''

def reducer_cons(x, *args):
    b, z, D1, D2, L1, L2, Z = x
    Z = round(Z)

    return [
        ((27)/(b*z**2*Z) - 1),
        ((397.5)/(b*z**2*Z**2) - 1),
        ((1.93*D1**3)/(z*Z*L1**4) - 1),
        ((1.93*D2**3)/(z*Z*L2**4) - 1),
        (sqrt(((745*D1)/(z*Z))**2+16.9e+06)/(110*L1**3) - 1),
        (sqrt(((745*D2)/(z*Z))**2+157.5e+06)/(85*L2**3) - 1),
        ((z*Z)/(40) - 1),
        ((5*z)/(b) - 1),
        ((b)/(12*z) - 1),
        ((1.5*L1+1.9)/(D1) - 1),
        ((1.1*L2+1.9)/(D2) - 1)
    ]


def reducer_minimize(x, *args):
    b, z, D1, D2, L1, L2, Z = x
    Z = round(Z)

    return (0.7854*b*z**2
                *(3.3333*Z**3 + 14.9334*Z - 43.0934) 
                - 1.508*b*(L1**2 + L2**2) + 7.4777*(L1**3+L2**3) 
                + 0.7854*(D1*L1**2 + D2*L2**2))

reducer_args = ()
reducer_bounds = [[3.5, 0.7, 7.3, 7.3, 2.9, 5, 17], [3.6, 0.72, 8.3, 8.3, 3.9, 5.5, 28]]
reducer_out = []

def reducer_print(x, message):
    [(b, z, D1, D2, L1, L2, Z), fx] = x
    print(message, "f(x) = ", format(round(fx, 3), '.3f'), "| b = ", format(round(b, 3), '.3f'),
        "| z = ", format(round(z, 3), '.3f'), "| Z = ", round(Z),
        "| D1 = ", format(round(D1, 3), '.3f'), "| D2 = ", format(round(D2, 3), '.3f'),
        "| L1 = ", format(round(L1, 3), '.3f'), "| L2 = ", format(round(L2, 3), '.3f'))

'''
if __name__ == "__main__":
    for i in range(20):
        reducer_out.append(
            pso(reducer_minimize, reducer_bounds, fcons=reducer_cons,
                args=reducer_args, swarmsize=50, maxiter=100, error=1e-08,
                wmax=0.6, wmin=1.4, c1=0.5, c2=1.5))

    print("Speed reducer (top 5 out of 20): ")
    for row in sorted(filter(None, reducer_out), key=itemgetter(1))[:5]:
        reducer_print(row, "")


x1=b
x2=z
x3=Z
x4=D1
x5=D2
x6=L1
x7=L2

if __name__ == "__main__":
    b = 3.5
    z = 0.7
    D1 = 7.3
    D2 = 7.72
    L1 = 3.35
    L2 = 5.29
    Z = 17

    print("PDF:", (0.7854*b*z**2)
            *(3.3333*Z**3 + 14.9334*Z - 43.0934) 
            - (1.508*b*(L1**2 + L2**2) + 7.4777*(L1**3+L2**3)) 
            + (0.7854*(D1*L1**2 + D2*L2**2)))

    print("Task:", (0.7854*b*z**2)
            *(3.3333*Z**3 + 14.9334*Z - 43.0934) 
            - (1.508*b*(D1**2 + D2**2) + 7.4777*(D1**3+D2**3)) 
            + (0.7854*(L1*D1**2 + L2*D2**2)))

'''
        
from multiprocessing import Process, Manager
def parallel(reducer_out2, coil_out2, each_thread):  # the managed list `L` passed explicitly.
    for i in range(each_thread):
        reducer_out2.append(pso(reducer_minimize, reducer_bounds, fcons=reducer_cons,
            args=reducer_args, swarmsize=50, maxiter=100, error=1e-08,
            wmax=1.4, wmin=0.6, c1=0.5, c2=1.5))
        coil_out2.append(
            pso(coil_minimize, coil_bounds, fcons=coil_cons,
                args=coil_args, swarmsize=20, maxiter=50, error=1e-08,
                wmax=1.4, wmin=0.6, c1=0.5, c2=1.5))

if __name__ == "__main__":
    with Manager() as manager:
        reducer_out2 = manager.list()
        coil_out2 = manager.list()
        processes = []
        
        rang = 500
        threads = 12
        each_thread = floor(rang/threads)
        for i in range(threads):
            p = Process(target=parallel, args=(reducer_out2, coil_out2, each_thread))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        print(f"Closed coil helical spring (top 5 out of {len(list(filter(None, coil_out2)))}/{len(coil_out2)}): ")
        coil_print([(0.599, 1.924, 0), 42.099], "Expected:  ")
        for row in sorted(filter(None, coil_out2), key=itemgetter(1))[:5]:
            coil_print(row, "Calculated:")

        print(f"Speed reducer (top 5 out of {len(list(filter(None, reducer_out2)))}/{len(reducer_out2)}): ")
        reducer_print([(3.5, 0.7, 7.3, 7.72, 3.35, 5.29, 17), 3000], "Expected:   ")
        for row in sorted(filter(None, reducer_out2), key=itemgetter(1))[:5]:
            reducer_print(row, "Calculated:")
            
'''
Closed coil helical spring (top 5 out of 4588/4992):
Expected:   f(x) =  42.099 | d =  0.599 | D =  1.924 | N =  0
Calculated: f(x) =  42.873 | d =  0.741 | D =  3.517 | N =  7
Calculated: f(x) =  42.916 | d =  0.741 | D =  3.519 | N =  7
Calculated: f(x) =  42.918 | d =  0.741 | D =  3.520 | N =  7
Calculated: f(x) =  42.919 | d =  0.741 | D =  3.519 | N =  7
Calculated: f(x) =  42.931 | d =  0.741 | D =  3.520 | N =  7
Speed reducer (top 5 out of 4566/4992):
Expected:    f(x) =  3000.000 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.300 | D2 =  7.720 | L1 =  3.350 | L2 =  5.290
Calculated: f(x) =  23755.734 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.300 | D2 =  7.721 | L1 =  3.351 | L2 =  5.287
Calculated: f(x) =  23755.870 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.326 | D2 =  7.722 | L1 =  3.350 | L2 =  5.287
Calculated: f(x) =  23756.245 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.300 | D2 =  7.734 | L1 =  3.351 | L2 =  5.287
Calculated: f(x) =  23756.314 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.300 | D2 =  7.720 | L1 =  3.352 | L2 =  5.287
Calculated: f(x) =  23756.343 | b =  3.500 | z =  0.700 | Z =  17 | D1 =  7.300 | D2 =  7.735 | L1 =  3.351 | L2 =  5.287
'''