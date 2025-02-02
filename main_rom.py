from Iterative_methods.newtonRaphson import newton_raphson 
from Iterative_methods.bisection_method import bisection_method ,find_root_intervals
import math



if __name__ == '__main__':
    f = lambda x: (x * math.exp(-x) + math.log(x**2)) * (2*x**3 + 2*x**2 - 3*x - 5)
    df = lambda x: ((math.exp(-x) - x * math.exp(-x) + 2/x) * (2*x**3 + 2*x**2 - 3*x - 5) +
                     (x * math.exp(-x) + math.log(x**2)) * (6*x**2 + 4*x - 3))
    p0 = 1.5  
    TOL = 1e-10
    N = 100
    roots = newton_raphson(f, df,p0,TOL,N)
    
    def f(x):
        if x <= 0:
            return float('inf')
        return (x * math.exp(-x) + math.log(x**2)) * (2*x**3 + 2*x**2 - 3*x - 5)

    x_start, x_end = 0, 1.5  
    step = 0.1  
    tol = 1e-6

    root_intervals = find_root_intervals(f, x_start, x_end, step)

    if not root_intervals:
        print("No root intervals found in the given range.")
    else:
        print(f"\nRoot intervals found: {root_intervals}\n")
        roots = []
        for a, b in root_intervals:
            root = bisection_method(f, a, b, tol)
            roots.append(root)

        print(f"\nFinal approximated roots: {roots}")


