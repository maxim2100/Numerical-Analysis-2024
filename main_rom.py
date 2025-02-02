from Iterative_methods.newtonRaphson import newton_raphson 
from Iterative_methods.bisection_method import bisection_method, find_root_intervals
import math

if __name__ == '__main__':
    def f(x):
        return math.cos(x**2 + 5*x + 6) / (2 * math.exp(-x))

    def df(x):
        numerator = -math.sin(x**2 + 5*x + 6) * (2*x + 5)
        denominator = 2 * math.exp(-x)
        return numerator / denominator

    p0 = -1.5  
    TOL = 1e-10
    N = 100
    root_newton = newton_raphson(f, df, p0, TOL, N)

    x_start, x_end = -1.5, 2
    step = 0.1  
    tol = 1e-6

    root_intervals = find_root_intervals(f, x_start, x_end, step)

    if not root_intervals:
        print("No root intervals found in the given range.")
    else:
        print(f"\nRoot intervals found: {root_intervals}\n")
        roots_bisection = []
        for a, b in root_intervals:
            root = bisection_method(f, a, b, tol)
            roots_bisection.append(root)

        print(f"\nFinal approximated roots: {roots_bisection}")

    smallest_bisection_root = min(roots_bisection) if roots_bisection else None
    smallest_newton_root = root_newton

    if smallest_bisection_root is not None and smallest_newton_root is not None:
        smallest_root = min(smallest_bisection_root, smallest_newton_root)
    elif smallest_bisection_root is not None:
        smallest_root = smallest_bisection_root
    elif smallest_newton_root is not None:
        smallest_root = smallest_newton_root
    else:
        smallest_root = None

    print(f"\nSmallest root found using Bisection Method: {smallest_bisection_root}")
    print(f"Smallest root found using Newton-Raphson Method: {smallest_newton_root}")
    
    if smallest_root is not None:
        print(f"\nThe smallest root found overall: {smallest_root}")
    else:
        print("\nNo roots found in any method.")
