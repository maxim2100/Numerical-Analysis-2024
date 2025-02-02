from Iterative_methods.newtonRaphson import newton_raphson 
from Iterative_methods.bisection_method import bisection_method, find_root_intervals
from Methods_for_Solving_Linear_Systems_of_Equations.Jacobi_and_Gauss_Seidel import gauss_seidel
from Matrix.matrix_tool_2 import find_L_matrix,print_matrix,inverse_matrix,solve_system_with_LU,multiply_matrix_vector
from Numerical_Integration_methods.Simpson_method import simpsons_rule
from Numerical_Integration_methods.Trapezoidal_method import trapezoidal_rule
from src.machine_precision.machine_precision import machine_epsilon
from src.matrices.matrix_operations import MatrixOperations
from Interpolation_and_Polynomial_Approximation.linear_interpolation import linearInterpolation
import numpy as np
import math

def GaussianElimination(A,b,n):
    A_orig = MatrixOperations.copy_matrix(A)

    # 1) מפעילים את האלגוריתם
    steps, invertible = MatrixOperations.gauss_jordan_with_elementary_matrices(A)

    # 2) מדפיסים שלבים (המטריצות האלמנטריות)
    print( "===== פעולות אלמנטריות =====\n")
    for idx, (E, desc) in enumerate(steps, start=1):
        print( f"\n--- שלב {idx}: {desc} ---\n")
        print( "מטריצה אלמנטרית E:\n")
        print( MatrixOperations.matrix_to_string(E) + "\n")

    # 3) אם A אינה הפיכה
    if not invertible:
        print( "\nA אינה הפיכה (דטרמיננטה=0).\n")
        return

    # 4) חושבים על A^-1 כמכפלת כל המטריצות האלמנטריות (E_k ... E_1)
    A_inv = MatrixOperations.build_identity(n)
    for (E, _) in steps:
        A_inv = MatrixOperations.multiply_matrices(E, A_inv)

    print( "\n===== A^-1 (המטריצה ההופכית) =====\n")
    print( MatrixOperations.matrix_to_string(A_inv) + "\n")

    # 5) בדיקות: A*A^-1 ו-A^-1*A
    prod1 = MatrixOperations.multiply_matrices(A_orig, A_inv)
    prod2 = MatrixOperations.multiply_matrices(A_inv, A_orig)

    print( "\nבדיקה: A * A^-1:\n")
    print( MatrixOperations.matrix_to_string(prod1) + "\n")
    print( f"האם זו I? {MatrixOperations.is_identity(prod1)}\n")

    print( "\nבדיקה: A^-1 * A:\n")
    print( MatrixOperations.matrix_to_string(prod2) + "\n")
    print( f"האם זו I? {MatrixOperations.is_identity(prod2)}\n")

    # 6) פתרון Ax = b => x = A^-1 * b
    x = MatrixOperations.multiply_matrix_vector(A_inv, b)

    print( "\n===== פתרון המערכת Ax = b =====\n")
    print( "x = A^-1 * b:\n")
    for i, val in enumerate(x):
        print( f"x[{i}] = {val:.5f}\n")

    # בודקים A*x ~ b
    Ax = MatrixOperations.multiply_matrix_vector(A_orig, x)
    print( "\nבדיקה: A*x (צריך להיות קרוב ל-b)\n")
    print( f"A*x = {Ax}\n")
    print( f"b    = {b}\n")
    close_check = MatrixOperations.is_close_to_vector(Ax, b)
    print( f"האם קרוב? {close_check}\n")

    if close_check is None:
        print("No unique solution", "The system might be singular or have infinite solutions.")

def LU(A,b):
    lmatrix, umatrix = find_L_matrix(matrix)
    print("L matrix:","\n")
    print_matrix(lmatrix)
    print("\n","U matrix:","\n")
    print_matrix(umatrix)
    invLMatrix = inverse_matrix(lmatrix)
    print("\n","Inverse L matrix:","\n")
    print_matrix(invLMatrix)
    invUMatrix = inverse_matrix(umatrix)
    print("\n","Inverse U matrix:","\n") 
    print_matrix(invUMatrix)
    ans = solve_system_with_LU(matrix, b)
    print("x vector:", ans,"\n")
    print("Check answer:", multiply_matrix_vector(np.array(matrix), np.array(ans)),"\n")
    print("LU [x,y,z]:", ans,"\n")

if __name__ == '__main__':
    # q11
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
    # q21

    matrix=[[1,0.5,0.333],[0.5,0.333,0.25],[0.333,0.25,0.2]]
    A = [[1,0.5,0.333],[0.5,0.333,0.25],[0.333,0.25,0.2]]
    b = [1,0,0]

    GaussianElimination(A,b,3)
    # LU(matrix,b)
    print("Gauss Seidel [x,y,z]:",gauss_seidel(matrix,b))
    # q10

    a, b = -0.5, 0.5
    n = 100  # מספר חיתוכים (זוגי)
    def g(x):
        denominator = (2 * x) ** 3 + 5 * x ** 2 - 6
        if denominator == 0:
            return 0  # מניעת חלוקה באפס
        return math.sin(2 * math.e ** (-2 * x)) / denominator
    result = simpsons_rule(g, a, b, n)
    print(f"Approximate integral using Simpson's rule: {result}")
    result = trapezoidal_rule(g, a, b, n)
    print("Approximate integral:", result)




    # q25

    matrix=[[2,1,0],[3,-1,0],[1,4,-2]]
    A = [[2,1,0],[3,-1,0],[1,4,-2]]
    b = [-3,1,-5]
    GaussianElimination(A,b,3)
    LU(matrix,b)


    # q3
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


    # q38
    table_points = [(6.5, 2.14451), (6.7, 2.35585), (7.0, 2.74748), (8.0, 5.67127)]
    x = 6.9
    print( "----------------- Interpolation & Extrapolation Methods -----------------\n")
    print( "Table Points: ", table_points)
    print( "Finding an approximation to the point: ",  x)
    linearInterpolation(table_points, x)
    print( "\n---------------------------------------------------------------------------\n")
    


