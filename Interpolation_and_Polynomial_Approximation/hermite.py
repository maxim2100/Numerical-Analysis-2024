# https://github.com/maxim2100/Hermite-Interpolation


import sympy as sp
from sympy.utilities.lambdify import lambdify

x = sp.symbols('x')

def Lagrange_i(data_points, i):
    """Returns the i-th Lagrange basis polynomial."""
    xi = data_points[i][0]
    return sp.prod((x - data_points[j][0]) / (xi - data_points[j][0])
                   for j in range(len(data_points)) if j != i)

def Lagrange_squared(data_points, i):
    """Returns the squared i-th Lagrange basis polynomial."""
    return Lagrange_i(data_points, i)**2

def Lagrange_Derivative(data_points, i):
    """Returns the derivative of the i-th Lagrange basis polynomial."""
    return Lagrange_i(data_points, i).diff(x)

def Hermite_Interpolation(data_points):
    """Calculates the Hermite polynomial and its derivative."""
    hermite_function = 0
    for i in range(len(data_points)):
        xi, yi, mi = data_points[i]
        la_squared = Lagrange_squared(data_points, i)
        la_derivative = Lagrange_Derivative(data_points, i)
        hermite_function += ((1 - 2 * (x - xi) * la_derivative.subs(x, xi)) * la_squared * yi +
                             (x - xi) * la_squared * mi)

    h = sp.simplify(hermite_function)
    dh = sp.simplify(h.diff(x))

    return lambdify(x, h), lambdify(x, dh), h, dh

def calculate_error_term_at_x(value, data_points, h, higher_derivative):
    """Calculates the error term based on the Hermite interpolation error formula at a given x."""
    n = len(data_points) - 1  # Degree of interpolation
    product_term = sp.prod((value - data_points[j][0])**2 for j in range(len(data_points)))
    error_value = abs(higher_derivative(value)) / sp.factorial(2 * (n + 1)) * product_term
    return error_value

def display_results(data_points, h, dh, simplified_h, simplified_dh):
    """Displays the Hermite polynomial, its derivative, and errors in a clear format."""
    print("\n===== Hermite Polynomial Results =====")
    print(f"\nSimplified Hermite Polynomial H(x):\n{simplified_h}")
    print(f"\nSimplified Derivative H'(x):\n{simplified_dh}")
    print("\n=====================================")

    print("\nPoint-wise Evaluation of H(x) and H'(x):")
    for xi, yi, mi in data_points:
        print(f"x = {xi}, f(x) = {yi}, f'(x) = {mi}, H(x) = {h(xi):.5f}, H'(x) = {dh(xi):.5f}")

    # print("\n===== Errors =====")
    # for xi, error_f, error_f_prime in errors:
    #     print(f"x = {xi}, |H(x) - f(x)| = {error_f:.5f}, |H'(x) - f'(x)| = {error_f_prime:.5f}")
    
    # print("\n===== Error Term Calculations =====")
    # for xi, error_term in error_terms:
    #     print(f"x = {xi}, Error Term = {error_term:.5f}")
    print("=====================================")

def Drive():
    """Collects user input and computes H(x), H'(x), errors, and error terms."""
    num_points = int(input("How many points do you want to enter? "))
    data_points = [(
        float(input(f"Enter xi for point {i + 1}: ")),
        float(input(f"Enter f(xi) for point {i + 1}: ")),
        float(input(f"Enter f'(xi) for point {i + 1}: "))
    ) for i in range(num_points)]
    # ln  data_points = [(1, 0, 1), (5, 1.6094377, 0.2), (9, 2.197224577, 0.111111111)]
    # data_points = [(1, 0, 1), (2, 0.693147181, 0.5)]
    h, dh, simplified_h, simplified_dh = Hermite_Interpolation(data_points)
    
    # Display the results
    display_results(data_points, h, dh, simplified_h, simplified_dh)

    # Allow the user to evaluate the polynomial until they type "stop"
    while True:
        user_input = input("\nEnter a value to evaluate H(x) or type 'stop' to end: ")
        if user_input.lower() == "stop":
            break
        try:
            value = float(user_input)
            result = h(value)
            derivative_result = dh(value)
            print(f"H({value}) = {result:.5f}, H'({value}) = {derivative_result:.5f}")

            # Ask for the higher-order derivative function
            higher_derivative_str = str(simplified_dh)
            higher_derivative_input = higher_derivative_str.split(" ")[0]
            higher_derivative_expr = sp.sympify(higher_derivative_input)
            higher_derivative = lambdify(x, higher_derivative_expr)

            # Calculate the error term using the higher-order derivative
            error_term = calculate_error_term_at_x(value, data_points, h, higher_derivative)
            print(f"Error term at x = {value} is approximately: {error_term:.5f}")

        except ValueError:
            print("Please enter a valid number or 'stop'.")


# Run the Drive function to start
Drive()