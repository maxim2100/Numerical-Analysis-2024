# https://github.com/maxim2100/Matrix-Analysis-Toolkit.git

# maxim teslenko 321916116
# rom ihia 207384934
# Rony Bubnovsky 314808825
# Bar Levi 314669664 
# Aviel Esperansa 324062116

import numpy as np


def get_input(choose):
    list_of_points=[]
    loop=3
    if choose==1:
        loop=2
    for i in range(loop):
        x = float(input("Enter X value: "))
        y = float(input("Enter Y value: "))
        list_of_points.append((x, y))
    return list_of_points

def linear_interpolation(points,x):
    x0,x1,y0,y1 = points[0][0],points[1][0],points[0][1],points[1][1]
    m = (y1 - y0) / (x1 - x0)
    if x1 - x0 == 0:  # Avoid division by zero
        raise ValueError("The two points have the same x value, resulting in a vertical line, which cannot be represented as a linear equation in the form y = mx + b.")

    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m * x + b
    

def polynomial_interpolation(points, x):
    # נשתמש בפולינום עבור 3 נקודות: f(x) = ax^2 + bx + c
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    
    # נפתור את מערכת המשוואות למציאת a, b, c
    A = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]])
    B = np.array([y0, y1, y2])
    coeffs = np.linalg.solve(A, B)
    
    a, b, c = coeffs
    return a*x**2 + b*x + c

def lagrange_interpolation(points, x):
    # נשתמש בפולינום לגראנז' עבור 3 נקודות.
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    print (x0, y0, x1, y1, x2, y2 ,x)
    L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    print (L0, L1,L2)
    print (y0 * L0 + y1 * L1 + y2 * L2)
    return y0 * L0 + y1 * L1 + y2 * L2


def menu():
    points=[]
    while True:
        print("\nChoose the following interpolation functions")
        print("1. Linear interpolation")
        print("2. Polynomial interpolation")
        print("3. Lagrange interpolation")
        print("4. Exit")
        choice = input("Choose an option: ")
        points = get_input(int(choice))
        x = float(input("Enter the x value for which you want to find the interpolated y value: "))
        if choice == '1':
            y = linear_interpolation(points, x)
        elif choice == '2':
            y = polynomial_interpolation(points, x)
        elif choice == '3':
            y = lagrange_interpolation(points, x)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
        if y is not None:
            print(f"The interpolated value at x = {x} is y = {y}")
        else:
            print("The value is out of the interpolation range.")

if __name__ == "__main__":
    menu()