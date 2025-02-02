import numpy as np
# https://github.com/maxim2100/Matrix-Analysis-Toolkit.git

# maxim teslenko 321916116
# rom ihia 207384934
# Rony Bubnovsky 314808825
# Bar Levi 314669664 
# Aviel Esperansa 324062116


UNIT_MATRIX = np.eye(3)
VECTOR_ZERO = np.zeros(3)
MATRIX_ZERO = np.zeros((3, 3))

def vector_1_norm(vector):
    """Calculate the 1-norm of a 3x1 vector."""
    return np.sum(np.abs(vector))

def vector_inf_norm(vector):
    """Calculate the infinity-norm of a 3x1 vector."""
    return np.max(np.abs(vector))

def one_matrix_norm(matrix):
    """Calculate the 1-norm of a 3x3 matrix."""
    return np.max(np.sum(np.abs(matrix), axis=0))

def infinity_matrix_norm(matrix):
    """Calculate the infinity-norm of a 3x3 matrix."""
    return np.max(np.sum(np.abs(matrix), axis=1))

def condA(matrix):
    """Calculate the condition number of the matrix."""
    matrix_inv = np.linalg.inv(matrix)
    norm_A = infinity_matrix_norm(matrix)
    norm_A_inv = infinity_matrix_norm(matrix_inv)
    cond_A = norm_A * norm_A_inv
    return cond_A

def equal_matrix(A, B):
    """Check if two 3x3 matrices A and B are equal."""
    return np.array_equal(A, B)

def swap_row(matrix, i, j):
    """Perform an elementary row operation to swap two rows of a 3x3 matrix."""
    matrix[[i, j]] = matrix[[j, i]]
    return matrix

def pivot_all(matrix):
    """Pivot a 3x3 matrix and return the pivoted matrix and the elementary matrix used."""
    a = MATRIX_ZERO.copy()
    l = VECTOR_ZERO.copy()
    element_matrix = UNIT_MATRIX.copy()
    for i in range(3):
        for j in range(3):
            col = (i + j) % 3
            a[i][j] = matrix[j][col]
    
    for i in range(3):
        l[i] = vector_1_norm(a[i])
        if 0 in a[i]:
            l[i] = -1
    max_index = np.argmax(l)
    if max_index == 0:
        return matrix
    elif max_index == 1:
        matrix = swap_row(matrix, 2, 0)
        element_matrix = swap_row(element_matrix, 2, 0)
        matrix = swap_row(matrix, 1, 2)
        element_matrix = swap_row(element_matrix, 1, 2)
    elif max_index == 2:
        matrix = swap_row(matrix, 0, 1)
        element_matrix = swap_row(element_matrix, 0, 1)
        matrix = swap_row(matrix, 1, 2)
        element_matrix = swap_row(element_matrix, 1, 2)
    return matrix, element_matrix

def copy_matrix(matrix):
    """Create a copy of a 3x3 matrix."""
    return np.copy(matrix)

def inverse_matrix(matrix):
    """Compute the inverse of a 3x3 matrix."""
    inverse_matrix = copy_matrix(UNIT_MATRIX)
    matrix = copy_matrix(matrix)

    for i in range(3):
        if matrix[i][i] == 0:
            for k in range(i + 1, 3):
                if matrix[k][i] != 0:
                    swap_row(matrix, i, k)
                    swap_row(inverse_matrix, i, k)
                    break
            if matrix[i][i] == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")

        diag_element = matrix[i][i]
        for j in range(3):
            matrix[i][j] /= diag_element
            inverse_matrix[i][j] /= diag_element

        for k in range(3):
            if k != i:
                factor = matrix[k][i]
                for j in range(3):
                    matrix[k][j] -= factor * matrix[i][j]
                    inverse_matrix[k][j] -= factor * inverse_matrix[i][j]

    return inverse_matrix


def find_L_matrix(matrix):
    """Find the lower and upper triangular matrices L and U from the matrix A."""
    umatrix = copy_matrix(matrix).astype(float)
    lmatrix = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            factor = umatrix[j][i] / umatrix[i][i]
            umatrix[j] -= factor * umatrix[i]
            lmatrix[j][i] = factor
    return lmatrix, umatrix

def multiply_matrix_vector(matrix, vector):
    # Ensure the input matrix is 3x3 and the vector is 3x1
    if matrix.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("Matrix must be 3x3 and vector must be 3x1.")
    
    # Perform matrix-vector multiplication
    result = np.dot(matrix, vector)
    
    return result

def matrix_multiplication(A, B):
    """Multiply two 3x3 matrices A and B."""
    return np.dot(A, B)

def print_matrix(matrix):
    """Print the matrix in a readable format."""
    print("\n".join(" ".join(f"{elem:.4f}" for elem in row) for row in matrix))

def input_vector():
    """Get a 3x1 vector from user input."""
    print("Enter the elements of the 3x1 vector:")
    vector = np.zeros(3)
    for i in range(3):
        vector[i] = float(input(f"Enter element {i+1}: "))
    return vector

def input_matrix():
    """Get a 3x3 matrix from user input."""
    input_matrix = []
    print("Enter a 3x3 matrix input by rows")
    for i in range(3):
        row = []
        for j in range(3):
            row.append(float(input(f"Enter element {j+1}: ")))
        input_matrix.append(row)
    return np.array(input_matrix)

def solve_system_with_LU(A, b):
    """Solve the system of equations Ax = b using LU decomposition."""
    lmatrix, umatrix = find_L_matrix(A)
    y = np.linalg.solve(lmatrix, b)
    x = np.linalg.solve(umatrix, y)
    return x

def menu():
    """Menu for Matrix Analysis Toolkit."""
    data = {}
    data['A'] = np.array([[2,1,0],[3,-1,0],[1,4,-2]])
    print("the matrix that all ready saved in the database:\n")
    print_matrix(data['A'])
    while True:
        print("\nMatrix Analysis Toolkit")
        print("1. Enter a new matrix")
        print("2. Display saved matrix")
        print("3. Find inverse of matrix")
        print("4. Calculate norm of matrix")
        print("5. Calculate norm of inverse matrix")
        print("6. Calculate condition number of matrix")
        print("7. Print all matrices")
        print("8. LU Decomposition and Solve System")
        print("9. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            while True:
                name = input("Enter matrix name: ")
                if name in data.keys():
                    print("A matrix with that name already exists. Please enter a new name.")
                else:
                    break
            matrix = input_matrix()
            data[name] = matrix
            print("Matrix entered successfully.")

        elif choice == '2':
            name = input("Enter matrix name: ")
            if name in data.keys():
                print(f"Matrix {name}:")
                print_matrix(data[name])
            else:
                print("No matrix found. Please enter a matrix first.")

        elif choice == '3':
            name = input("Enter matrix name: ")
            if name in data.keys():
                matrix_inv = inverse_matrix(data[name])
                print(f"Matrix {name} inverse: ")
                print_matrix(matrix_inv)
                if input("Add inverse to data(Y/N): ") == 'Y':
                    while True:
                        name = input("Enter name for inverse matrix: ")
                        if name in data.keys():
                            print("A matrix with that name already exists. Please enter a new name.")
                        else:
                            data[name] = matrix_inv
                            break
            else:
                print("No matrix found. Please enter a matrix first.")

        elif choice == '4':
            name = input("Enter matrix name: ")
            if name in data.keys():
                norm_A = infinity_matrix_norm(data[name])
                print(f"Norm of matrix {name}: {norm_A:.4f}")
            else:
                print("No matrix found. Please enter a matrix first.")
    
        elif choice == '5':
            name = input("Enter matrix name: ")
            if name in data.keys():
                try:
                    norm_A_inv = infinity_matrix_norm(inverse_matrix(data[name]))
                    print(f"Infinity norm of inverse matrix {name}: {norm_A_inv:.4f}")
                except ValueError as e:
                    print(e)
            else:
                print("No matrix found. Please enter a matrix first.")

        elif choice == '6':
            name = input("Enter matrix name: ")
            if name in data.keys():
                try:
                    cond_A = condA(data[name])
                    print(f"Condition number of matrix {name}: {cond_A:.4f}")
                except ValueError as e:
                    print(e)
            else:
                print("No matrix found. Please enter a matrix first.")

        elif choice == '7':
            if data:
                for name, matrix in data.items():
                    print(f"\nMatrix {name}:")
                    print_matrix(matrix)
            else:
                print("No matrices saved.")

        elif choice == '8':
            if 'A' in data:
                lmatrix, umatrix = find_L_matrix(data['A'])
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
                vector = np.array([-3,1,-5])
                print("Chosen vector b:", vector,"\n")
                ans = solve_system_with_LU(data['A'], vector)
                print("x vector:", ans,"\n")
                print("Check answer:", multiply_matrix_vector(data['A'], ans),"\n")
            else:
                print("Matrix A not found. Please enter matrix A first.")

        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()

