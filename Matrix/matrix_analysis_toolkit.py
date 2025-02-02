UNIT_MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
VECTOR_ZIRO = [0, 0, 0]
MATRIX_ZIRO = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def vector_1_norm(vector):
    """Calculate the 1-norm of a 3x1 vector."""
    return sum(abs(v) for v in vector)

def vector_inf_norm(vector):
    """Calculate the infinity-norm of a 3x1 vector."""
    return max(abs(v) for v in vector)

def one_matrix_norm(matrix):
    """Calculate the 1-norm of a 3x3 matrix."""
    return max(sum(abs(matrix[i][j]) for i in range(3)) for j in range(3))

def infinty_matrix_norm(matrix):
    """Calculate the infinity-norm of a 3x3 matrix."""
    return max(sum(abs(matrix[i][j]) for j in range(3)) for i in range(3))

def condA(matrix):
    """Calculate the condition number of the matrix."""
    matrix_inv = inverse_matrix(matrix)
    norm_A = infinty_matrix_norm(matrix)
    norm_A_inv = infinty_matrix_norm(matrix_inv)
    cond_A = norm_A * norm_A_inv
    return cond_A

def equal_matrix(A, B):
    """Check if two 3x3 matrices A and B are equal."""
    for i in range(3):
        for j in range(3):
            if A[i][j]!= B[i][j]:
                return False
    return True

def swap_row(matrix, i, j):
    """Perform an elementary row operation to swap two rows of a 3x3 matrix."""
    for k in range(3):
        temp = matrix[i][k]
        matrix[i][k] = matrix[j][k]
        matrix[j][k] = temp
    return matrix

def pivot_all(matrix):
    """Pivot a 3x3 matrix and return the pivoted matrix and the elementary matrix used."""
    a = MATRIX_ZIRO
    l = VECTOR_ZIRO
    element_matrix = UNIT_MATRIX
    for i in range(3):
        for j in range(3):
            col = 0
            if i+j >= 3:
                col=i+j-3
            else:
                col=j+i
            a[i][j] = matrix[j][col]
    
    for i in range(3):
        l[i] = vector_1_norm(a[i])
        if 0 in a[i]:
            l[i] = -1
    max_index = l.index(max(l))
    if max_index== 0:
        return matrix
    elif max_index == 1:
        matrix=swap_row(matrix,2,0)
        element_matrix=swap_row(element_matrix,2,0)
        matrix=swap_row(matrix,1,2)
        element_matrix=swap_row(element_matrix,1,2)
    elif max_index == 2:
        matrix=swap_row(matrix,0,1)
        element_matrix=swap_row(element_matrix,0,1)
        matrix=swap_row(matrix,1,2)
        element_matrix=swap_row(element_matrix,1,2)
    return matrix,element_matrix

def copy_matrix(matrix):
    """Create a copy of a 3x3 matrix."""
    copy = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            copy[i][j] = matrix[i][j]
    return copy

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

def vector_multiplication(a, b):
    """Multiply two vectors(a 1x3 * b 3x1) and return the result as a number."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Both vectors must be of length 3.")
    for i in range(3):
        result += [a[i] * b[i]]
    return result

def matrix_and_vector_multiplication(A, x):
    """Multiply matrix A 3x3 by vector x 3x1."""
    result = [0, 0, 0]
    result1 = [0, 0, 0]
    for i in range(3):
        result1[i] = vector_multiplication(A[i], x)

    for i in range(3):
        for j in range(3):
            result[i] += A[i][j] * x[j]

    if result==result1:
        print(f"The result is {result}")

    return result

def matrix_multiplication(A, B):
    """Multiply two 3x3 matrices A and B."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result

def print_matrix(matrix):
    """Print the matrix in a readable format."""
    for row in matrix:
        print(" ".join(f"{elem:.4f}" for elem in row))

def input_vector():
    """Get a 3x1 vector from user input."""
    print("Enter the elements of the 3x1 vector:")
    vector = VECTOR_ZIRO
    for i in range(3):
        vector = float(input(f"Enter element {i+1}: "))
    return vector

def input_matrix():
    """Get a 3x3 matrix from user input."""
    input_matrix = []
    print("Enter a 3x3 matrix input by rows")
    for i in range(3):
        print(f"Enter row {i+1}: ")
        row = []
        for j in range(3):
            row.append(float(input(f"Enter element {j+1}: ")))
        input_matrix.append(row)
    return input_matrix
    
def menu():
    """Menu for Matrix Analysis Toolkit."""
    data = {}
    data['A'] = [[0,1,2],[3,124,2],[2,1,23]]
    while True:
        print("\nMatrix Analysis Toolkit")
        print("1. Enter a new matrix")
        print("2. Display save matrix")
        print("3. Find inverse of matrix")
        print("4. Calculate norm of matrix")
        print("5. Calculate norm of inverse matrix")
        print("6. Calculate condition number of matrix")
        print("7. Print all matrices")
        print("8. Exit")
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
                norm_A = infinty_matrix_norm(data[name])
                print(f"Norm of matrix {name}: {norm_A:.4f}")
            else:
                print("No matrix found. Please enter a matrix first.")
    
        elif choice == '5':
            name = input("Enter matrix name: ")
            if name in data.keys():
                try:
                    norm_A_inv = infinty_matrix_norm(inverse_matrix(data[name]))
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
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
