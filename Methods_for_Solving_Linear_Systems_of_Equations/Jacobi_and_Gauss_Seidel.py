# https://github.com/maxim2100/Matrix-Analysis-Toolkit.git

# maxim teslenko 321916116
# rom ihia 207384934
# Rony Bubnovsky 314808825
# Bar Levi 314669664 
# Aviel Esperansa 324062116

import numpy as np

UNIT_MATRIX = np.eye(3)
VECTOR_ZERO = np.zeros(3)
MATRIX_ZERO = np.zeros((3, 3))
PRECISION = 0.0001

def jacobi(matrix, b):
    if not dominant_matrix(matrix):
        print("The matrix have not dominant diagonal!!!")
        return
    varx=[]
    vary=[]
    varz=[]
    varx.append(0)
    vary.append(0)
    varz.append(0)
    i = 0
    print(f"X{i}= {varx[i]}")
    print(f"Y{i}= {vary[i]}")
    print(f"Z{i}= {varz[i]}")
    while True:
        # print(f"X{i+1} =({b[0]} - {matrix[0][1]}*Y{i} - {matrix[0][2]}*Z{i})/{matrix[0][0]} = {(b[0]-vary[i]*matrix[0][1]-varz[i]*matrix[0][2])/matrix[0][0]}")
        # print(f"Y{i+1} =({b[1]} - {matrix[1][0]}*X{i} - {matrix[1][2]}*Z{i})/{matrix[1][1]} = {(b[1]-varx[i]*matrix[1][0]-varz[i]*matrix[1][2])/matrix[1][1]}")
        # print(f"Z{i+1} =({b[2]} - {matrix[2][0]}*X{i} - {matrix[2][1]}*Y{i})/{matrix[2][2]} = {(b[2]-varx[i]*matrix[2][0]-vary[i]*matrix[2][1])/matrix[2][2]}")
        varx.append((b[0]-vary[i]*matrix[0][1]-varz[i]*matrix[0][2])/matrix[0][0])
        vary.append((b[1]-varx[i]*matrix[1][0]-varz[i]*matrix[1][2])/matrix[1][1])
        varz.append((b[2]-varx[i]*matrix[2][0]-vary[i]*matrix[2][1])/matrix[2][2])
        i+=1
        if abs(varx[i]-varx[i-1])<PRECISION and abs(vary[i]-vary[i-1])<PRECISION and abs(varz[i]-varz[i-1])<PRECISION:
            return [varx[i],vary[i],varz[i]]

def gauss_seidel(matrix,b):
    if not dominant_matrix(matrix):
        print("The matrix have not dominant diagonal!!!")
        return
    varx=[]
    vary=[]
    varz=[]
    varx.append(0)
    vary.append(0)
    varz.append(0)
    i = 0
    print(f"X{i}= {varx[i]}")
    print(f"Y{i}= {vary[i]}")
    print(f"Z{i}= {varz[i]}")
    while True:
        varx.append((b[0]-vary[i]*matrix[0][1]-varz[i]*matrix[0][2])/matrix[0][0])
        vary.append((b[1]-varx[i+1]*matrix[1][0]-varz[i]*matrix[1][2])/matrix[1][1])
        varz.append((b[2]-varx[i+1]*matrix[2][0]-vary[i+1]*matrix[2][1])/matrix[2][2])
        # print(f"X{i+1} =({b[0]} - {matrix[0][1]}*Y{i} - {matrix[0][2]}*Z{i})/{matrix[0][0]} = {(b[0]-vary[i]*matrix[0][1]-varz[i]*matrix[0][2])/matrix[0][0]}")
        # print(f"Y{i+1} =({b[1]} - {matrix[1][0]}*X{i+1} - {matrix[1][2]}*Z{i})/{matrix[1][1]} = {(b[1]-varx[i+1]*matrix[1][0]-varz[i]*matrix[1][2])/matrix[1][1]}")
        # print(f"Z{i+1} =({b[2]} - {matrix[2][0]}*X{i+1} - {matrix[2][1]}*Y{i})/{matrix[2][2]} = {(b[2]-varx[i+1]*matrix[2][0]-vary[i+1]*matrix[2][1])/matrix[2][2]}")
        i+=1
        if abs(varx[i]-varx[i-1])<PRECISION and abs(vary[i]-vary[i-1])<PRECISION and abs(varz[i]-varz[i-1])<PRECISION:
            return [varx[i],vary[i],varz[i]]


def vector_1_norm(vector):
    """Calculate the 1-norm of a 3x1 vector."""
    return sum(abs(v) for v in vector)

def dominant_matrix(matrix):
    a = MATRIX_ZERO.copy()
    l = VECTOR_ZERO.copy()
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
        return True
    return False





# if __name__ == "__main__":
#     matrix=[[1,0.5,0.333],[0.5,0.333,0.25],[0.333,0.25,0.2]]
#     b= [1,0,0]


#     print("jacobi [x,y,z]:",jacobi(matrix,b))
#     print("Gauss Seidel [x,y,z]:",gauss_seidel(matrix,b))
