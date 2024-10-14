import pathlib
from readMatrix import load_from_file

file_path = pathlib.Path("tema1.txt")
A, b = load_from_file(file_path)

def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def det_3x3(matrix):
    a, b, c = matrix[0]
    submatrix1 = [row[1:] for row in matrix[1:]]
    submatrix2 = [row[::2] for row in matrix[1:]]
    submatrix3 = [row[:2] for row in matrix[1:]]

    return a * det_2x2(submatrix1) - b * det_2x2(submatrix2) + c * det_2x2(submatrix3)

def cramers_rule(A, b):
    det_A = det_3x3(A)
    if det_A == 0:
        return None # Multiple solutions

    solutions = []
    for i in range(3):
        Ai = [row[:] for row in A]
        for j in range(3):
            Ai[j][i] = b[j]
        solutions.append(det_3x3(Ai) / det_A)
    return solutions

def trace(matrix):
    return sum([matrix[i][i] for i in range(len(matrix))])

def vector_norm(vector):
    return sum([b ** 2 for b in vector]) ** 0.5

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_multiply(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]

def solve_cramer(matrix, vector):
    det_A = det_3x3(matrix)
    if det_A == 0:
        return None # Multiple solutions

    def replace_column(matrix, index, new_column):
        return [[new_column[i] if j == index else matrix[i][j] for j in range(3)] for i in range(3)]

    A_x = replace_column(matrix, 0, vector)
    A_y = replace_column(matrix, 1, vector)
    A_z = replace_column(matrix, 2, vector)

    return [det_3x3(A_x) / det_A, det_3x3(A_y) / det_A, det_3x3(A_z) / det_A]

def matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

def cofactor_matrix(matrix):
    return [[(-1) ** (i + j) * det_2x2(matrix_minor(matrix, i, j)) for j in range(len(matrix))] for i in range(len(matrix))]

def adjugate(matrix):
    return transpose(cofactor_matrix(matrix))

def inverse(matrix):
    det = det_3x3(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    return [[adj[i][j] / det for j in range(len(matrix))] for i in range(len(matrix))]

def solve_using_inversion(matrix, vector):
    inv = inverse(matrix)
    if inv is None:
        return None

    return matrix_multiply(inv, vector)

print("\nCramer's rule:")
print(solve_cramer(A, b))

print("\nSolve using inversion:")
print(solve_using_inversion(A, b))
