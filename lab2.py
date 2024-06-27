import numpy as np
from matrix import Matrix


def one_zad(matrix):
    # преобразование A в LU
    matrix.decompose_lu()

    # проверка на правильное преобразование
    res_mat = matrix.find_res_mat()
    print('L*U-P*A:')
    print(res_mat)

    # поиск ранга матрицы через LU
    rank = matrix.rank()
    print(f'Rank = {rank}')

    # решение системы и поиск x
    x = matrix.find_x(matrix.b)
    print('x: ')
    print(x)

    prov = Matrix.raz_vectors(Matrix.matrix_vector_multiply(matrix.a, x), matrix.b)
    print('Ax - b: ')
    print(prov)


def two_zad(matrix):
    print('A: ')
    print(matrix.a)

    A_inv = matrix.find_reverse_matrix()
    print('A^(-1): ')
    print(A_inv)

    print('A*A^(-1): ')
    print(matrix.a*A_inv)


def three_zad(matrix):
    # поиск определителя матрицы через LU
    det = matrix.det()
    print(f'Determinant = {det}')

    print('Число обусловленности матрицы A: ')
    cubic_norm = matrix.cubic_norm(matrix.a) * matrix.cubic_norm(matrix.find_reverse_matrix())
    print('1 норма: ',cubic_norm)

    octo_norm = matrix.octo_norm(matrix.a) * matrix.octo_norm(matrix.find_reverse_matrix())
    print('2 норма: ',octo_norm)

    evclidova_norm = matrix.evclidova_norm(matrix.a)
    print('3 норма: ', evclidova_norm)


def main():
    a = np.matrix([
        [-9.7, 5.2, -3.2, -0.9],
        [1.1, -0.8, -3.1, -8.4],
        [8.0, 8.2, -1.5, 0.3],
        [8.0, -2.0, 2.2, -9.4],
        ])
    x = [1, 2, 3, 4]

    print('Variant = 3')

    print('x: ')
    print(x)

    # расчет b
    b = Matrix.matrix_vector_multiply(a, x)
    print('b: ')
    print(b)

    matrix = Matrix(a, x, b)

    # 1 часть
    print('ЗАДАНИЕ 1')
    one_zad(matrix)

    # 2 часть
    print('ЗАДАНИЕ 2')
    two_zad(matrix)

    # 3 часть
    print('ЗАДАНИЕ 3')
    three_zad(matrix)


if __name__ == '__main__':
    main()
