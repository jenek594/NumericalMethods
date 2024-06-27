import numpy as np
import math


class Matrix:
    """Класс для выполнения операций над матрицей
    Атрибуты:
    ---------
    a: np.matrix  исходная матрица
    lu: np.matrix матрица полученная после lu разложения исходной
    n: int размерность исходной матрицы
    x: list решение системы уравнения
    e: np.matrix единичная матрица с размерностью исходной
    b: list правая часть системы уравнений
    c: list порядок строк в исходной матрице
    """
    def __init__(self, a, x, b):
        self.a = np.matrix(a)
        self.lu = self.a.copy()
        self.n = self.a.shape[0]
        self.x = x.copy()
        self.b = np.array(b)
        self.e = self.create_e_mat()
        self.c = [i+1 for i in range(self.a.shape[0])]

    def create_e_mat(self):
        e = np.eye((self.n))
        return e

    def get_L(self):
        L = self.lu.copy()
        for i in range(L.shape[0]):
            L[i, i + 1:] = 0
        return L

    def get_U(self):
        U = self.lu.copy()
        for i in range(U.shape[0]):
            U[i, i] = 1
            U[i, :i] = 0
        return U

    def find_max(self, k: int):
        ap = abs(self.lu[k, k])
        p = k
        for i in range(k, self.n):
            if abs(self.lu[i, k]) > ap:
                ap = abs(self.lu[i, k])
                p = i
        return self.lu[p, k], p

    def decompose_to_lu(self, print_info=True):
        for k in range(self.n):
            # ищем максимальный элемент по модулю в k столбце
            ap, p = self.find_max(k)
            # перестановка строк k и p в начальной матрице, в x и в b
            self.a[[k, p]] = self.a[[p, k]]
            self.c[k], self.c[p] = self.c[p], self.c[k]
            self.b[k], self.b[p] = self.b[p], self.b[k]

            # поиск коэффициента m и домножение всех строк кроме p на коэффициент m
            for i in range(k, self.n):
                if i != p:
                    m = - self.lu[i, k] / self.lu[p, k]
                    # домножение строк
                    self.lu[i, k:] += self.lu[p, k:] * m
                    # заполнение L матрицы
                    if k != 0:
                        self.lu[i, k] = - self.lu[p, k] * m

            # нормализация главной строки
            for t in range(k+1, self.n):
                self.lu[p, t] /= ap
            # перестановка строк в lu матрице начиная с k столбца
            self.lu[k, k:], self.lu[p, k:] = self.lu[p, k:].copy(), self.lu[k, k:].copy()

            # дозаполнение L матрицы (заполнение 1 столбца)
            for j in range(k):
                if j == 0:
                    self.lu[k, j] = self.a[k, j]
            if print_info:
                print('L:')
                Matrix.print_matrix(self.get_L())
                print('U:')
                Matrix.print_matrix(self.get_U())
                print(f'ap = {ap:.7f} p = {p+1} k = {k+1}')

    
    def find_y(self, b):
        # находим y
        L = self.get_L()
        y = [0 for _ in range(self.n)]
        for i in range(self.n):
            y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
        return y

    def find_X(self, y):
        # находим x
        U = self.get_U()
        x = [0 for _ in range(self.n)]
        for i in range(self.n-1, -1, -1):
            x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, self.n))) / U[i, i]
        return x

    def find_x(self, b):
        y = self.find_y(b)
        x = self.find_X(y)
        return x

    def find_sobst_elem(mat):
        # транспонируем матрицу, преобразуем матрицу к симметричной
        A_trans = Matrix.transpose_matrix(mat)
        A_sim = mat * A_trans
        # пользуясь методом Якоби находим собственные значения
        sobs = Matrix.jacobi_method(A_sim)
        # находим норму матрицы как корень из отношения максимального элемента к минимальному
        return sobs
    
    def evclidova_norm(mat):
        sobs = Matrix.find_sobst_elem(mat)
        res = math.sqrt(max(abs(sobs)))
        return res
    
    def evclidova_obysl(mat):
        sobs = Matrix.find_sobst_elem(mat)
        res = math.sqrt(max(abs(sobs))/min(abs(sobs)))
        return res

    def transpose_matrix(mat):
        n = mat.shape[0]
        A_trans = np.zeros(mat.shape)

        for i in range(n):
            for j in range(n):
                A_trans[j, i] = mat[i, j]

        return A_trans

    def off(A):
        """
        Вычисляет норму внедиагональных элементов матрицы.
        Параметры:
        A : Матрица.
        Возвращает:
        norm : Норма внедиагональных элементов.
        """
        return np.sqrt(np.sum(A * A) - np.sum(np.diag(A)**2))

    def jacobi_method(matrix, tol=1e-10, iterations=100):
        """
        Реализация метода Якоби для нахождения собственных значений матрицы.
        Параметры:
        matrix : Симметричная квадратная матрица.
        iterations : Количество итераций.
        Возвращает:
        matrix : Диагональная матрица с собственными значениями на диагонали.
        """
        mat = matrix.copy()
        n = mat.shape[0]

        for _ in range(iterations):
            # Находим максимальный внедиагональный элемент
            max_val = 0
            p, q = 0, 0
            for i in range(n):
                for j in range(i+1, n):
                    if abs(mat[i, j]) > max_val:
                        max_val = abs(mat[i, j])
                        p, q = i, j

            # Вычисляем угол поворота
            if mat[p, p] == mat[q, q]:
                theta = np.pi / 4 if mat[p, q] > 0 else -np.pi / 4
            else:
                theta = 0.5 * np.arctan(2 * mat[p, q] / (mat[p, p] - mat[q, q]))

            # Вычисляем элементы матрицы поворота
            cos = np.cos(theta)
            sin = np.sin(theta)

            # Применяем матрицу поворота
            matrix_new = np.copy(mat)
            for i in range(n):
                if i != p and i != q:
                    matrix_new[p, i] = matrix_new[i, p] = cos * mat[p, i] + sin * mat[q, i]
                    matrix_new[q, i] = matrix_new[i, q] = -sin * mat[p, i] + cos * mat[q, i]
            
            matrix_new[p, p] = cos**2 * mat[p, p] + 2 * sin * cos * mat[p, q] + sin**2 * mat[q, q]
            matrix_new[q, q] = sin**2 * mat[p, p] - 2 * sin * cos * mat[p, q] + cos**2 * mat[q, q]
            
            matrix_new[p, q] = matrix_new[q, p] = 0.0

            if Matrix.off(matrix_new) < tol:
                break
            
            # Обновляем матрицу
            mat = np.copy(matrix_new)

        return np.diag(mat)

    def vector_norm(x):
        return sum(abs(i) for i in x)
    
    def vector_evclidova_norm(x):
        return math.sqrt(sum(i ** 2 for i in x))
    
    def print_info(A, itr, tau, norm_r, x, prev_x, q, alpha=0):
        n = A.shape[0]
        # оценка погрешности
        dif = sum(x[i]-prev_x[i] for i in range(n))/sum(prev_x)
        if alpha != 0:
            print(f'{itr:5} | {tau:6.4f} | {q:6.4f} | {norm_r:13.7f} | {dif:11.7f} | {x[0]:8.5f} | {x[1]:8.5f} | {x[2]:8.5f} | {x[3]:8.5f} | {alpha:8.5f}')
        else:
            print(f'{itr:5} | {tau:6.4f} | {q:6.4f} | {norm_r:13.7f} | {dif:11.7f} | {x[0]:8.5f} | {x[1]:8.5f} | {x[2]:8.5f} | {x[3]:8.5f}')
    
    def simple_iteration(self, A, b, x0, tau, eps, max_iterations):
        x = x0.copy()
        A = np.array(A)
        itr = 1
        print(' Itr  |  Tau   |   q    | Норма невязки | Погрешность |   x[1]   |   x[2]   |   x[3]   |   x[4]   ')
        prev_norm_q = Matrix.vector_norm(x)
        for k in range(max_iterations):
            prev_x = x.copy()
            r = b - np.dot(A, x)  # Вычисляем невязку
            x += tau * r  # Обновляем приближение
            norm_r = Matrix.vector_evclidova_norm(x - self.x)
            norm_q = Matrix.vector_norm(x - prev_x)
            q = norm_q / prev_norm_q
            Matrix.print_info(A, itr, tau, norm_r, x, prev_x, q)
            if norm_r < eps:
                return x, k
            prev_norm_q = norm_q
            itr += 1
        raise Exception("Метод не сошелся в заданное количество итераций.")
    
    def gradient_descent(self, A, b, x0, eps, tau, max_iterations):
        x = x0.copy()
        A = np.array(A)
        print(' Itr  |  Tau   |   q    | Норма невязки | Погрешность |   x[1]   |   x[2]   |   x[3]   |   x[4]   ')
        prev_norm_q = Matrix.vector_norm(x)
        for k in range(1, max_iterations):
            prev_x = x.copy()
            r = np.dot(A, x) - b
            Ar = np.dot(r, A)
            tau = np.dot(r.T, r) / np.dot(Ar, r)
            x -= tau * r
            norm_r = Matrix.vector_evclidova_norm(x - self.x)
            
            norm_q = Matrix.vector_norm(x - prev_x)
            q = norm_q / prev_norm_q
            
            Matrix.print_info(A, k, tau, norm_r, x, prev_x, q)
            if norm_r < eps:
                return x, k
            prev_norm_q = norm_q
        raise Exception("Метод не сошелся в заданное количество итераций.")

    def sor_method(self, A, b, x0, eps, w, max_iterations, print_info=True):
        x = x0.copy()
        n = len(b)
        A = np.array(A)
        itr = 1
        prev_norm_q = Matrix.vector_norm(x)
        if print_info:
            print(' Itr  |  Tau   |   q    | Норма невязки | Погрешность |   x[1]   |   x[2]   |   x[3]   |   x[4]   ')
        for k in range(1, max_iterations):
            prev_x = x.copy()
            for i in range(n):
                sigma = 1 / A[i, i] * (b[i] - sum(A[i, j]*x[j] for j in range(i)) - sum(A[i, j]*prev_x[j] for j in range(i+1, n)))
                x[i] += w * (sigma - x[i])
            norm_r = Matrix.vector_evclidova_norm(x - self.x)
            
            norm_q = Matrix.vector_norm(x - prev_x)
            q = norm_q / prev_norm_q
            
            if print_info:
                Matrix.print_info(A, itr, w, norm_r, x, prev_x, q)
            if norm_r < eps:
                return itr, x, k
            prev_norm_q = norm_q
            itr += 1
        raise Exception("Метод не сошелся в заданное количество итераций.")

    def choose_optimal_w(self, A, b, x0, eps, max_iterations):
        min_itr, opt_w = 9999, 0
        i = 0.1
        while i < 2:
            itr, _, _ = self.sor_method(A, b, x0, eps, i, max_iterations, print_info=False)
            print(f'w={i:.1f} Itr={itr}')
            if itr < min_itr:
                min_itr = itr
                opt_w = i
            i += 0.1
        return min_itr, opt_w

    def conjugate_gradient(self, A, b, x0, eps, max_iterations):
        x = x0.copy()
        A = np.array(A)
        # начальная невязка
        r = b - np.dot(A, x)
        # направление поиска
        d = r
        itr = 1
        prev_norm_q = Matrix.vector_evclidova_norm(x)
        alpha = 1
        print(' Itr  |  Tau   |   q    | Норма невязки | Погрешность |   x[1]   |   x[2]   |   x[3]   |   x[4]   |   alpha   ')
        for k in range(1, max_iterations):
            prev_x = x.copy()
            alpha = np.dot(r, r) / np.dot(d, np.dot(A, d))
            x = x + alpha * d
            r_new = r - alpha * np.dot(A, d)
            norm_r = Matrix.vector_evclidova_norm(r)
            
            norm_q = Matrix.vector_norm(x - prev_x)
            q = norm_q / prev_norm_q
            tau = np.dot(r_new, r_new) / np.dot(r, r)
            Matrix.print_info(A, itr, tau, norm_r, x, prev_x, q, alpha)
            
            if norm_r < eps:
                return x, k
            prev_norm_q = norm_q
            d = r_new + tau * d
            r = r_new
            itr += 1
        raise Exception("Метод не сошелся в заданное количество итераций.")