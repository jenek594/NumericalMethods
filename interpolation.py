import re
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify
from prettytable import PrettyTable
from math import factorial


class Interpolation:
    def __init__(self, file_name, x_start=1, x_end=2, n=5):
        self.equation = None
        self.x = symbols('x')
        self.x_start = x_start
        self.x_end = x_end
        self.n = n

        self.load_function(file_name)
        self.f = lambdify(self.x, self.equation)
        self.f_prime = lambdify(self.x, diff(self.equation, self.x))
        self.f_double_prime = lambdify(self.x, diff(diff(self.equation, self.x)))

        self.x_values = np.linspace(self.x_start, self.x_end, self.n + 1)
        self.y_values = [self.f(x) for x in self.x_values]

    def load_function(self, file_name):
        with open(file_name, 'r') as file:
            equation = file.readline().strip()

        self.equation = self._transform_equation(equation)

    def _max_derivative(self, n, points):
        nth_derivative = lambdify(self.x, diff(self.equation, self.x, n))
        derivative_values = [nth_derivative(point) for point in points]
        return max(derivative_values)

    @staticmethod
    def _transform_equation(equation):
        equation = equation.replace("f(x) = ", "")
        match = re.search(r'e\^\((.*?)\)', equation)
        if match:
            exp_expression = match.group(1)
            equation = equation.replace(f"e^({exp_expression})", f"exp({exp_expression})")
        equation = equation.replace("^", "**")
        return equation

    def newton_method(self):
        print("Интерполяционная форма Ньютона")
        split_difference = np.zeros((self.n + 1, self.n + 1))

        h = (self.x_end - self.x_start) / self.n
        for i in range(self.n + 1):
            split_difference[i, 0] = self.f(self.x_values[i])

        for j in range(1, self.n + 1):
            for i in range(self.n - j + 1):
                split_difference[i, j] = split_difference[i + 1, j - 1] - split_difference[i, j - 1]

        table_split_difference = PrettyTable()

        table_split_difference.field_names = ["x"] + ["y"] + [f"delta^{j} y" for j in range(1, self.n + 1)]

        for i in range(self.n + 1):
            row = [f"{self.x_values[i]:.6f}" if self.x_values[i] != 0 else ""]
            row += [f"{val:.6f}" if val != 0 else "" for val in split_difference[i, :]]
            table_split_difference.add_row(row)

        print("Таблица разделенных разностей")
        print(table_split_difference)

        x_shifted = np.arange(self.x_start + h / 2, self.x_end, h)

        table_comparison = PrettyTable()
        table_comparison.field_names = ["x", "f(x)", "Pn(x)", "Delta", "Оценка"]

        m6 = self._max_derivative(6, x_shifted)
        fact_n_plus_1 = factorial(self.n + 1)
        print(f"M6 = {m6:.6f}")

        for x in x_shifted:
            Pn = split_difference[0, 0]
            f_x = self.f(x)
            h_pow_i = h
            fact_i = 1
            omega = 1

            for i in range(1, self.n + 1):
                fact_i *= i
                prod_term = 1
                for j in range(i):
                    prod_term *= (x - self.x_values[j])
                Pn += split_difference[0, i] * prod_term / (fact_i * h_pow_i)
                h_pow_i *= h

                omega *= (x - self.x_values[i - 1])

            delta = abs(f_x - Pn)

            estimate_error = abs(m6 * omega / fact_n_plus_1)

            row_comparison = [f"{x:.2f}", f"{f_x:.5f}", f"{Pn:.5f}", f"{delta:.2e}", f"{estimate_error:.2e}"]
            table_comparison.add_row(row_comparison)

        print(table_comparison)

    @staticmethod
    def _tridiagonal_matrix_alg(a, b, c, d):
        n = len(d)
        p = np.zeros(n)
        q = np.zeros(n)

        x = np.zeros(n)

        p[0] = -c[0] / b[0]
        q[0] = d[0] / b[0]

        for i in range(1, n):
            divider = b[i] + a[i] * p[i - 1]
            p[i] = -c[i] / divider
            q[i] = (d[i] - a[i] * q[i - 1]) / divider

        x[n - 1] = q[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = p[i] * x[i + 1] + q[i]

        return x

    def cubic_spline_method(self):
        print("\nИнтерполяция кубическим сплайном")
        h = (self.x_end - self.x_start) / self.n

        a = [0] + [0.5] * (self.n - 1) + [0]
        b = [1] + [2] * (self.n - 1) + [1]
        c = [0] + [0.5] * (self.n - 1) + [0]
        d = [0] + [(3 * (self.y_values[i - 1] - 2 * self.y_values[i] + self.y_values[i + 1])) / (h * h) for i in
                   range(1, self.n)] + [0]

        m_values = self._tridiagonal_matrix_alg(a, b, c, d)

        m5 = self._max_derivative(5, self.x_values)
        print(f"M5 = {m5:.6f}")
        estimate_error = m5 * h * h / 24

        table_m = PrettyTable()
        table_m.field_names = ["x[i]", "df/dx(x[i])", "m[i]", "Delta", "Оценка"]
        for i in range(self.n):
            delta = abs(m_values[i] - self.f_prime(self.x_values[i]))
            table_m.add_row(
                [f"{self.x_values[i]:.2f}", f"{self.f_prime(self.x_values[i]):.7f}", f"{m_values[i]:.7f}",
                 f"{delta:.2e}",
                 f"{estimate_error:.2e}"])

        print(table_m)

        table_s31 = PrettyTable()
        table_s31.field_names = ["x", "f(x)", "S31(f;x)", "Abs(f(x)-S31(f;x))", "Оценка"]
        x_shifted = np.arange(self.x_start + h / 2, self.x_end, h)

        m4 = self._max_derivative(4, x_shifted)
        print(f"M4 = {m4:.6f}")
        estimate_error_s31 = m4 * h ** 4 / 384

        i = 0
        for x in x_shifted:
            dx = x - self.x_values[i]
            dx_1 = x - self.x_values[i + 1]
            S31_x = m_values[i] * dx * (h ** 2 - dx ** 2) / (6 * h) + m_values[i] / 2 * dx * dx_1 + m_values[
                i + 1] * dx / (6 * h) * (dx ** 2 - h ** 2) + self.y_values[i + 1] * dx / h + self.y_values[i] * (
                            self.x_values[i + 1] - x) / h
            delta_s31 = abs(S31_x - self.f(x))
            i += 1

            table_s31.add_row(
                [f"{x:.2f}", f"{self.f(x):.5f}", f"{S31_x:.5f}", f"{delta_s31:.2e}", f"{estimate_error_s31:.2e}"])

        print(table_s31)

    @staticmethod
    def _print_matrix(matrix):
        table = PrettyTable(header=False)
        for row in matrix:
            table.add_row([f"{val:.5f}" for val in row])
        print(table)

    @staticmethod
    def _print_vector(vector):
        table = PrettyTable(header=False)
        table.add_column("Value", [f"{val:.5f}" for val in vector])
        print(table)

    def _plot_polynomials(self, polynomial_discrete, polynomial_continuous):
        x = np.linspace(self.x_start, self.x_end, 1000)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].scatter(self.x_values, self.y_values, color='red', label='f(x)')
        axs[0].plot(x, polynomial_discrete(x), label='p2(x)')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('Дискретное среднеквадратичное приближение')
        axs[0].legend()
        # Среденквардатичное приближение не проходит через заданные точки, а находит
        # приблежнную функцию и стремится мниинмизирвоать общие отклонения
        # Сплайн это кубический полином  матрица вектор свободных членов полином
        axs[1].plot(x[50:-50], self.f(x[50:-50]), label='f(x)')
        axs[1].plot(x[50:-50], polynomial_continuous(x[50:-50]), label='p2(x)')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].set_title('Непрерывное среднеквадратичное приближение')
        axs[1].legend()

        plt.show()

    def _compute_error(self, polynomial, x_values):
        polynomial_values = polynomial(x_values)
        function_values = self.f(x_values)

        errors = np.abs(polynomial_values - function_values)

        return np.max(errors)

    def rms_approximation(self):
        print("\nСреднеквадратичное приближение")

        print("Дискретный вариант")
        n = self.n + 1
        factors_number = 3

        matrix = np.zeros((factors_number, factors_number))
        for i in range(factors_number):
            for j in range(factors_number):
                matrix[i, j] = sum(self.x_values[k] ** (i + j) for k in range(n))
        print("Матрица")
        self._print_matrix(matrix)

        vector = np.zeros(factors_number)
        for i in range(factors_number):
            vector[i] = sum(self.y_values[k] * self.x_values[k] ** i for k in range(n))
        print("Вектор правых частей")
        self._print_vector(vector)

        coefficients = np.linalg.solve(matrix, vector)
        print(f"P2(x) = ({coefficients[0]:.6f}) + ({coefficients[1]:.6f})x + ({coefficients[2]:.6f})x^2")
        polynomial_equation_discrete = f"({coefficients[0]:}) + ({coefficients[1]})*x + ({coefficients[2]})*x**2"
        polynomial_discrete = lambdify(self.x, polynomial_equation_discrete)

        print(f"Норма погрешности {self._compute_error(polynomial_discrete, self.x_values):.6f}")

        print("Непрерывный вариант")

        x_values = np.linspace(self.x_start, self.x_end, 1000)

        matrix = np.zeros((factors_number, factors_number))
        for i in range(factors_number):
            for j in range(factors_number):
                matrix[i, j] = np.trapz(x_values ** (i + j), x_values)
        print("Матрица")
        self._print_matrix(matrix)

        vector = np.zeros(factors_number)
        for i in range(factors_number):
            vector[i] = np.trapz(self.f(x_values) * x_values ** i, x_values)
        print("Вектор правых частей")
        self._print_vector(vector)

        coefficients = np.linalg.solve(matrix, vector)
        print(f"P2(x) = ({coefficients[0]:.6f}) + ({coefficients[1]:.6f})x + ({coefficients[2]:.6f})x^2")
        polynomial_equation_continuous = f"({coefficients[0]}) + ({coefficients[1]})*x + ({coefficients[2]})*x**2"
        polynomial_continuous = lambdify(self.x, polynomial_equation_continuous)

        print(f"Норма погрешности {self._compute_error(polynomial_continuous, x_values[50:-50]):.6f}")

        self._plot_polynomials(polynomial_discrete, polynomial_continuous)

    def inverse_interpolation(self, c):
        print("\nРешение уравнения методом обратной интерполяции")
        split_difference = np.zeros((self.n + 2, self.n + 2))

        h = (self.x_end - self.x_start) / self.n

        for i in range(self.n + 1):
            split_difference[i, 0] = self.f(self.x_values[i])
            split_difference[i, 1] = self.x_values[i]

        for j in range(2, self.n + 2):
            for i in range(self.n - j + 2):
                split_difference[i, j] = (split_difference[i + 1, j - 1] - split_difference[i, j - 1]) / \
                                         (split_difference[i + j - 1, 0] - split_difference[i, 0])

        table_split_difference = PrettyTable()

        table_split_difference.field_names = ["y"] + ["x"] + [f"delta^{j} y" for j in range(1, self.n + 1)]

        for i in range(self.n + 1):
            row = [f"{val:.6f}" if val != 0 else "" for val in split_difference[i, :]]
            table_split_difference.add_row(row)

        print("Таблица разделенных разностей")
        print(table_split_difference)

        closest_i = min(range(self.n + 1), key=lambda i: abs(split_difference[i, 0] - c))

        x_root = split_difference[closest_i, 1]
        for j in range(2, closest_i + 2):
            product = 1
            for k in range(j - 1):
                product *= (c - split_difference[closest_i - k, 0])
            x_root += product * split_difference[closest_i - j + 1, j]

        print(f"Корень x для заданного c = {c} равен {x_root:.6f}, j = {closest_i + 1}")
        print(f"Невязка = abs(f(x) - c) = {abs(self.f(x_root) - c):.6f}")
