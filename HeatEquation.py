import numpy as np
import sympy as sp
from prettytable import PrettyTable


class HeatEquation:
    def __init__(self, f, u):
        self.x, self.t = sp.symbols('x t')
        self.f = sp.lambdify((self.t, self.x), f)
        self.u = sp.lambdify((self.t, self.x), u)

    def print_table(self, mat, headers=[], full=True):
        table = PrettyTable()
        table.field_names = headers
        if not full:
            for i in range(len(mat)):
                mat[i][2] = mat[i][2][:64]
        table.add_rows(mat)
        print(table)
        
    def explicit_scheme(self, G, N):
        h = 1.0 / N  # Шаг сетки
        r = h**2 / (4*G)  # Шаг по времени для явной схемы
        M = int(1/r) + 1  # Количество временных слоев

        # Инициализация сетки
        u = np.zeros((M, N+1))

        # Начальные и граничные условия
        for j in range(N+1):
            u[0, j] = j * h
        for n in range(M):
            u[n, 0] = 0
            u[n, N] = 1

        table = []
        # Явная схема
        for n in range(M-1):
            t = n*r
            for j in range(1, N):
                u[n+1, j] = u[n, j] + r * G * (u[n, j+1] - 2*u[n, j] + u[n, j-1]) / h**2 + r * self.f(t, j*h)
            # Расчет delta
            delta = max([abs(u[n, j] - self.u(t, j*h)) for j in range(N+1)])
            numbers_str = " ".join([f"{num:.5f}" for num in u[n]])
            table.append([f'{t:.3f}', delta, numbers_str])

        return table
    
    def implicit_scheme(self, G, N):
        h = tau = 1 / N
        x = np.arange(0, 1 + h, h)
        t = np.arange(0, 1 + h, tau)
        d = tau * G / (h * h)
        delta = np.zeros(len(t))

        temp = np.zeros((len(t), len(x)))
        temp[0, :] = x
        temp[:, -1] = 1

        table = []
        
        for n in range(1, len(t)):
            delta_max = 0
            p = np.zeros(len(x))
            q = np.zeros(len(x))
            for j in range(1, len(x)):
                div = 1 + 2 * d - d * p[j - 1]
                p[j] = d / div
                q[j] = (d * q[j-1] + temp[n-1, j] + tau * self.f(t[n], x[j])) / div
            for j in range(len(x) - 2, 0, -1):
                temp[n, j] = p[j] * temp[n, j + 1] + q[j]
                delta_max = max(delta_max, abs(self.u(t[n], x[j]) - temp[n, j]))
            delta[n] = delta_max
            # Форматирование строки для таблицы
            numbers_str = " ".join([f"{num:.5f}" for num in temp[n]])
            table.append([f'{t[n]:.3f}', delta[n], numbers_str])

        return table