import numpy as np
import sympy as sp
from prettytable import PrettyTable


class BoundaryProblem:
    def __init__(self, a, b, c, ynp):
        self.x, self.y, self.t = sp.symbols('x y t')
        self.A = a
        self.B = b
        self.C = c
        self.ynp = ynp
        self.ynp_prime = self.get_der(self.ynp)
        self.ynp_double_prime = self.get_der(self.ynp_prime)
        self.F = self.ynp_double_prime + self.A * self.ynp_prime - self.B * self.ynp + self.C * sp.sin(self.ynp)
        self.A_lamb = self.get_lamb(self.A)
        self.B_lamb = self.get_lamb(self.B)
        self.C_lamb = self.get_lamb(self.C)
        self.F_lamb = self.get_lamb(self.F)

    def f(self, t, y):
        return np.array([y[1], self.F_lamb(t) - self.A_lamb(t) * y[1] + self.B_lamb(t) * y[0] - self.C_lamb(t) * np.sin(y[0])])
    
    def get_lamb(self, f):ф
        return sp.lambdify((self.x), f)

    def get_der(self, f, order=1):
        return sp.diff(f, self.x, order)
    
    def runge_kutta_4(self, y0, x):
        Ynp = self.get_lamb(self.ynp)
        y = np.zeros((len(x), len(y0)))
        y[0] = y0
        res = []
        for i in range(len(x) - 1):
            h = x[i+1] - x[i]
            k1 = h * self.f(x[i], y[i])
            k2 = h * self.f(x[i] + h/2, y[i] + k1/2)
            k3 = h * self.f(x[i] + h/2, y[i] + k2/2)
            k4 = h * self.f(x[i] + h, y[i] + k3)
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            res.append([f'{x[i]:.5f}', f'{y[i][0]:.5f}', f'{Ynp(x[i]):.5f}', f'{y[i][1]:.5f}', abs(Ynp(x[i]) - y[i][0])])
        return y, res

    def adaptive_shooting_step(self, x0, x_end, tol):
        Ynp = self.get_lamb(self.ynp)
        h = 0.1  # начальный шаг
        max_iter = 1000  # максимальное количество итераций
        y0 = [1, 1]
        for i in range(max_iter):
            x = np.arange(x0, x_end + h, h)
            y, _ = self.runge_kutta_4(y0, x)
            error = abs(Ynp(x[-1]) - y[-1][0])
            if error < tol:
                return h, y
            else:
                h /= 2  # уменьшаем шаг вдвое
        raise Exception("Не удалось достичь желаемой точности за отведенное количество итераций")
    
    def shooting_method(self, x, eps):
        Ynp = self.get_lamb(self.ynp)
        inc = -1.5
        res = []
        itr = 1
        y0 = [1, 0]
        y, _ = self.runge_kutta_4(y0, x)
        res.append([itr, f'{y0[1]:.5f}', f'{y[-1][0]:.6f}', max([abs(Ynp(x[i]) - y[i][0]) for i in range(len(x) - 1)])])
        
        while abs(Ynp(x[-1]) - y[-1][0]) > eps:
            y_prev = y[-1][0]
            inc = inc / 2
            if Ynp(x[-1]) < y_prev:
                y0[1] += inc
            else:
                y0[1] -= inc
            y, _ = self.runge_kutta_4(y0, x)
            res.append([itr, f'{y0[1]:.5f}', f'{y[-1][0]:.6f}', max([abs(Ynp(x[i]) - y[i][0]) for i in range(len(x) - 1)])])
            itr += 1
        
        return y[-1][1], res

    def print_table(self, mat, headers=[]):
        table = PrettyTable()
        table.field_names = headers
        table.add_rows(mat)
        print(table)
