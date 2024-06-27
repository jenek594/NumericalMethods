import sympy as sp
import numpy as np
from prettytable import PrettyTable


class Integration:
    def __init__(self, f, a, b):
        self.x = sp.symbols('x')
        self.f = f
        self.a = a
        self.b = b
        self.lamb_f = self.get_lamb(self.f)
        self.f_der = self.get_der(self.f)

    def get_lamb(self, f):
        return sp.lambdify((self.x), f)
    
    def get_der(self, f, order=1):
        return sp.diff(f, self.x, order)

    def print_table(self, mat, headers=[]):
        table = PrettyTable()
        table.field_names = headers
        table.add_rows(mat)
        print(table)

    def find_err_runge(self, I, prev_I, n, k):
        return (n//2)**k/(n**k - (n//2)**k)*(I - prev_I)
    
    def calc_trapezoid_method(self, n, h):
        x_values = np.arange(self.a, self.b+h/2, h)
        y_values = np.array([self.lamb_f(val) for val in x_values])
        if n == 1:
            res = h*0.5*(self.lamb_f(self.a) + self.lamb_f(self.b))
        else:
            res = h * (0.5*y_values[0] + 0.5*y_values[-1] + np.sum(y_values[1:-1]))
        return res
    
    def trapezoid_method(self, eps, limit):
        n = 1
        k = 2
        res = []
        h = (self.b - self.a)/n
        delta = 1
        prev_I = self.calc_trapezoid_method(n, h)
        prev_delta = 1
        res.append([n, round(h, 4), round(prev_I, 6), '-', '-'])
        
        for i in range(limit):
            if abs(delta) < eps:
                return res, I, n+1
            n *= 2
            h = (self.b - self.a)/(n)
            I = self.calc_trapezoid_method(n, h)
            delta = self.find_err_runge(I, prev_I, n, k)
            if prev_delta != 1:
                new_k = np.log2(abs(prev_delta/delta))
                res.append([n, round(h, 4), round(I, 6), delta, round(new_k, 4)])
            else:
                res.append([n, round(h, 4), round(I, 6), delta, '-'])
            prev_I = I
            prev_delta = delta
            
        return res, I
    
    def calc_mod_trapezoid_method(self, h):
        x_values = np.arange(self.a, self.b+h/2, h)
        y_values = np.array([self.lamb_f(val) for val in x_values])
        f_der_lamb = self.get_lamb(self.f_der)
        res = h * (0.5*y_values[0] + 0.5*y_values[-1] + np.sum(y_values[1:-1])) + (h**2)/12 * (f_der_lamb(x_values[0]) - f_der_lamb(x_values[-1]))
        return res
    
    def mod_trapezoid_method(self, eps, limit):
        n = 1
        k = 4
        res = []
        h = (self.b - self.a)/n
        delta = 1
        prev_I = self.calc_mod_trapezoid_method(h)
        prev_delta = 1
        res.append([n, round(h, 4), round(prev_I, 6), '-', '-'])
        for i in range(limit):
            if abs(delta) < eps:
                return res, I, n+1
            n *= 2
            h = (self.b - self.a)/(n)
            I = self.calc_mod_trapezoid_method(h)
            delta = self.find_err_runge(I, prev_I, n, k)
            if prev_delta != 1:
                new_k = np.log2(abs(prev_delta/delta))
                res.append([n, round(h, 4), round(I, 6), delta, round(new_k, 4)])
            else:
                res.append([n, round(h, 4), round(I, 6), delta, '-'])
            prev_I = I
            prev_delta = delta
        return res, I
    
    def calc_simpson_method(self, n, h):
        x_values = np.arange(self.a, self.b+h/2, h)
        y_values = np.array([self.lamb_f(val) for val in x_values])
        return h / 3 * (y_values[0] + y_values[-1] + sum(4*y_values[i] if i % 2 == 1 else 2*y_values[i] for i in range(1, n)))
    
    def simpson_method(self, eps, limit):
        n = 1
        k = 4
        res = []
        h = (self.b - self.a)/n
        delta = 1
        prev_I = self.calc_simpson_method(n*k, (self.b - self.a)/(n*k))
        prev_delta = 1
        res.append([n, round(h, 4), round(prev_I, 6), '-', '-'])
        for i in range(limit):
            if abs(delta) < eps and i > 3:
                return res, I, n*2+1
            n *= 2
            h = (self.b - self.a)/(n)
            I = self.calc_simpson_method(n*k, (self.b - self.a)/(n*k))
            delta = self.find_err_runge(I, prev_I, n, k)
            if prev_delta != 1:
                new_k = np.log2(abs(prev_delta/delta))
                res.append([n, round(h, 4), round(I, 6), delta, round(new_k, 4)])
            else:
                res.append([n, round(h, 4), round(I, 6), delta, '-'])
            prev_I = I
            prev_delta = delta
        return res, I
    
    def calc_gauss_three_point(self, n, h, t, C):
        x_values = np.arange(self.a, self.b+h/2, h)
        I = 0
        for i in range(n):
            I += h / 2 * sum(C[j] * self.lamb_f(x_values[i] + h/2 * (t[j] + 1)) for j in range(3))
        return I
    
    def gauss_three_point(self, eps, limit):
        t = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        C = np.array([5/9, 8/9, 5/9])
        n = 1
        k = 6
        res = []
        kobr = 1
        h = (self.b - self.a)/n
        delta = 1
        prev_I = self.calc_gauss_three_point(n, h, t, C)
        prev_delta = 1
        res.append([n, round(h, 4), round(prev_I, 6), '-', '-'])
        for i in range(limit):
            if abs(delta) < eps and i > 1:
                return res, I, kobr*3
            n *= 2
            h = (self.b - self.a)/(n)
            I = self.calc_gauss_three_point(n, h, t, C)
            delta = self.find_err_runge(I, prev_I, n, k)
            if prev_delta != 1:
                new_k = np.log2(abs(prev_delta/delta))
                res.append([n, round(h, 4), round(I, 6), delta, round(new_k, 4)])
            else:
                res.append([n, round(h, 4), round(I, 6), delta, '-'])
            prev_I = I
            prev_delta = delta
            kobr += n
        return res, I
