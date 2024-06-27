import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


class SystemSolver:
    def __init__(self, equations: list):
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')
        self.equations = equations

    def solve(self, x, y):
        for equation in self.equations:
            print(equation.solve(x, y), equation.solve(x, y))
    
    def visualize_system(self):
        # создаем список уравнений
        f_list = []
        for equation in self.equations:
            f_list.append(equation.f_lambdify)
        
        vals = np.linspace(-1, 1.5, 20)
        # создаем список решений уравнений
        f_vals_list = []
        for f in f_list:
            f_vals_list.append(f(vals))
        
        # Создание графика
        plt.figure(figsize=(10, 6))
        for i in range(len(f_vals_list)):
            if self.equations[i].arg == self.x:
                plt.plot(f_vals_list[i], vals)
            else:
                plt.plot(vals, f_vals_list[i])
        plt.title('График функции')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.xticks(np.arange(min(vals), max(vals) + 0.1, 0.1))
        plt.yticks(np.arange(min(vals), max(vals) + 0.1, 0.1))
        plt.show()
    
    def find_only_y_der(self):
        only_y = sp.Matrix()
        for i in range(len(self.equations)):
            only_y = only_y.row_insert(i, sp.Matrix([[self.equations[i].f0, sp.diff(self.equations[i].f0, self.y)]]))
        return only_y

    def find_only_x_der(self):
        only_x = sp.Matrix()
        for i in range(len(self.equations)):
            only_x = only_x.row_insert(i, sp.Matrix([[sp.diff(self.equations[i].f0, self.x), self.equations[i].f0]]))
        return only_x
    
    def find_all_der(self):
        all = sp.Matrix()
        for i in range(len(self.equations)):
            all = all.row_insert(i, sp.Matrix([[sp.diff(self.equations[i].f0, self.x), sp.diff(self.equations[i].f0, self.y)]]))
        return all
    
    def find_jacobian(self):
        jacobian = sp.Matrix()
        for i in range(len(self.equations)):
            jacobian = jacobian.row_insert(i, sp.Matrix([[self.equations[i].fx, self.equations[i].fy]]))
        return jacobian

    def find_values_mat(self, mat, x0, y0):
        mat_values = sp.Matrix()
        if mat.shape[1] == 1:
            for i in range(mat.shape[0]):
                f = sp.lambdify((self.x, self.y), mat[i, 0])
                val = f(x0, y0)
                mat_values = mat_values.row_insert(i, sp.Matrix([[val]]))
        else:
            for i in range(mat.shape[0]):
                fx_val = sp.lambdify((self.x, self.y), mat[i, 0])(x0, y0)
                fy_val = sp.lambdify((self.x, self.y), mat[i, 1])(x0, y0)
                mat_values = mat_values.row_insert(i, sp.Matrix([[fx_val, fy_val]]))
        return mat_values
    
    def print_jacobian(self, jacobian):
        for i in range(jacobian.shape[0]):
            for j in range(jacobian.shape[1]):
                print('{}'.format(jacobian[i, j]), end='\t')
            print()
    
    def print_mat(self, mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                print('{:.4f}'.format(mat[i, j]), end='\t')
            print()
    
    def norm_mat(self, mat):
        max_sum = 0
        for i in range(mat.shape[0]):
            sum_elem = 0
            for j in range(mat.shape[1]):
                sum_elem += abs(mat[i, j])
            max_sum = max(sum_elem, max_sum)
        return max_sum

    def simple_iteration(self, x0, y0, alfa0=1, eps=1e-4, max_iter=100):
        x_prev, y_prev = x0, y0
        f1_lambdify = self.equations[0].f_lambdify
        f2_lambdify = self.equations[1].f_lambdify
        f01 = sp.lambdify((self.x, self.y), self.equations[0].f0)
        f02 = sp.lambdify((self.x, self.y), self.equations[1].f0)
        jacobian = self.find_jacobian()
        table = []
        for i in range(max_iter):
            # обновляем значения x и y
            x_next = alfa0 * self.equations[0].solve_f(f1_lambdify, x_prev, y_prev)
            y_next = alfa0 * self.equations[1].solve_f(f2_lambdify, x_prev, y_prev)
            
            # вычисляем норму невязки
            norm_residual = sp.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2)
            # вычисляем норму Якобиана в текущей точке
            norm_jacobian = self.norm_mat(self.find_values_mat(jacobian, x_next, y_next))
            
            val_f1 = f01(x_next, y_next)
            val_f2 = f02(x_next, y_next)
            table.append([i+1, x_next, y_next, norm_residual, val_f1, val_f2, norm_jacobian])
            if norm_residual < eps:
                print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Норма невязки', 'F1', 'F2', 'Норма Якобиана']))
                return x_next, y_next
            x_prev, y_prev = x_next, y_next
        print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Норма невязки', 'F1', 'F2', 'Норма Якобиана']))
        return None
    
    def newton(self, x0, y0, lambda_=1, eps=1e-4, max_iter=100):
        x_prev, y_prev = x0, y0
        
        f01 = sp.lambdify((self.x, self.y), self.equations[0].f0)
        f02 = sp.lambdify((self.x, self.y), self.equations[1].f0)
        
        # вычисляем матрицу производных по обоим переменным и также только по x, или только по y
        all_der = self.find_all_der()
        only_y_der = self.find_only_y_der()
        only_x_der = self.find_only_x_der()
        table = []
        for i in range(max_iter):
            # вычисляем определитель матриц производных в текущей точке
            d = self.find_values_mat(all_der, x_prev, y_prev).det()
            d_only_y = self.find_values_mat(only_y_der, x_prev, y_prev).det()
            d_only_x = self.find_values_mat(only_x_der, x_prev, y_prev).det()
            
            # находим текущие h и k
            h = - 1 / d * d_only_y
            k = - 1 / d * d_only_x
        
            # обновляем значения x и y
            x_next = float(x_prev + lambda_ * h)
            y_next = float(y_prev + lambda_ * k)
            
            val_f1 = f01(x_next, y_next)
            val_f2 = f02(x_next, y_next)
            
            # вычисляем норму невязки
            norm_residual = sp.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2)
            table.append([i+1, x_next, y_next, norm_residual, val_f1, val_f2])
            if norm_residual < eps:
                print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Норма невязки', 'F1', 'F2']))
                return x_next, y_next
            x_prev, y_prev = x_next, y_next
        print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Норма невязки', 'F1', 'F2']))
        return None
    
    def gradient_descent(self, x0, y0, alpha=0.5, eps=1e-4, max_iter=100):
        f01 = self.equations[0].f0
        f02 = self.equations[1].f0
        # определяем функцию F
        F = f01**2 + f02**2

        # вычисляем производные функции F
        dFdx = sp.diff(F, self.x)
        dFdy = sp.diff(F, self.y)

        # преобразуем выражения sympy в функции
        f01 = sp.lambdify((self.x, self.y), f01)
        f02 = sp.lambdify((self.x, self.y), f02)
        dFdx = sp.lambdify((self.x, self.y), dFdx)
        dFdy = sp.lambdify((self.x, self.y), dFdy)

        x_prev, y_prev = x0, y0
        table = []
        for i in range(max_iter):
            # вычисляем значения функций и производных в текущей точке
            val_f1 = f01(x_prev, y_prev)
            val_f2 = f02(x_prev, y_prev)
            dFdx_val = dFdx(x_prev, y_prev)
            dFdy_val = dFdy(x_prev, y_prev)

            # обновляем значения x и y
            x_next = x_prev - alpha * dFdx_val
            y_next = y_prev - alpha * dFdy_val

            # вычисляем норму невязки
            norm_residual = sp.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2)

            table.append([i+1, x_next, y_next, alpha, norm_residual, val_f1, val_f2, sp.sqrt(val_f1**2 + val_f2**2), 1])

            if norm_residual < eps:
                print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Alfa', 'Норма невязки', 'F1', 'F2', 'FF', 'k']))
                return x_next, y_next

            x_prev, y_prev = x_next, y_next
        print(tabulate(tabular_data=table, headers=['Itr', 'x', 'y', 'Alfa', 'Норма невязки', 'F1', 'F2', 'FF', 'k']))
        return None
        