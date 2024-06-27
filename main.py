import numpy as np
import sympy as sp
from BoundaryProblem import BoundaryProblem
from HeatEquation import HeatEquation


def first_task():
    x = sp.symbols('x')
    A = 50*(x-0.5)
    B = -x*x+2
    C = 2*x-1
    Ynp = 1 + x + 10 * np.log(31) * x**3 * (1 - x)**3
    
    bp = BoundaryProblem(A, B, C, Ynp)
    eps = 1e-5
    h, _ = bp.adaptive_shooting_step(0, 1, eps)
    x = np.arange(0, 1 + h, h)
    z, table = bp.shooting_method(x, eps)
    bp.print_table(table, headers=['Itr', 'z(0)', 'y(1)', 'Delta'])
    # Начальные условия
    y0 = [1, z]

    # Временные интервалы
    x = np.arange(0, 1 + 2*h, h)

    # Решите систему дифференциальных уравнений
    y, res = bp.runge_kutta_4(y0, x)

    headers = ['x', 'y(x)', 'Ypr(x)', 'z(x)', 'Delta']
    bp.print_table(res, headers)
    print('Del_Max = ', max(abs(row[4]) for row in res))


def second_task():
    G = 0.2
    x, t = sp.symbols('x t')
    f = 0.1 * sp.sin(np.pi * x) * 30 + 0.1 * G * t * np.pi**2 * sp.sin(np.pi * x) * 30
    u_exact = x + 0.1 * t * sp.sin(np.pi * x) * 30
    
    N_list = [8, 16, 32]
    headers = ['t', 'delta', 'x']
    
    hq = HeatEquation(f, u_exact)
    
    print('xI = ', f'{G:.2f}')
    print('Метод конечных разностей (явная схема)')
    
    for N in N_list:
        print('N = ', N)
        table = hq.explicit_scheme(G, N)
        if N in [16, 32]:
            hq.print_table(table, headers=headers, full=False)
        else:
            hq.print_table(table, headers=headers)
        print('Del_T = ', max(row[1] for row in table))
        
    print('Метод конечных разностей (неявная схема)')
    for N in N_list:
        print('N = ', N)
        table = hq.implicit_scheme(G, N)
        if N in [16, 32]:
            hq.print_table(table, headers=headers, full=False)
        else:
            hq.print_table(table, headers=headers)
        print('Del_T = ', max(row[1] for row in table))


def main():
    print('Вариант 3')
    first_task()
    second_task()


if __name__ == '__main__':
    main()