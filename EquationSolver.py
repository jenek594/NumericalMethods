import sympy as sp

class EquationSolver:
    def __init__(self, f0, f, arg):
        self.x, self.y = sp.symbols('x y')
        self.f0 = f0
        self.f = f
        self.arg = arg
        self.fx = sp.diff(f, self.x)
        self.fy = sp.diff(f, self.y)
        self.f0_lambdify = sp.lambdify((self.x, self.y), f0)
        self.f_lambdify = self.prepare_f()
        self.df_lambdify = self.prepare_df()

    def prepare_f(self):
        if self.arg == self.x:
            return sp.lambdify(self.y, self.f)
        else:
            return sp.lambdify(self.x, self.f)
    
    def prepare_df(self):
        if self.arg == self.x:
            return sp.lambdify((self.y, self.x), self.fy)
        else:
            return sp.lambdify((self.x, self.y), self.fx)

    def solve_f(self, f_lambdify, x0, y0):
        if self.arg == self.x:
            return f_lambdify(y0)
        else:
            return f_lambdify(x0)