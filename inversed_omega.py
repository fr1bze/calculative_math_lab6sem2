import sympy as sp

# Define the matrix
u, c, gamma_1 = sp.symbols('u c gamma_1')
omega_t = sp.Matrix([[-u*c, c, gamma_1],
                     [-c*c, 0, gamma_1],
                     [u*c, -c, gamma_1]])

omega_t_inv = omega_t.inv()

print(omega_t_inv)
