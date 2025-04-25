import numpy as np

# Constante de permeabilidad del vacío
mu0 = 4 * np.pi * 1e-7

# ----------------------------------------------
# Funciones de distribución de corriente
# ----------------------------------------------

def currents_along_line(I, L, dL):
    z = np.linspace(-L/2, L/2, int(L/dL))  
    p_curr = np.array([[0] * len(z), [0] * len(z), z])
    curr = np.array([I] * len(z))
    return p_curr, curr

def currents_along_circle(I, R, dtheta):
    theta = np.arange(0, 2 * np.pi, dtheta)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    p_curr = np.vstack((x, y, np.zeros_like(x)))  
    curr = np.array([I] * len(theta))
    return p_curr, curr

# ----------------------------------------------
# Cálculo de campo magnético (sólo placeholder)
# ----------------------------------------------

def get_magnetic_field(p, p_curr, curr):
    # Esto es solo un placeholder (campo cero en todos los puntos)
    B = np.zeros_like(p)
    return B

# ----------------------------------------------
# Comparación entre dos vectores
# ----------------------------------------------

def compare_two_vectors(V_exact, V_apprx):
    norm_exact = np.linalg.norm(V_exact)
    norm_apprx = np.linalg.norm(V_apprx)
    diff = V_exact - V_apprx
    norm_diff = np.linalg.norm(diff)
    print("Diferencia en magnitudes: %4.3f%%" % (norm_diff / norm_exact * 100))
    dot_prod = np.dot(V_exact.flatten(), V_apprx.flatten()) / (norm_exact * norm_apprx)
    print("Diferencia en dirección (Producto punto normalizado): %4.3f" % dot_prod)
    print('')

# ----------------------------------------------
# Comparación 1: Hilo recto (Solución analítica vs numérica)
# ----------------------------------------------

print("Comparación 1: Hilo recto")
print("Campo B en (0.2, 0, 0) debido a un hilo de L = 5, I = 2")

L, I, r = 5, 2, 0.2
phi = 0  # Punto sobre eje x → phi = 0

# Solución analítica
Bx = 0
By = (L) / (r * np.sqrt((4 * r**2) + L**2))
Bz = 0
B_wire_exact = np.array([[Bx], [By], [Bz]]) * (mu0 * I) / (2 * np.pi)

# Solución numérica
p = np.array([[r], [0], [0]])  # Punto de observación en (0.2, 0, 0)
p_curr, curr = currents_along_line(I, L, L * 0.05)
B_wire_approx = get_magnetic_field(p, p_curr, curr)

# Comparar
compare_two_vectors(B_wire_exact, B_wire_approx)

# ----------------------------------------------
# Comparación 2: Anillo
# ----------------------------------------------

print("Comparación 2: Anillo")
print("Campo B en (0, 0, 0.3) debido a un anillo de R = 0.5, I = 2")

R, I, z = 0.5, 2, 0.3

# Solución analítica (solo componente z)
Bz = (mu0 * I * R**2) / (2 * (R**2 + z**2)**(3/2))
B_ring_exact = np.array([[0], [0], [Bz]])

# Solución numérica
p = np.array([[0], [0], [z]])
p_curr, curr = currents_along_circle(I, R, np.pi/10)
B_ring_approx = get_magnetic_field(p, p_curr, curr)

# Comparar
compare_two_vectors(B_ring_exact, B_ring_approx)

# ----------------------------------------------
# Resumen
# ----------------------------------------------

print("Resumen de comparaciones:")
print("1. Hilo recto: solución analítica vs numérica.")
print("2. Anillo: solución analítica vs numérica.")
