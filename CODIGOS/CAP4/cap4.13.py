# ================================================================
# Este código evalúa una integral definida que depende del parámetro k,
# el cual puede tomar valores mayores o menores que 1.
# Se analiza cómo cambia el resultado de la integral según el valor
# de k (específicamente si k > 1 o k < 1), bajo la suposición de que
# g es un número positivo. Esto es útil para explorar propiedades
# simétricas de integrales en coordenadas polares.
# ================================================================

import sympy as sym

# Declarar variables simbólicas
g = sym.Symbol('g', positive=True)  # g > 0
k = sym.Symbol('k', positive=True)  # k se definirá luego en términos de g
t = sym.Symbol('t')                 # variable de integración (ángulo)

# CASO 1: k > 1, se asegura tomando k = g + 1
k = g + 1
print('When k = g + 1 > 1 with positive g,')

# Definir y evaluar la integral simbólicamente
res = sym.integrate(
    (k - sym.cos(t)) / ((k)**2 + 1 - 2 * k * sym.cos(t)),
    (t, 0, 2 * sym.pi)
)

# Mostrar el resultado
sym.pprint(res)
print('')  # línea en blanco

# CASO 2: k < 1, se asegura tomando k = 1 - g
k = 1 - g
print('When k = 1 - g < 1 with positive g,')

# Evaluar nuevamente la misma integral con el nuevo valor de k
res = sym.integrate(
    (k - sym.cos(t)) / ((k)**2 + 1 - 2 * k * sym.cos(t)),
    (t, 0, 2 * sym.pi)
)

# Mostrar el resultado
sym.pprint(res)
