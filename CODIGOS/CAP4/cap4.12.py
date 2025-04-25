"""
Este código utiliza la biblioteca SymPy para realizar integraciones simbólicas de la función 
f(t) = (k - cos(t)) / (k² + 1 - 2k cos(t)) en el intervalo de 0 a 2π. 
Se evalúa la integral para dos valores específicos de k (2 y 0.25) y se muestra el resultado. 
Además, se explora el caso en el que k es una variable simbólica, 
donde se observa que SymPy puede no calcular correctamente la integral.
"""

import sympy as sym  # Importa la biblioteca SymPy para cálculos simbólicos

# Definir la variable simbólica
t = sym.Symbol('t')  # Crea una variable simbólica t
k = 2  # Valor de k para la primera evaluación

# Evaluación para k = 2 (exterior)
print('When k = %4.3f (exterior)' % k)  # Imprime el valor de k
print('symbolic integration returns correct result (2*pi/k for k>1):')  # Mensaje informativo

# Realizar la integración simbólica
res = sym.integrate((k - sym.cos(t)) / ((k)**2 + 1 - 2 * (k) * sym.cos(t)), (t, 0, 2 * sym.pi))

# Mostrar el resultado de la integración
print(res)  # Muestra el resultado de la integración simbólica
print('')  # Línea en blanco para separación

# Evaluación para k = 0.25 (interior)
k = 0.25  # Valor de k para la segunda evaluación
print('When k = %4.3f (interior)' % k)  # Imprime el valor de k
print('symbolic integration returns correct result (0 for 0<k<1):')  # Mensaje informativo

# Realizar la integración simbólica
res = sym.integrate((k - sym.cos(t)) / ((k)**2 + 1 - 2 * (k) * sym.cos(t)), (t, 0, 2 * sym.pi))

# Mostrar el resultado de la integración
print('%4.3f' % res)  # Imprime el resultado de la integración
print('')  # Línea en blanco para separación

# Evaluación cuando k es una variable simbólica
k = sym.Symbol('k', positive=True)  # Crea una variable simbólica k, restringida a valores positivos
print('When k is a symbol,')  # Mensaje informativo
print('symbolic integration returns incorrect result:')  # Mensaje informativo

# Realizar la integración simbólica con k como variable simbólica
res = sym.integrate((k - sym.cos(t)) / ((k)**2 + 1 - 2 * (k) * sym.cos(t)), (t, 0, 2 * sym.pi))

# Mostrar el resultado de la integración
print(res)  # Muestra el resultado de la integración simbólica