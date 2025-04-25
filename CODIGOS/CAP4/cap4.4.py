"""
Este script utiliza la biblioteca SymPy para realizar una integración simbólica.

En este caso, se calcula la integral definida de la función 1/(1+t^2) desde menos infinito hasta más infinito. Esta integral es conocida y su resultado es π, que es el área bajo la curva de la función en el intervalo especificado.

El código realiza las siguientes tareas:

1. **Importación de la biblioteca SymPy**: Se importa la biblioteca SymPy, que se utiliza para cálculos simbólicos en Python.

2. **Definición de la variable simbólica**: Se define una variable simbólica `t` que se utilizará en la expresión de la función.

3. **Cálculo de la integral**: Se calcula la integral definida de la función 1/(1+t^2) en el intervalo de menos infinito a más infinito.

4. **Impresión del resultado**: Se imprime el resultado de la integral.
"""

import sympy as sym  # Importa la biblioteca SymPy para cálculos simbólicos

# Define la variable simbólica
t = sym.Symbol('t')

# Calcula la integral definida de 1/(1+t^2) desde -infinito hasta +infinito
integral_result = sym.integrate(1/(1+t**2), (t, -sym.oo, sym.oo))

# Imprime el resultado de la integral
print(f"El resultado de la integral es: {integral_result}")