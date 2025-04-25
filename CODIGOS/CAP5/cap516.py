"""
Código 5.16 - Cálculo de la constante de proporcionalidad alpha para una densidad de corriente J = alpha*r^2

Este script utiliza SymPy para encontrar la relación entre la corriente total I y la constante alpha
en una distribución de corriente cilíndrica donde la densidad de corriente varía con el radio al cuadrado.
"""

import sympy as sym

# Definir los símbolos matemáticos
I, r, R, phi = sym.symbols('I r R phi', real=True, positive=True)

# Calcular alpha simbólicamente mediante integración
# La corriente total I es la integral de J·dA = integral(alpha*r^2 * r dr dphi)
alpha_symbolic = I / sym.integrate(r**3, (r, 0, R), (phi, 0, 2*sym.pi))

# Mostrar el resultado (usando print en lugar de display para entornos no interactivos)
alpha_simplified = alpha_symbolic.simplify()
print(f"La constante de proporcionalidad alpha es:")
print(f"α = {alpha_simplified}")

# Versión numérica para verificación
if __name__ == '__main__':
    # Valores de ejemplo
    I_val = 1.0  # Amperes
    R_val = 0.1  # Metros
    
    # Sustituir valores numéricos
    alpha_numeric = alpha_symbolic.subs({I: I_val, R: R_val})
    
    print("\nEjemplo numérico:")
    print(f"Para I = {I_val} A y R = {R_val} m:")
    print(f"α = {float(alpha_numeric.evalf()):.3f} A/m^4")