"""
Este código realiza una verificación numérica de la integral de la función 
f(φ) = (k - cos(φ)) / (k² + 1 - 2k cos(φ)) mediante la suma del área bajo la curva. 
Se evalúa para dos valores de k (2 y 0.25) y se compara el resultado numérico 
con el resultado esperado. El gráfico muestra la función y el área bajo la curva 
para cada valor de k, y se guarda como un archivo PDF.
"""

import numpy as np  # Importa la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importa Matplotlib para visualización gráfica

# Crear figura
fig = plt.figure(figsize=(6, 3))  # Inicializa una figura con un tamaño de 6x3 pulgadas
d_phi = 0.00001  # Paso para el rango de ángulos
phi = np.arange(0, 2 * np.pi, d_phi)  # Rango de ángulos de 0 a 2π

# Inicialización del índice para subgráficos
i = 0

# Evaluar la integral para diferentes valores de k
for k in (2, 0.25):
    # Calcular el integrando
    integrand = (k - np.cos(phi)) / (k**2 + 1 - 2 * k * np.cos(phi))
    
    # Calcular el resultado de la integral mediante la suma
    integral_result = np.sum(integrand) * d_phi
    
    # Imprimir resultados
    print('k = %4.3f' % k)  # Imprime el valor de k
    print('Numerical result = %8.7f' % integral_result)  # Imprime el resultado numérico
    
    # Comparar con el resultado esperado
    if k > 1:
        print(' Expected result = %8.7f (2*pi/k for k>1)' % (2 * np.pi / k))  # Resultado esperado para k > 1
    else:
        print(' Expected result = 0.0 (for 0<k<1)')  # Resultado esperado para 0 < k < 1
    print('')  # Línea en blanco para separación

    # Graficar el integrando
    i += 1  # Incrementar el índice para el subgráfico
    plt.subplot(1, 2, i)  # Crear un subgráfico
    y = (k - np.cos(phi)) / (k**2 + 1 - 2 * k * np.cos(phi))  # Calcular la función para graficar
    plt.plot(phi, y, color='black')  # Graficar la función
    plt.fill_between(phi, y, color='gray')  # Rellenar el área bajo la curva
    plt.title('k = %3.2f' % k)  # Título del subgráfico
    plt.xlabel(r'$\phi$')  # Etiqueta del eje x
    plt.ylabel(r'$\frac{k-\cos \phi}{k^2+1-2k\cos \phi}$')  # Etiqueta del eje y

# Ajustes finales del gráfico
plt.tight_layout()  # Ajustar el diseño para evitar superposiciones
plt.savefig('fig_ch4_numerical_integral.pdf', bbox_inches='tight')  # Guardar el gráfico como un archivo PDF
plt.show()  # Mostrar el gráfico en pantalla