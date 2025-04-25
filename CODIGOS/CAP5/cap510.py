import numpy as np
import matplotlib.pyplot as plt

# Este código cuantifica la diferencia entre dos vectores A y B, 
# visualiza los vectores y calcula la diferencia en magnitudes y dirección.

# Crear una figura para graficar
fig = plt.figure(figsize=(3, 3))

# Definir los vectores A y B
A = np.array([0.5, 0.8])  # Vector A
B = np.array([0.55, 0.75])  # Vector B

# Calcular la diferencia entre B y A
C = B - A  # Vector de diferencia

# Graficar el vector A
plt.quiver(0, 0, A[0], A[1],
           angles='xy', scale_units='xy', scale=1, color='#000000')

# Graficar el vector B
plt.quiver(0, 0, B[0], B[1],
           angles='xy', scale_units='xy', scale=1, color='#CCCCCC')

# Graficar el vector de diferencia C
plt.quiver(A[0], A[1], C[0], C[1],
           angles='xy', scale_units='xy', scale=1, color='#808080')

# Añadir leyenda para los vectores
plt.legend(('A', 'B', 'B-A'))

# Configurar las marcas en los ejes
plt.xticks((0, 0.5, 1))
plt.yticks((0, 0.5, 1))

# Configurar los límites de los ejes
plt.xlim((-0.1, 1.1))
plt.ylim((-0.1, 1.1))

# Configurar la relación de aspecto de la gráfica
plt.axis('square')

# Guardar la figura como un archivo PDF
plt.savefig('fig_ch5_vec_diff.pdf')

# Mostrar la gráfica
plt.show()

# Función para comparar dos vectores
def compare_two_vectors(V_exact, V_apprx):
    # Calcular la norma (magnitud) del vector exacto
    norm_exact = np.sqrt(np.sum(V_exact**2))
    
    # Calcular la norma (magnitud) del vector aproximado
    norm_apprx = np.sqrt(np.sum(V_apprx**2))
    
    # Calcular la diferencia entre los dos vectores
    diff = V_exact - V_apprx
    
    # Calcular la norma de la diferencia
    norm_diff = np.sqrt(np.sum(diff**2))
    
    # Imprimir la diferencia en magnitudes como un porcentaje
    print("Difference in magnitudes: %4.3f percent" % (norm_diff / norm_exact * 100))
    
    # Calcular el producto punto normalizado para comparar direcciones
    dot_prod = np.sum(V_exact * V_apprx) / (norm_exact * norm_apprx)
    
    # Imprimir la diferencia en dirección
    print("Difference in direction (Norm. Dot Product): %4.3f" % dot_prod)
    print('')

# Ejemplo de comparación de dos vectores
print('Example case: comparing two vectors')
compare_two_vectors(A, B)  # Comparar los vectores A y B
plt.show()