"""
Cálculo y visualización de campos magnéticos para:
1. Alambre recto
2. Espira cuadrada
3. Anillo circular
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================== FUNCIONES AUXILIARES ==================

def producto_cruz(a, b):
    """Calcula el producto cruz entre dos vectores 3D"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],  # Componente x
        a[2]*b[0] - a[0]*b[2],  # Componente y
        a[0]*b[1] - a[1]*b[0]   # Componente z
    ])

def puntos_en_cartesianas(L=1.0, delta=0.2):
    """Genera puntos en un cubo 3D para evaluar el campo"""
    v = np.arange(-L, L + delta, delta)
    x, y, z = np.meshgrid(v, v, v, indexing='ij')
    return np.vstack((x.flatten(), y.flatten(), z.flatten()))

# ================== CONFIGURACIONES DE CORRIENTE ==================

def current_elements_along_line(I=1.0, length=1.0, dl=0.1):
    """Elementos de corriente para un alambre recto en el eje y"""
    y = np.arange(-length/2, length/2 + dl, dl)
    P = np.zeros((3, len(y)))
    P[1] = y  # Todos los puntos en el eje y
    J = np.zeros((3, len(y)))
    J[1] = I * dl  # Corriente en dirección +y
    return P, J

def current_elements_along_square(I=1.0, side=1.0, dl=0.1):
    """Elementos de corriente para una espira cuadrada en el plano xy"""
    # Un lado del cuadrado (derecho: x = side/2, y varía)
    vec = np.arange(-side/2, side/2, dl)
    N = len(vec)
    
    P = np.zeros((3, 4*N))
    J = np.zeros((3, 4*N))
    
    # Lado derecho (corriente +y)
    P[0, :N] = side/2
    P[1, :N] = vec
    J[1, :N] = I * dl
    
    # Lado superior (corriente -x)
    P[0, N:2*N] = vec[::-1]
    P[1, N:2*N] = side/2
    J[0, N:2*N] = -I * dl
    
    # Lado izquierdo (corriente -y)
    P[0, 2*N:3*N] = -side/2
    P[1, 2*N:3*N] = vec[::-1]
    J[1, 2*N:3*N] = -I * dl
    
    # Lado inferior (corriente +x)
    P[0, 3*N:4*N] = vec
    P[1, 3*N:4*N] = -side/2
    J[0, 3*N:4*N] = I * dl
    
    return P, J

def current_elements_along_circle(I=1.0, radius=1.0, dphi=np.pi/10):
    """Elementos de corriente para un anillo circular en el plano xy"""
    phi = np.arange(0, 2*np.pi, dphi)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    
    P = np.vstack((x, y, np.zeros_like(x)))
    J = np.vstack((-np.sin(phi), np.cos(phi), np.zeros_like(phi))) * I * radius * dphi
    
    return P, J

# ================== CÁLCULO DEL CAMPO MAGNÉTICO ==================

def get_campo_magnetico(P, P_curr, J_curr):
    """Calcula el campo magnético usando la ley de Biot-Savart"""
    mu0 = 4*np.pi*1e-7  # Permeabilidad del vacío
    B = np.zeros_like(P)
    
    for i in range(P.shape[1]):
        for j in range(P_curr.shape[1]):
            r = P[:,i] - P_curr[:,j]
            r_norm = np.linalg.norm(r)
            
            if r_norm < 1e-10:  # Evitar división por cero
                continue
                
            dB = (mu0/(4*np.pi)) * producto_cruz(J_curr[:,j], r) / (r_norm**3)
            B[:,i] += dB
            
    return B

# ================== VISUALIZACIÓN ==================

def plot_configuration(P_curr, J_curr, P, B, title=""):
    """Visualiza la configuración de corriente y el campo magnético"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar elementos de corriente
    ax.plot(P_curr[0], P_curr[1], P_curr[2], 'gray', linewidth=3)
    
    # Graficar campo magnético (solo puntos con |B| significativo)
    B_norm = np.linalg.norm(B, axis=0)
    mask = B_norm > 0.01 * B_norm.max()
    
    ax.quiver(P[0,mask], P[1,mask], P[2,mask],
              B[0,mask], B[1,mask], B[2,mask],
              color='black', length=0.3, normalize=True,
              arrow_length_ratio=0.3)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(title)
    plt.tight_layout()

# ================== EJECUCIÓN PRINCIPAL ==================

if __name__ == "__main__":
    I = 1.0  # Corriente en amperios
    
    # 1. Alambre recto
    print("Calculando campo para alambre recto...")
    P_curr, J_curr = current_elements_along_line(I=I, length=8.0, dl=0.1)
    P = puntos_en_cartesianas(L=1.5, delta=0.4)
    B = get_campo_magnetico(P, P_curr, J_curr)
    plot_configuration(P_curr, J_curr, P, B, "Alambre recto (eje y)")
    
    # 2. Espira cuadrada
    print("Calculando campo para espira cuadrada...")
    P_curr, J_curr = current_elements_along_square(I=I, side=1.0, dl=0.1)
    P = puntos_en_cartesianas(L=1.5, delta=0.4)
    B = get_campo_magnetico(P, P_curr, J_curr)
    plot_configuration(P_curr, J_curr, P, B, "Espira cuadrada (plano xy)")
    
    # 3. Anillo circular
    print("Calculando campo para anillo circular...")
    P_curr, J_curr = current_elements_along_circle(I=I, radius=1.0, dphi=np.pi/10)
    P = puntos_en_cartesianas(L=1.5, delta=0.4)
    B = get_campo_magnetico(P, P_curr, J_curr)
    plot_configuration(P_curr, J_curr, P, B, "Anillo circular (plano xy)")
    
    plt.show()