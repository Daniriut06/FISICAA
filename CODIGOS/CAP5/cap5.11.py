import numpy as np
import matplotlib.pyplot as plt

# ================== VECTOR COMPARISON VISUALIZATION ==================

def plot_vector_comparison(A, B, filename='vector_comparison.pdf'):
    """
    Visualize two vectors and their difference
    
    Parameters:
        A: First vector (exact)
        B: Second vector (approximation)
        filename: Name for saving the figure
    """
    fig = plt.figure(figsize=(4, 4))
    C = B - A  # Difference vector
    
    # Plot vectors with different colors and styles
    plt.quiver(0, 0, A[0], A[1],
               angles='xy', scale_units='xy', scale=1,
               color='#000000', width=0.008, label='A (Exact)')
    
    plt.quiver(0, 0, B[0], B[1],
               angles='xy', scale_units='xy', scale=1,
               color='#CCCCCC', width=0.008, label='B (Approx)')
    
    plt.quiver(A[0], A[1], C[0], C[1],
               angles='xy', scale_units='xy', scale=1,
               color='#808080', width=0.008, linestyle='--', label='Difference (B-A)')
    
    # Add annotation for the difference vector
    mid_point = A + C/2
    plt.annotate(f'Δ = {np.linalg.norm(C):.2f}', 
                 xy=mid_point, xytext=(5,5),
                 textcoords='offset points')
    
    # Configure plot appearance
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('square')
    plt.title('Vector Comparison')
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# ================== VECTOR COMPARISON METRICS ==================

def compare_two_vectors(V_exact, V_apprx):
    """
    Calculate and print comparison metrics between two vectors
    
    Parameters:
        V_exact: Reference vector
        V_apprx: Vector to compare
    """
    # Calculate magnitudes
    norm_exact = np.linalg.norm(V_exact)
    norm_apprx = np.linalg.norm(V_apprx)
    
    # Calculate difference vector and its magnitude
    diff = V_exact - V_apprx
    norm_diff = np.linalg.norm(diff)
    
    # Calculate angle between vectors (in degrees)
    dot_prod = np.dot(V_exact, V_apprx) / (norm_exact * norm_apprx)
    angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    
    # Print comparison metrics
    print("\n=== Vector Comparison Metrics ===")
    print(f"Reference vector (A): {V_exact}")
    print(f"Comparison vector (B): {V_apprx}")
    print(f"\nMagnitude of A: {norm_exact:.4f}")
    print(f"Magnitude of B: {norm_apprx:.4f}")
    print(f"Magnitude difference: {abs(norm_exact - norm_apprx):.4f}")
    print(f"Relative magnitude difference: {norm_diff/norm_exact*100:.2f}%")
    print(f"\nAngle between vectors: {angle_deg:.2f}°")
    print(f"Normalized dot product: {dot_prod:.4f}")
    print(f"Euclidean distance between vectors: {norm_diff:.4f}")

# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    # Example vectors
    A = np.array([0.5, 0.8])  # Reference vector
    B = np.array([0.55, 0.75])  # Approximate vector
    
    # Visualize the vectors
    plot_vector_comparison(A, B, 'fig_ch5_vec_diff.pdf')
    
    # Calculate and print comparison metrics
    compare_two_vectors(A, B)