import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def current_elements_along_square(L=1.0, dl=0.1):
    """
    Generate current elements along a square loop centered at origin
    Returns positions (3xN) and current vectors (3xN)
    """
    # One side of the square (right side)
    vec = np.arange(-L/2, L/2, dl)  # Points along one side
    N = len(vec)
    
    pos = np.zeros((3, 4*N))  # 3D positions (x,y,z)
    curr = np.zeros((3, 4*N))  # Current vectors
    
    # Right side (x = L/2, y varies, z=0)
    pos[0, :N] = L/2
    pos[1, :N] = vec
    curr[1, :N] = 1  # Current in +y direction
    
    # Top side (y = L/2, x varies, z=0)
    pos[0, N:2*N] = vec[::-1]
    pos[1, N:2*N] = L/2
    curr[0, N:2*N] = -1  # Current in -x direction
    
    # Left side (x = -L/2, y varies, z=0)
    pos[0, 2*N:3*N] = -L/2
    pos[1, 2*N:3*N] = vec[::-1]
    curr[1, 2*N:3*N] = -1  # Current in -y direction
    
    # Bottom side (y = -L/2, x varies, z=0)
    pos[0, 3*N:4*N] = vec
    pos[1, 3*N:4*N] = -L/2
    curr[0, 3*N:4*N] = 1  # Current in +x direction
    
    # Normalize current vectors
    curr = curr * (dl/L)
    return pos, curr

def current_elements_along_square_rot(L=1.0, dl=0.1):
    """
    Generate current elements by rotation (alternative implementation)
    """
    # Rotation matrix for 90 degrees around z-axis
    R90 = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])
    
    # Create one side (right side)
    vec = np.arange(-L/2, L/2, dl)
    N = len(vec)
    
    pos = np.zeros((3, 4*N))
    curr = np.zeros((3, 4*N))
    
    # Right side
    pos[0, :N] = L/2
    pos[1, :N] = vec
    curr[1, :N] = 1
    
    # Rotate to create other sides
    for i in range(1, 4):
        pos[:, i*N:(i+1)*N] = np.dot(R90, pos[:, (i-1)*N:i*N])
        curr[:, i*N:(i+1)*N] = np.dot(R90, curr[:, (i-1)*N:i*N])
    
    # Normalize current vectors
    curr = curr * (dl/L)
    return pos, curr

# Create figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Generate current elements
L = 1.0
dl = 0.1
pos, curr = current_elements_along_square(L, dl)

# Plot the square loop
ax.plot([L/2, L/2, -L/2, -L/2, L/2],
        [L/2, -L/2, -L/2, L/2, L/2],
        [0, 0, 0, 0, 0], 'k-', linewidth=2)

# Plot current elements as arrows
ax.quiver(pos[0], pos[1], pos[2],
          curr[0], curr[1], curr[2],
          color='red', length=0.1, normalize=True)

# Set equal aspect ratio
ax.set_box_aspect([1,1,1])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Current elements along square loop')

plt.tight_layout()
plt.show()