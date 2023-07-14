import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def compute_permeability_matrix(phi_x, phi_y, phi_z, B, m):
    k11 = B * ( phi_x)**m
    k22 = B * ( phi_y)**m
    k33 = B * ( phi_z)**m

    permeability_matrix = np.maximum(np.array([[k11, 0, 0],
                                               [0, k22, 0],
                                               [0, 0, k33]]), 0.0)

    return permeability_matrix

def sph2cart(r, phi, tta):
    x = r * np.sin(tta) * np.cos(phi)
    y = r * np.sin(tta) * np.sin(phi)
    z = r * np.cos(tta)
    return x, y, z
 
def ellips2cart(r, phi, tta, a, b, c):
    x = a * r * np.sin(tta) * np.cos(phi)
    y = b * r * np.sin(tta) * np.sin(phi)
    z = c * r * np.cos(tta)
    return x, y, z    

# Streamlit interface
st.title("Orthotropic Permeability Tensor Visualization")

# User input for porosity values
phi_x = st.number_input("Porosity in the x-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
phi_y = st.number_input("Porosity in the y-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.3)
phi_z = st.number_input("Porosity in the z-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.4)

# User input for base permeability constant and exponent
B = st.number_input("Base Permeability Constant", min_value=0.0, value=1.0)
m = st.number_input("Exponent", min_value=0.0, value=2.0)

# User input for colormap
cmap_name = st.selectbox("Select a color map", ['jet', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 
                      'turbo', 'nipy_spectral', 'gist_ncar', 'Pastel1', 'Pastel2',
                      'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10',
                      'tab20', 'tab20b', 'tab20c', 'twilight', 'twilight_shifted',
                      'hsv', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'binary',
                      'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',
                      'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot',
                      'gist_heat', 'copper', 'Greys', 'Purples', 'Blues', 'Greens',
                      'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
                      'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                      'viridis', 'plasma', 'inferno', 'magma', 'cividis'])

# Compute permeability matrix
permeability_matrix = compute_permeability_matrix(phi_x, phi_y, phi_z, B, m)

# Calculate the permeability values at each point
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2 * np.pi, 200)
Theta, Phi = np.meshgrid(theta, phi)
#X, Y, Z = sph2cart(1, Phi, Theta)

# Calculate the major and minor radii of the ellipsoid based on permeability values
a = np.sqrt(permeability_matrix[0, 0] / B)
b = np.sqrt(permeability_matrix[1, 1] / B)
c = np.sqrt(permeability_matrix[2, 2] / B)
X, Y, Z = ellips2cart(1, Phi, Theta,a,b,c)

#a=0.5
#b=1.0
#c=10.0
# Calculate the permeability tensor values
permeability_values = (
    permeability_matrix[0, 0] * (X/a)**2
    + permeability_matrix[1, 1] * (Y/b)**2
    + permeability_matrix[2, 2] * (1 - (X/a)**2 - (Y/b)**2)
)

# Plot permeability tensor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#cmap = 'viridis'
#norm = plt.Normalize(permeability_tensor.min(), permeability_tensor.max())
# Set the colormap and normalization
cmap = cm.get_cmap(cmap_name)
norm = plt.Normalize(0, np.max(permeability_values))
#norm = plt.Normalize(np.min(permeability_matrix), np.max(permeability_matrix))

# Calculate the major and minor radii of the ellipsoid based on permeability values
#radius_a = np.sqrt(permeability_matrix[0, 0] / B)
#radius_b = np.sqrt(permeability_matrix[1, 1] / B)
#radius_c = np.sqrt(permeability_matrix[2, 2] / B)

# Scale the coordinates with the radii
#X_scaled = X * radius_a
#Y_scaled = Y * radius_b
#Z_scaled = Z * radius_c

#[X,Y,Z]=[X_scaled, Y_scaled, Z_scaled]
ax.plot_surface(
    X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(permeability_values)),
    rstride=1, cstride=1, linewidth=0.1, antialiased=True
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#########################################################
#Code to make something look elliptic
ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
#########################################################
ax.set_title('Orthotropic Permeability Tensor')

#fig.colorbar(
#    ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(permeability_values))),
#    label='Permeability'
#)
# Add the colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label('Permeability', fontsize=12)
cbar.ax.tick_params(labelsize=10)


st.pyplot(fig)
