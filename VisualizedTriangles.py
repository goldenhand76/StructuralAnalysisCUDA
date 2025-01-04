import matplotlib.pyplot as plt
import matplotlib.patches as patches

# List of triangle vertices as tuples (x, y) for each point of a triangle
triangles = [

]

# Set up the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.xlim(-1, 40)
plt.ylim(-1, 40)

# Function to plot each triangle
for triangle in triangles:
    polygon = patches.Polygon(triangle, closed=True, edgecolor='black', facecolor='skyblue', alpha=0.7)
    ax.add_patch(polygon)

# Show the plot
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Triangles Plot")
plt.show()
