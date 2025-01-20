import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create random data points
num_points = 20
points = np.random.rand(num_points, 3)  # Generating 3D random points

# Create a random target point
target_point = np.random.rand(1, 3)

# Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    points[:, 0], points[:, 1], points[:, 2], c="blue", label="Data Points"
)
target_scatter = ax.scatter(
    target_point[:, 0],
    target_point[:, 1],
    target_point[:, 2],
    c="magenta",
    label="Target Point",
    s=100,
    marker="x",
)
nearest_point = points[0]


# Function to update the animation
def update(frame):
    global nearest_point
    ax.clear()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", label="Data Points")
    ax.scatter(
        target_point[:, 0],
        target_point[:, 1],
        target_point[:, 2],
        c="magenta",
        label="Target Point",
        marker="x",
        s=100,
    )

    # Find the nearest point
    distances = np.linalg.norm(points - target_point, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = points[nearest_index]

    # Plot the nearest point
    ax.scatter(
        nearest_point[0],
        nearest_point[1],
        nearest_point[2],
        c="black",
        label="Nearest Point",
        marker="x",
        s=100,
    )
    # ax.text(nearest_point[0], nearest_point[1], nearest_point[2], 'Nearest Point', fontsize=8, color='black')

    ax.set_title("Finding Nearest Point")
    ax.legend()


# Create the animation
# animation = FuncAnimation(fig, update, frames=range(10), interval=1000, repeat=False)
# add axes labels
update(None)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
# color axes labels
ax.xaxis.label.set_color("red")
ax.yaxis.label.set_color("green")
ax.zaxis.label.set_color("blue")
plt.show()
