import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


objects_names = ["parallelogram", "square", "star"]
p_o_graph = np.load("./task/baseline_votes_parallelogram_1.npy")


print(-np.sum(p_o_graph, axis=0))
for i, column in enumerate(p_o_graph.T):
    # Create the interpolation function (cubic spline interpolation)
    interpolation_function = interp1d(np.arange(1, 11), column, kind="cubic")

    # Generate new x-coordinates for a smooth curve
    smooth_x = np.linspace(1, 10, 100)

    # Calculate interpolated y-coordinates for the new x-coordinates
    smooth_y = interpolation_function(smooth_x)
    plt.plot(smooth_x, smooth_y, label=objects_names[i])
plt.title(f"Parallelogram experiment")
plt.ylabel("object weighted probability")
plt.xlabel("number of touches")
plt.legend()
plt.show()
