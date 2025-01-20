import numpy as np
import matplotlib.pyplot as plt

data_toys_array = np.load("star_parallelogram_square_model_7_data_array.npy")
# data_cutter_array = np.load('cutter_data_array.npy')

# Create a box plot
# bp=plt.boxplot(data_toys_array, vert=True,showfliers=False,patch_artist=True)
# # bp_cutter=plt.boxplot(data_cutter_array, vert=True,showfliers=False,patch_artist=True)
# # Set the color for boxes and medians
# box_color = 'lightblue'
# median_color = 'red'

# Customize the appearance of the boxes and medians
# for box in bp['boxes']:
#     box.set(color=box_color, linewidth=1.5)
#     box.set(facecolor=box_color)  # Fill the boxes with the chosen color

# for median in bp['medians']:
#     median.set(color=median_color, linewidth=2)

# # Set the color for boxes and medians
# box_color = 'darkgreen'
# median_color = 'yellow'

# Customize the appearance of the boxes and medians
# for box in bp_cutter['boxes']:
#     box.set(color=box_color, linewidth=1.5)
#     box.set(facecolor=box_color)  # Fill the boxes with the chosen color

# for median in bp_cutter['medians']:
#     median.set(color=median_color, linewidth=2)

# # Extract median values for each box
# medians_values = [median.get_ydata()[1] for median in bp['medians']]

# # Plot a line connecting the medians
# x_positions = np.arange(1, len(medians_values) + 1)
# plt.plot(x_positions, medians_values, color=median_color, marker='o', markersize=6, linestyle='-')

# Add labels and title
# plt.ylim(0, 0.3)
# plt.xlabel('dimensions')
# plt.ylabel('epoch tactile mean squared errors (14 epochs)')
# # plt.legend([bp["boxes"][0], bp_cutter["boxes"][0]], ['toys', 'cutter'])
# # plt.title('Unseen toys vs cutter box plot')
# plt.show()

# plot each dimension of data_toys_array axis = 1 separately across epochs and add label to each curve
for i in range(data_toys_array.shape[1]):
    # increase the contrast of the colors
    plt.plot(
        data_toys_array[:, i],
        label=str(i),
        color=(
            i / data_toys_array.shape[1],
            i / data_toys_array.shape[1],
            i / data_toys_array.shape[1],
        ),
    )
plt.xlabel("epochs")
plt.ylabel("tactile mean squared errors")
plt.legend()
plt.show()
