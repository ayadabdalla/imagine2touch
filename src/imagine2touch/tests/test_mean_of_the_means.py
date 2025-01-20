import numpy as np

matrix = np.array([[1, 2, 3], [4, 13, 6], [7, 9, 9]])

# Calculate mean across rows first (axis=1), then mean across columns (axis=0)
mean_of_means_1 = np.mean(np.mean(matrix, axis=1))
print(np.mean(matrix, axis=1))
print("Mean of means (rows -> columns):", mean_of_means_1)

# Calculate mean across columns first (axis=0), then mean across rows (axis=1)
mean_of_means_2 = np.mean(np.mean(matrix, axis=0))
print(np.mean(matrix, axis=0))
print("Mean of means (columns -> rows):", mean_of_means_2)
