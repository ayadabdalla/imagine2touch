import numpy as np

elements = [1.1, 2.2, 3.3, 4.4, 5.5]  # replace with indices of points
probabilities = [0.2, 0.5, 0.1, 0.1, 0.1]  # replace with weights of points
print(np.random.choice(elements, 5, p=probabilities, replace=False))
