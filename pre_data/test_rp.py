import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def recurrence_plot(data, threshold=0.1):
    """
    Generate a recurrence plot from a time series.

    :param data: Time series data
    :param threshold: Threshold to determine recurrence
    :return: Recurrence plot
    """
    # Calculate the distance matrix
    N = len(data)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.abs(data[i] - data[j])

    # Create the recurrence plot
    recurrence_plot = np.where(distance_matrix <= threshold, 1, 0)

    return recurrence_plot

# Set a seed for reproducibility
np.random.seed(0)

# Generate 500 data points of white noise
white_noise = np.random.normal(size=500)
ecg_mat = loadmat("ecg.mat")['data']
time_points = np.linspace(1, 1000, 1000)                  # type array
time_points = time_points.tolist()                  # 转list
FHR1 = ecg_mat[0, 0:1000]

recurrence = recurrence_plot(FHR1, threshold=0.1)



plt.figure(figsize=(8, 8))
plt.imshow(recurrence, cmap='rainbow', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time')
plt.ylabel('Time')
plt.colorbar(label='Recurrence')
plt.show()