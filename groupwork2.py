import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
Lidar_data_test = pd.read_csv("Lidar_data_fusion_new.csv")
Lidar_data_nolidar = pd.read_csv("Lidar_data_lidar_new.csv")
imu_path_data = pd.read_csv("imu_path_data_rotate velocity_new_scale.csv")

# Function to calculate drifts (Euclidean distance)
def calculate_drift(data):
    start = data.iloc[0]
    end = data.iloc[-1]
    return np.sqrt((end['X'] - start['X']) ** 2 + (end['Y'] - start['Y']) ** 2)

# Function to plot data with constraints for Lidar data
def plot_with_constraints(data, color, label):
    plt.scatter([data['X'].iloc[0], data['X'].iloc[-1]],
                [data['Y'].iloc[0], data['Y'].iloc[-1]], color=color)
    for i in range(1, len(data) - 1):
        if np.linalg.norm(data.iloc[i][['X', 'Y']] - data.iloc[i - 1][['X', 'Y']]) <= 2:
            plt.plot(data['X'].iloc[i - 1:i + 1], data['Y'].iloc[i - 1:i + 1], color=color, linewidth=0.5,
                     label=label if i == 1 else "")

# Function to plot IMU data without constraints
def plot_imu_data(data, color, label):
    plt.plot(data['X'], data['Y'], color=color, label=label,linewidth=0.5,)
    plt.scatter([data['X'].iloc[0], data['X'].iloc[-1]],
                [data['Y'].iloc[0], data['Y'].iloc[-1]], color=color)

# Function to create custom legend entries for the drift values
def create_drift_legend_entry(drift_value, label):
    return plt.Line2D([], [], color='none', label=f'{label}: {drift_value:.2f}')

# Create the plot
def create_plot(mode1, mode2, mode3):
    plt.figure(figsize=(10, 6))
    legend_elements = []

    if mode1:
        drift_test = calculate_drift(Lidar_data_test)
        plot_with_constraints(Lidar_data_test, 'blue', 'IMU and Lidar fusion')
        legend_elements.append(create_drift_legend_entry(drift_test, 'Drift IMU & Lidar'))
        legend_elements.append(plt.Line2D([0], [0], color='blue', label='IMU and Lidar fusion'))

    if mode2:
        drift_nolidar = calculate_drift(Lidar_data_nolidar)
        plot_with_constraints(Lidar_data_nolidar, 'green', 'Only Lidar')
        legend_elements.append(create_drift_legend_entry(drift_nolidar, 'Drift Lidar'))
        legend_elements.append(plt.Line2D([0], [0], color='green', label='Only Lidar'))

    if mode3:
        drift_imu = calculate_drift(imu_path_data)
        plot_imu_data(imu_path_data, 'red', 'Only IMU')
        legend_elements.append(create_drift_legend_entry(drift_imu, 'Drift IMU'))
        legend_elements.append(plt.Line2D([0], [0], color='red', label='Only IMU'))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR and IMU Data Trajectory Plot')
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Example usage
create_plot(mode1=True, mode2=True, mode3=True)  # This will plot all three datasets
