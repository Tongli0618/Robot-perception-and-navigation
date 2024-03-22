import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.integrate import cumtrapz
from scipy.signal import detrend
from scipy.spatial.transform import Rotation as R

def read_acceleration_component(file_path):
    with open(file_path, 'r') as file:
        accelerations = []
        for line in file:
            # Split the line at ':' and strip whitespace from the resulting string
            parts = line.split(':')
            if len(parts) > 1:
                number_str = parts[1].strip()
                # Only convert to float if the string is not empty
                if number_str:
                    accelerations.append(float(number_str))
        return accelerations
def read_quaternions(file_path):
    with open(file_path, 'r') as file:
        quaternions = []
        for line in file:
            # Split the line at ':' and strip whitespace from the resulting string
            parts = line.split(':')
            if len(parts) > 1:
                number_str = parts[1].strip()
                # Only convert to float if the string is not empty
                if number_str:
                    quaternions.append(float(number_str))
        return quaternions
def read_angular_velocity(file_path):
    with open(file_path, 'r') as file:
        angulars = []
        for line in file:
            # Split the line at ':' and strip whitespace from the resulting string
            parts = line.split(':')
            if len(parts) > 1:
                number_str = parts[1].strip()
                # Only convert to float if the string is not empty
                if number_str:
                    angulars.append(float(number_str))
        return angulars
def read_timestamps(file_path):
    with open(file_path, 'r') as file:
        timestamps = []
        secs, nsecs = None, None
        inside_stamp = False

        for line in file:
            if 'stamp:' in line:
                inside_stamp = True
            elif 'time_ref:' in line:
                inside_stamp = False
                secs, nsecs = None, None

            if inside_stamp:
                if 'secs' in line and 'nsecs' not in line:
                    secs = int(line.split(':')[1].strip())
                elif 'nsecs' in line:
                    nsecs = int(line.split(':')[1].strip())

                    if secs is not None and nsecs is not None:
                        timestamps.append({'secs': secs, 'nsecs': nsecs})
    return timestamps
def transform_accelerations(accelerations, transformation_matrices):
    transformed_accelerations = np.zeros_like(accelerations)
    for i in range(len(accelerations)):
            transformed_accelerations[i] = np.dot(transformation_matrices[i], accelerations[i])
    return transformed_accelerations
def integrate_motion(transformed_accelerations, time_intervals):
    velocities = np.zeros_like(transformed_accelerations)
    positions = np.zeros_like(transformed_accelerations)
    for i in range(1, len(time_intervals)):
        if np.linalg.norm(transformed_accelerations[i]) < 0.45:
            velocities[i] = 0
        else:
            velocities[i] = velocities[i-1] + transformed_accelerations[i] * time_intervals[i]
        #velocities[i] = transformed_accelerations[i] * time_intervals[i]
        positions[i] = positions[i-1] + velocities[i] * time_intervals[i]
    return velocities, positions
#option1-quaternion
def quaternion_to_rotation_matrix(quaternions):
    rotation_matrices = []
    for q in quaternions:
        q1, q2, q3, q4 = q

        r11 = q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2
        r12 = 2 * (q1 * q2 - q3 * q4)
        r13 = 2 * (q1 * q3 + q2 * q4)

        r21 = 2 * (q1 * q2 + q3 * q4)
        r22 = -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2
        r23 = 2 * (q2 * q3 - q1 * q4)

        r31 = 2 * (q1 * q3 - q2 * q4)
        r32 = 2 * (q2 * q3 + q1 * q4)
        r33 = -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2

        rotation_matrix = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        #option1
        #rotation_matrix = R.from_quat(q).as_matrix()
        #option2
        rotation_matrices.append(rotation_matrix)
    return rotation_matrices
#option2-direction cosines，book page 183
def skew_symmetric_matrix(omega):
    omega_x, omega_y, omega_z = omega
    return np.array([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ])
#option2-direction cosines，book page 183
def update_rotation_matrix(R, omega, delta_t):
    theta = omega * delta_t
    theta_magnitude = np.linalg.norm(theta)

    #if theta_magnitude == 0:
     #   return R

    S = skew_symmetric_matrix(theta)
    S_squared = np.dot(S, S)


    I = np.eye(3)
    sin_term = np.sin(theta_magnitude) / theta_magnitude
    cos_term = (1 - np.cos(theta_magnitude)) / (theta_magnitude ** 2)

    R_update = I + sin_term * S + cos_term * S_squared
    #R_update = I + S
    return np.dot(R, R_update)
#option2-direction cosines，book page 183
def calculate_rotation_matrix(angular_velocities, time_intervals):

    R = np.eye(3)
    rotation_matrices = []
    for omega, delta_t in zip(angular_velocities, time_intervals):
        R = update_rotation_matrix(R, omega, delta_t)
        rotation_matrices.append(R)
    return rotation_matrices
#option3-euler angle
def calculate_euler_angles(angular_velocities, time_intervals):
    euler_angles = np.zeros_like(angular_velocities)
    for i in range(len(time_intervals)):
        if i == 0:
            euler_angles[i] = angular_velocities[i] * time_intervals[i]
        else:
            euler_angles[i] = euler_angles[i - 1] + angular_velocities[i] * time_intervals[i]
    return euler_angles
def eulerangle_transformation_matrices(angular_velocities, time_intervals):
    rotation_matrices = []
    euler_angles = []
    euler_angles = calculate_euler_angles(angular_velocities, time_intervals)
    print(euler_angles[-1])
    transformation_matrices = []
    for angles in euler_angles:
        pitch, roll, yaw = angles

        Rx = np.array([[1, 0, 0],
                        [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])

        Ry = np.array([[math.cos(roll), 0, math.sin(roll)],
                       [0, 1, 0],
                       [-math.sin(roll), 0, math.cos(roll)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        '''
        R = [[math.cos(yaw)*math.cos(roll) - math.sin(yaw)*math.sin(pitch)*math.sin(roll), -math.sin(yaw)*math.cos(pitch), math.cos(yaw)*math.sin(roll)+math.sin(yaw)*math.sin(pitch)*math.cos(roll)],
            [math.sin(yaw)*math.cos(roll)+math.cos(yaw)*math.sin(pitch)*math.sin(roll), math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.sin(roll)-math.cos(yaw)*math.sin(pitch)*math.cos(roll)],
            [-math.cos(pitch)*math.sin(roll), math.sin(pitch),math.cos(pitch)*math.cos(roll)]]
        '''
        transformation_matrices.append(R)
    return transformation_matrices


#read acceleration
x_acc = read_acceleration_component('ascii_x_acc.txt')
y_acc = read_acceleration_component('ascii_y_acc.txt')
z_acc = np.array(read_acceleration_component('ascii_z_acc.txt'))#-9.81
#z_acc = np.array(read_acceleration_component('ascii_z_acc.txt'))
combined_acc = zip(x_acc, y_acc, z_acc)
accelerations = np.array([list(acc_tuple) for acc_tuple in combined_acc])
mean_values = np.mean(accelerations, axis=0)
accelerations = accelerations - mean_values

bg_acc = np.array([0.0021817, 0.0169198, -0.0043685])
sg_acc = np.array([-0.000327, -0.000656, 0.00012492])
accelerations = (accelerations - bg_acc) / (sg_acc+1)

#read auglar
x_av = read_angular_velocity('ascii_x_ang.txt')
y_av = read_angular_velocity('ascii_y_ang.txt')
z_av = read_angular_velocity('ascii_z_ang.txt')

combined_av = zip(x_av, y_av, z_av)
angular_velocities = np.array([list(av_tuple) for av_tuple in combined_av])
bg_avg = np.array([-7.35656E-05, 2.14421E-05, -2.38792E-05])
sg_avg = np.array([-0.993074083, -0.985698374, -0.971665299])
angular_velocities = (angular_velocities - bg_avg)#/(sg_avg+1)
#read quaternions
x_qt = read_quaternions('ascii_x_qua.txt')
y_qt = read_quaternions('ascii_y_qua.txt')
z_qt = read_quaternions('ascii_z_qua.txt')
w_qt = read_quaternions('ascii_w_qua.txt')
combined_qt = zip(x_qt,y_qt,z_qt,w_qt)
quaternions = np.array([list(qt_tuple) for qt_tuple in combined_qt])
#read timestamps
timestamps = read_timestamps('ascii_time_ref.txt')
timestamps_in_ns = np.array([ts['secs'] * 1e9 + ts['nsecs'] for ts in timestamps])
time_intervals = np.diff(timestamps_in_ns) / 1e9

#length aligned
min_length = min(len(accelerations), len(quaternions), len(time_intervals),len(angular_velocities))
accelerations = accelerations[:min_length]
angular_velocities = angular_velocities[:min_length]
quaternions = quaternions[:min_length]
time_intervals = time_intervals[:min_length]
timestamps_in_ns = timestamps_in_ns[:min_length]

#create new timestamps from 0, but it will be unnecessary if using integrate_motion function
timestamps_new = np.zeros_like(accelerations )
timestamps_new = np.insert(np.cumsum(time_intervals), 0, 0)[:(min_length)]


#Main
transformation_matrices = []
#option1-quaternion
#transformation_matrices = quaternion_to_rotation_matrix(quaternions)
#option2-direction cosines，book page 183
#transformation_matrices = calculate_rotation_matrix(angular_velocities, time_intervals)
#option3-euler angle
transformation_matrices = eulerangle_transformation_matrices(angular_velocities, time_intervals)



transformed_accelerations = transform_accelerations(accelerations, transformation_matrices)
#transformed_accelerations = np.apply_along_axis(detrend, 0, transformed_accelerations)
#option1-integral
velocities, positions = integrate_motion(transformed_accelerations, time_intervals)
#option2-cumtrapz function
#velocities = cumtrapz(transformed_accelerations, x=timestamps_new, axis=0)
#positions = cumtrapz(velocities, x=timestamps_new[:-1],  axis=0)


# Plot
'''
plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=1)
#plt.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='blue', s=100)
#plt.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],  color='red', s=100)
ax.set_title("3D Motion Trajectory")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
plt.grid(True)
plt.show()
'''
start = positions[0]
end = positions[-1]
drift = np.linalg.norm(end - start)

plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1], linewidth=1)
plt.scatter(positions[0, 0], positions[0, 1], color='blue', s=100)  # 起点
plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100)  # 终点

plt.annotate(f'Drift: {drift:.2f}', xy=(end[0], end[1]), xytext=(10,10), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title("2D Motion Trajectory(direction cosines-after calibration)")
plt.xlabel("X displacement")
plt.ylabel("Y displacement")
plt.grid(True)
plt.show()






#do not use it
def create_csv(accelerations, quaternions1, timestamps_in_ns):

    df = pd.DataFrame({
        'time': timestamps_in_ns,
        'field.orientation.x': quaternions1[:, 0],
        'field.orientation.y': quaternions1[:, 1],
        'field.orientation.z': quaternions1[:, 2],
        'field.orientation.w': quaternions1[:, 3],
        'field.linear_acceleration.x': accelerations[:, 0],
        'field.linear_acceleration.y': accelerations[:, 1],
        'field.linear_acceleration.z': accelerations[:, 2]
    })


    df.to_csv('data618.csv', index=False)


#create_csv(accelerations, quaternions1, timestamps_in_ns)