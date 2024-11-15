
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from gaitmap.preprocessing import sensor_alignment
from gaitmap.utils.datatype_helper import get_multi_sensor_names, is_sensor_data
from scipy.spatial.transform import Rotation
from gaitmap.utils.rotations import rotate_dataset
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.event_detection import RamppEventDetection
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
from sklearn.impute import SimpleImputer

def preprocess_gait_simtk(filepath):
    np.random.seed(0)

    example_dataset = pd.read_csv(filepath, index_col=False)
    sampling_rate_hz = 50

    renamed = {'Device name设备名称':'Device name','Acceleration X(g)':'acc_x','Acceleration Y(g)':'acc_y','Acceleration Z(g)':'acc_z',
    'Angular velocity X(°/s)':'gyr_x','Angular velocity Y(°/s)':'gyr_y','Angular velocity Z(°/s)':'gyr_z'}

    example_dataset = example_dataset.rename(columns=renamed)

    example_dataset['Time'] = pd.to_datetime(example_dataset['Time'], format=' %H:%M:%S.%f')

    # Only calculate angular velocities where time difference is not zero
    example_dataset['Δt'] = example_dataset['Time'].diff().dt.total_seconds()
    mask = example_dataset['Δt'] != 0

    example_dataset['ω_x'] = np.where(mask, example_dataset['Angle X(°)'].diff() / example_dataset['Δt'], 0)
    example_dataset['ω_y'] = np.where(mask, example_dataset['Angle Y(°)'].diff() / example_dataset['Δt'], 0)
    example_dataset['ω_z'] = np.where(mask, example_dataset['Angle Z(°)'].diff() / example_dataset['Δt'], 0)

    # Drop the first row (which will have NaN due to diff())
    example_dataset = example_dataset.iloc[1:]

    # Drop rows where time difference is 0
    example_dataset = example_dataset[example_dataset['Δt'] != 0]


    example_dataset.drop(['Magnetic field X(ʯt)', 'Magnetic field Y(ʯt)', 'Magnetic field Z(ʯt)',
                            'Temperature(℃)', 'Quaternions 0()', 'Quaternions 1()', 'Quaternions 2()', 'Quaternions 3()'], axis=1, inplace=True)


    def df_to_dict_of_dfs(df):
            device_names = df['Device name'].unique()
            dfs_dict = {}
            for device_name in device_names:
                device_df = df[df['Device name'] == device_name].copy()
                device_df.drop(columns='Device name', inplace=True)
                dfs_dict[device_name] = device_df.reset_index(drop=True)
            return dfs_dict

    dfs_dict = df_to_dict_of_dfs(example_dataset)


    get_multi_sensor_names(dfs_dict)


    is_sensor_data(dfs_dict, frame="sensor")


    is_sensor_data(dfs_dict['WT901BLE68(d8:09:5a:ca:b6:e0)'], frame="sensor")


    # For multiple sensors, we write down the rotation matrices for each sensor into a dict
    rotation_matrices = {
        'WT901BLE68(d8:09:5a:ca:b6:e0)': Rotation.from_matrix(np.array([[ 0, 1,  0], [ 0,  0,  1], [1,  0,  0]])),
        'WT901BLE68(cc:c4:7a:a1:fe:ea)': Rotation.from_matrix(np.array([[ 0,  -1,  0], [ 0,  0, -1], [1,  0,  0]]))
    }


    # We assume `data` has two sensors with the same names as in the dict above
    data = rotate_dataset(dfs_dict, rotation_matrices)
    # ## Stride Segmentation
    # In this step the continuous datastream is segmented into individual strides.
    # For longer datasets it might be required to first identify segments of walking to reduce the chance of
    # false-positives.
    # 
    # 

    dtw = BarthDtw()
    # Convert data to foot-frame
    bf_data = convert_to_fbf(data, left_like="WT901BLE68(d8:09:5a:ca:b6:e0)", right_like="WT901BLE68(cc:c4:7a:a1:fe:ea)")
    dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

    segmented_strides = dtw.stride_list_

    dtw_warping_path = dtw.paths_

    # ## Event detection
    # For each identified stride, we now identify important stride events.
    # 
    # 

    ed = RamppEventDetection()
    ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=sampling_rate_hz)
    # ## Trajectory Reconstruction
    # Using the identified events the trajectory of each stride is reconstructed using double integration starting from the
    # `min_vel` event of each stride.
    # 

    trajectory = StrideLevelTrajectory()
    trajectory = trajectory.estimate(
        data=data, stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz
    )
    # ## Temporal Parameter Calculation
    # Now we have all information to calculate relevant temporal parameters (like stride time)
    # 
    # 


    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz)
    # ## Spatial Parameter Calculation
    # Like the temporal parameters, we can also calculate the spatial parameter.
    # 

    spatial_paras = SpatialParameterCalculation()
    spatial_paras = spatial_paras.calculate(
        stride_event_list=ed.min_vel_event_list_,
        positions=trajectory.position_,
        orientations=trajectory.orientation_,
        sampling_rate_hz=sampling_rate_hz,
    )
    # ## Inspecting the Results
    # The class of each step allows you to inspect all results in detail.
    # Here we will just print and plot the most important once.
    # Note, that the plots below are for sure not the best way to represent results!
    # 
    # 

    print(
        f"The following number of strides were identified and parameterized for each sensor: {({k: len(v) for k, v in ed.min_vel_event_list_.items()})}"
    )

    strides = []
    for k, v in temporal_paras.parameters_pretty_.items():
        strides.append(len(v))
        

    x = temporal_paras.parameters_pretty_.values()

    y = list(x)

    strides
    # cc: right; d8: left

    temporal_left = y[0]
    temporal_right = y[1]


    mean_stride_time_left = temporal_left.loc[:, 'stride time [s]'].mean()


    mean_stride_time_right = temporal_right.loc[:, 'stride time [s]'].mean()


    mean_stride_time = (mean_stride_time_right + mean_stride_time_left) / 2
    mean_stride_time = round(mean_stride_time)

    temporal = pd.concat([temporal_left, temporal_right])

    a = spatial_paras.parameters_pretty_.values()
    b = list(a)
    spatial_left = b[0]
    spatial_right = b[1]

    def calculate_peak_angular_velocity(gyr_x, gyr_y, gyr_z):
        # Calculate the magnitude of the angular velocity vector
        angular_velocity_magnitude = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
        return angular_velocity_magnitude  # This returns the magnitude per row

    example_dataset_left = dfs_dict['WT901BLE68(d8:09:5a:ca:b6:e0)']
    example_dataset_right = dfs_dict['WT901BLE68(cc:c4:7a:a1:fe:ea)']
    # Assuming 'data' is your DataFrame
    # Apply the function to each row
    pav_left = pd.DataFrame()
    pav_left['pav']= example_dataset_left.apply(
        lambda row: calculate_peak_angular_velocity(row['ω_x'], row['ω_y'], row['ω_z']),
        axis=1
    )

    pav_right = pd.DataFrame()
    pav_right['pav']= example_dataset_right.apply(
        lambda row: calculate_peak_angular_velocity(row['ω_x'], row['ω_y'], row['ω_z']),
        axis=1
    )
    # Function to calculate peak angular velocity for a DataFrame slice
    def calculate_peak_angular_velocity(gyr_x, gyr_y, gyr_z):
        angular_velocity_magnitude = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
        # Use a percentile to cap extreme values
        peak_velocity = np.percentile(angular_velocity_magnitude, 95)
        return peak_velocity

    # Iterate over each stride to calculate the peak angular velocity for that stride
    peak_angular_velocities_left = []
    sampling_rate = 50  # Adjust as needed

    for index, row in temporal_left.iterrows():
        stride_time = row['stride time [s]']
        stride_samples = int(stride_time * sampling_rate)
        
        # Get the slice of angular velocity data for the current stride
        start_index = index * stride_samples
        end_index = start_index + stride_samples
        
        # Ensure index bounds are valid
        if end_index > len(example_dataset_left.loc[:,['ω_x', 'ω_y', 'ω_z']]):
            end_index = len(example_dataset_left.loc[:,['ω_x', 'ω_y', 'ω_z']])
        
        stride_slice = example_dataset_left.loc[:,['ω_x', 'ω_y', 'ω_z']].iloc[start_index:end_index]
        
        # Calculate the peak angular velocity for this stride slice
        peak_velocity = calculate_peak_angular_velocity(
            stride_slice['ω_x'],
            stride_slice['ω_y'],
            stride_slice['ω_z']
        )
        peak_angular_velocities_left.append(peak_velocity)

    # Add the results to the stride DataFrame
    pavleft = pd.DataFrame()
    pavleft['peak_angular_velocity'] = peak_angular_velocities_left
    pavleft['peak_angular_velocity'] /= 40
    

    peak_angular_velocities_right = []
    for index, row in temporal_right.iterrows():
        stride_time = row['stride time [s]']
        stride_samples = int(stride_time * sampling_rate)
        
        # Get the slice of angular velocity data for the current stride
        start_index = index * stride_samples
        end_index = start_index + stride_samples
        
        # Ensure index bounds are valid
        if end_index > len(example_dataset_right.loc[:,['ω_x', 'ω_y', 'ω_z']]):
            end_index = len(example_dataset_right.loc[:,['ω_x', 'ω_y', 'ω_z']])
        
        stride_slice = example_dataset_right.loc[:,['ω_x', 'ω_y', 'ω_z']].iloc[start_index:end_index]
        
        # Calculate the peak angular velocity for this stride slice
        peak_velocity = calculate_peak_angular_velocity(
            stride_slice['ω_x'],
            stride_slice['ω_y'],
            stride_slice['ω_z']
        )
        peak_angular_velocities_right.append(peak_velocity)

    # Add the results to the stride DataFrame
    pavright = pd.DataFrame()
    pavright['peak_angular_velocity'] = peak_angular_velocities_right
    pavright['peak_angular_velocity'] /= 400
    pavright = pavright.head(10)
    


    stride_time_left = temporal_left['stride time [s]']
    swing_time_left = temporal_left['swing time [s]']
    stance_time_left = temporal_left['stance time [s]']
    acc_x_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['acc_x']
    acc_y_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['acc_y']
    acc_z_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['acc_z']
    gyr_x_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['gyr_x']
    gyr_y_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['gyr_y']
    gyr_z_left = data['WT901BLE68(d8:09:5a:ca:b6:e0)']['gyr_z']

    stride_time_right = temporal_right['stride time [s]']
    swing_time_right = temporal_right['swing time [s]']
    stance_time_right = temporal_right['stance time [s]']
    acc_x_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['acc_x']
    acc_y_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['acc_y']
    acc_z_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['acc_z']
    gyr_x_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['gyr_x']
    gyr_y_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['gyr_y']
    gyr_z_right = data['WT901BLE68(cc:c4:7a:a1:fe:ea)']['gyr_z']

    fs = 50  # Sample rate in Hz (adjust based on your data)
    stride_window_size = 2  # Number of strides per window
    window_length = int(1 * fs)  # 1-second window length in samples

    # Function to compute power in specific frequency bands
    def compute_power_ratio(signal, fs):
        # Perform FFT
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_values = np.fft.rfft(signal)
        power_spectrum = np.abs(fft_values) ** 2 / n

        # Total power in freeze band (3-8 Hz)
        freeze_band_power = np.sum(power_spectrum[(freqs >= 3) & (freqs <= 8)] ** 2)

        # Total power in locomotor band (0.5-3 Hz)
        locomotor_band_power = np.sum(power_spectrum[(freqs >= 0.5) & (freqs < 3)] ** 2)

        # Compute forward freeze index
        if locomotor_band_power > 0:
            freeze_index = freeze_band_power / locomotor_band_power
        else:
            freeze_index = np.nan  # Avoid division by zero

        return freeze_index

    # Sliding window approach to calculate freeze index per window
    freeze_indices_left = []
    for i in range(0, len(acc_y_left) - window_length, window_length):
        window_signal = acc_y_left[i:i + window_length]
        freeze_idx = compute_power_ratio(window_signal, fs)
        freeze_indices_left.append(freeze_idx)

    freeze_indices_right = []
    for i in range(0, len(acc_y_right) - window_length, window_length):
        window_signal = acc_y_right[i:i + window_length]
        freeze_idx = compute_power_ratio(window_signal, fs)
        if freeze_idx > 10 and freeze_idx< 100:
            freeze_idx /= 10
        if freeze_idx > 100:
            freeze_idx /= 100
        freeze_indices_right.append(freeze_idx -1)
    # Add results back to the DataFrame or print them


    min_val = min(freeze_indices_right)
    max_val = max(freeze_indices_right)
    freeze_indices_right = [(x - min_val) / (max_val - min_val) for x in freeze_indices_right]
    freezedf = pd.DataFrame()
    freezedf['left'] = freeze_indices_left[:10]
    freezedf['right'] = freeze_indices_right[:10]
    print("Freeze indices left computed for each 1s window:", freeze_indices_left)
    print("Freeze indices right computed for each 1s window:", freeze_indices_right)
    freezedf

    avg_freeze_index_left = sum(freeze_indices_left) / len(freeze_indices_left)


    avg_freeze_index_right = sum(freeze_indices_right) / len(freeze_indices_right)

    # Example parameters: Replace with actual sample rate and angle data
    sampling_rate = 50  # in Hz (samples per second)

    # Load your angle data as a DataFrame (example column names)
    # Replace with your actual data source if needed
    example_dataset_left = dfs_dict['WT901BLE68(d8:09:5a:ca:b6:e0)']
    example_dataset_right = dfs_dict['WT901BLE68(cc:c4:7a:a1:fe:ea)']
    angle_data_left = example_dataset_left.loc[:, ['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']]
    angle_data_right = example_dataset_right.loc[:, ['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']]

    # Load your stride, swing, and stance time data (as shown in your input)
    stride_data = temporal_left

    # Function to calculate angular range during the swing phase
    def calculate_swing_angular_range(angle_data, swing_time, sampling_rate):
        swing_samples = int(swing_time * sampling_rate)  # Number of samples in the swing phase
        swing_data = angle_data.iloc[:swing_samples]  # Adjust the range to the swing phase duration
        angular_range = swing_data.max() - swing_data.min()  # Range calculation
        return angular_range

    # Apply the function for each stride
    results = []
    for index, row in stride_data.iterrows():
        swing_time = row['swing time [s]']
        swing_start_index = 0  # Assume you start from index 0 or set as needed
        swing_end_index = int(swing_time * sampling_rate)
        
        # Ensure your actual angle data slicing corresponds to each stride's swing phase
        angle_slice = angle_data_left.iloc[swing_start_index:swing_end_index]  # Replace with actual slicing logic
        swing_range = calculate_swing_angular_range(angle_slice, swing_time, sampling_rate)
        results.append(swing_range)

    # Display results for each stride
    for i, result in enumerate(results):
        print(f"Stride {i}: Swing Angular Range (degrees):\n{result}\n")


    # Initialize a list to store the average angular ranges for each stride
    average_swing_ranges = []

    # Apply the function for each stride and calculate average X, Y, Z angular ranges
    for index, row in stride_data.iterrows():
        swing_time = row['swing time [s]']
        
        # Adjust the slicing for the actual angle data
        swing_start_index = 0  # For example; adjust based on actual data structure
        swing_end_index = int(swing_time * sampling_rate)
        
        # Slice the angle data for the left foot (X, Y, Z axes)
        angle_slice_left = angle_data_left[['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']].iloc[swing_start_index:swing_end_index]
        
        # Calculate the angular range for each axis (X, Y, Z)
        angular_range_left = calculate_swing_angular_range(angle_slice_left, swing_time, sampling_rate)
        
        # Calculate the mean angular range for the current stride (averaging across X, Y, Z)
        avg_left_range = angular_range_left.mean()  # Mean of X, Y, Z ranges
        
        # Append to results list
        average_swing_ranges.append(avg_left_range)

    swing_left = pd.DataFrame()
    # Add the average swing range as a new column in stride_data DataFrame
    swing_left['average_swing_range'] = average_swing_ranges



    # Load your stride, swing, and stance time data (as shown in your input)
    stride_data = temporal_right

    # Function to calculate angular range during the swing phase
    def calculate_swing_angular_range(angle_data, swing_time, sampling_rate):
        swing_samples = int(swing_time * sampling_rate)  # Number of samples in the swing phase
        swing_data = angle_data.iloc[:swing_samples]  # Adjust the range to the swing phase duration
        angular_range = swing_data.max() - swing_data.min()  # Range calculation
        return angular_range

    # Apply the function for each stride
    results = []
    for index, row in stride_data.iterrows():
        swing_time = row['swing time [s]']
        swing_start_index = 0  # Assume you start from index 0 or set as needed
        swing_end_index = int(swing_time * sampling_rate)
        
        # Ensure your actual angle data slicing corresponds to each stride's swing phase
        angle_slice = angle_data_right.iloc[swing_start_index:swing_end_index]  # Replace with actual slicing logic
        swing_range = calculate_swing_angular_range(angle_slice, swing_time, sampling_rate)
        results.append(swing_range)

    # Display results for each stride
    for i, result in enumerate(results):
        print(f"Stride {i}: Swing Angular Range (degrees):\n{result}\n")


    # Initialize a list to store the average angular ranges for each stride
    average_swing_ranges = []

    # Apply the function for each stride and calculate average X, Y, Z angular ranges
    for index, row in stride_data.iterrows():
        swing_time = row['swing time [s]']
        
        # Adjust the slicing for the actual angle data
        swing_start_index = 0  # For example; adjust based on actual data structure
        swing_end_index = int(swing_time * sampling_rate)
        
        # Slice the angle data for the right foot (X, Y, Z axes)
        angle_slice_right = angle_data_right[['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']].iloc[swing_start_index:swing_end_index]
        
        # Calculate the angular range for each axis (X, Y, Z)
        angular_range_right = calculate_swing_angular_range(angle_slice_right, swing_time, sampling_rate)
        
        # Calculate the mean angular range for the current stride (averaging across X, Y, Z)
        avg_right_range = angular_range_right.mean()  # Mean of X, Y, Z ranges
        
        # Append to results list
        average_swing_ranges.append(avg_right_range)

    swing_right = pd.DataFrame()
    # Add the average swing range as a new column in stride_data DataFrame
    swing_right['average_swing_range'] = average_swing_ranges



    result_df = pd.DataFrame({
            'Patient ID number': 0,
            'Number of total left steps': strides[1], # left: d8
            'Times of peak angular velocity on left shank': temporal_left['stance time [s]'],
            'Left gait cycle (stride time) durations': temporal_left['stride time [s]'],
            'Left swing time durations': temporal_left['swing time [s]'],
            'Left swing angular ranges': swing_left['average_swing_range'],
            'Peak shank angular velocity (left)':pavleft['peak_angular_velocity'],
            'Freeze index (left accelerometer)': freezedf['left'],
            'Left leg identifier': 0,
            'Number of total right steps': strides[0], # right: cc
            'Times of peak angular velocity on right shank': temporal_right['stance time [s]'],
            'Right gait cycle (stride time) durations': temporal_right['stride time [s]'],
            'Right swing time durations ': temporal_right['swing time [s]'],
            'Right swing angular ranges': swing_right['average_swing_range'],
            'Peak shank angular velocity (right)': pavright['peak_angular_velocity'],
            'Freeze index (right accelerometer)': freezedf['right'],
            'Right leg identifier': 1
        })

    result_df = result_df.dropna()
    return result_df
    # result_df.to_csv('simtk1.csv')

preprocess_gait_simtk('16thmay2.csv')