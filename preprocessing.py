# preprocessing.py

import numpy as np
import pandas as pd
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

def preprocess_gait_data(filepath):
    # Load the dataset
    example_dataset = pd.read_csv(filepath, index_col=False)
    
    # Rename columns
    renamed = {'Device name设备名称': 'Device name', 'Acceleration X(g)': 'acc_x', 'Acceleration Y(g)': 'acc_y', 'Acceleration Z(g)': 'acc_z',
               'Angular velocity X(°/s)': 'gyr_x', 'Angular velocity Y(°/s)': 'gyr_y', 'Angular velocity Z(°/s)': 'gyr_z'}
    example_dataset = example_dataset.rename(columns=renamed)
    
    # Drop unnecessary columns
    example_dataset.drop(['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)', 'Magnetic field X(ʯt)', 'Magnetic field Y(ʯt)', 'Magnetic field Z(ʯt)',
                          'Temperature(℃)', 'Quaternions 0()', 'Quaternions 1()', 'Quaternions 2()', 'Quaternions 3()'], axis=1, inplace=True)

    # Convert DataFrame to dictionary of DataFrames
    def df_to_dict_of_dfs(df):
        device_names = df['Device name'].unique()
        dfs_dict = {}
        for device_name in device_names:
            device_df = df[df['Device name'] == device_name].copy()
            device_df.drop(columns='Device name', inplace=True)
            dfs_dict[device_name] = device_df.reset_index(drop=True)
        return dfs_dict

    dfs_dict = df_to_dict_of_dfs(example_dataset)

    # Rotation matrices for sensor alignment
    rotation_matrices = {
        'WT901BLE68(d8:09:5a:ca:b6:e0)': Rotation.from_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
        'WT901BLE68(cc:c4:7a:a1:fe:ea)': Rotation.from_matrix(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
    }

    data = rotate_dataset(dfs_dict, rotation_matrices)

    # Stride segmentation
    dtw = BarthDtw()
    bf_data = convert_to_fbf(data, left_like="WT901BLE68(d8:09:5a:ca:b6:e0)", right_like="WT901BLE68(cc:c4:7a:a1:fe:ea)")
    dtw = dtw.segment(data=bf_data, sampling_rate_hz=50)

    # Event detection
    ed = RamppEventDetection()
    ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=50)

    # Trajectory reconstruction
    trajectory = StrideLevelTrajectory()
    trajectory = trajectory.estimate(data=data, stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=50)

    # Temporal parameter calculation
    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=50)

    # Spatial parameter calculation
    spatial_paras = SpatialParameterCalculation()
    spatial_paras = spatial_paras.calculate(stride_event_list=ed.min_vel_event_list_, positions=trajectory.position_, 
                                            orientations=trajectory.orientation_, sampling_rate_hz=50)

    # Combine results into a DataFrame
    x = temporal_paras.parameters_pretty_.values()

    
    y = list(x)

    temporal_left = y[0]
    temporal_right = y[1]

    temporal = pd.concat([temporal_left, temporal_right])

    a = spatial_paras.parameters_pretty_.values()
    b = list(a)
    spatial_left = b[0]
    spatial_right = b[1]

    columns = ['Elapsed Time(s)', 'Left Stride Interval(s)', 'Right Stride Interval(s)', 
            'Left Swing Interval(s)', 'Right Swing Interval(s)', 
            'Left Swing Interval(%)', 'Right Swing Interval(%)', 
            'Left Stance Interval(s)', 'Right Stance Interval(s)', 
            'Left Stance Interval(%)', 'Right Stance Interval(%)', 
            'Double Support Interval(s)', 'Double Support Interval(%)']

    # Create an empty DataFrame with the defined columns
    result_df = pd.DataFrame(columns=columns)

    result_df['Elapsed Time(s)'] = temporal['stride time [s]']

    # Derive required features
    result_df['Left Stride Interval(s)'] = temporal_left['stride time [s]']
    result_df['Right Stride Interval(s)'] = temporal_right['stride time [s]']

    
    imputer = SimpleImputer(strategy='mean')
    col3 = ['Elapsed Time(s)', 'Left Stride Interval(s)', 'Right Stride Interval(s)']
    # Apply the imputer to the columns of result_df
    result_df[col3] = imputer.fit_transform(result_df[col3])

    result_df['Left Swing Interval(s)'] = temporal_left['swing time [s]']
    result_df['Right Swing Interval(s)'] = temporal_right['swing time [s]']
    result_df['Left Stance Interval(s)'] = temporal_left['stance time [s]']
    result_df['Right Stance Interval(s)'] = temporal_right['stance time [s]']
    result_df['Left Swing Interval(%)'] = (result_df['Left Swing Interval(s)'] / result_df['Left Stride Interval(s)']) * 100
    result_df['Right Swing Interval(%)'] = (result_df['Right Swing Interval(s)'] / result_df['Right Stride Interval(s)']) * 100
    result_df['Left Stance Interval(%)'] = (result_df['Left Stance Interval(s)'] / result_df['Left Stride Interval(s)']) * 100
    result_df['Right Stance Interval(%)'] = (result_df['Right Stance Interval(s)'] / result_df['Right Stride Interval(s)']) * 100
    result_df['Double Support Interval(s)'] = result_df['Elapsed Time(s)'] - (result_df['Left Swing Interval(s)'] + result_df['Right Swing Interval(s)']) / 2
    result_df['Double Support Interval(%)'] = (result_df['Double Support Interval(s)'] / result_df['Elapsed Time(s)']) * 100

    
    result_df[columns[3:]] = imputer.fit_transform(result_df[columns[3:]])
    result_df.reset_index(drop=True, inplace=True)

    return result_df
