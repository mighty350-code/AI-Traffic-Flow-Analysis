# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/4 12:33 (adapted 2026/7/22 for DJI/COCO pipeline)
@Author  : Terry_CYY
@File    : 4_Multi-source_Data_Integration_and_Cleaning.py
@IDE     : PyCharm
@Function: Integrate and clean multi-source data.
           - 原版针对 dronevehicle/codroneLess（文件名带 _yolov8m）设计。
           - 适配版：文件名形如 DJI_04751_obb.csv / DJI_04751_hbb.csv，
             type 列已由 3_Object_Type_Mapping.py 映射为
             small-vehicle / large-vehicle / pedestrian / cyclist / motorcyclist / ship。
             按视频基类名(DJI_04751)合并 obb+hbb，不再整组跳过。
"""
import os
import re
import numpy as np
import pandas as pd
from collections import Counter

# Original path
input_path = 'output'
output_path = 'final'


def group_csv_files_by_prefix(folder_path):
    """
    按视频基类名(去掉 _obb / _hbb 后缀)把同一视频的多种检测结果归为一组，
    例如 DJI_04751_obb.csv 与 DJI_04751_hbb.csv -> 同一组 'DJI_04751'。
    """
    csv_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            base = re.sub(r'_(obb|hbb)\.csv$', '', filename)
            if base not in csv_files:
                csv_files[base] = []
            csv_files[base].append(os.path.join(folder_path, filename))
    return csv_files


def print_unique_frames(data):
    """Print the unique values of the frame column"""
    unique_frames = data['frame'].unique()
    print(f"Unique frames: {len(unique_frames)}")


def id_type_stat(data):
    """Check if there is a case where the type of the same id is not uniform, and return the non-uniform result"""
    inconsistent_ids = data.groupby('id')['type'].nunique()
    inconsistent_ids = inconsistent_ids[inconsistent_ids > 1].index.tolist()
    if inconsistent_ids:
        print('IDs with inconsistent types:', inconsistent_ids)
    else:
        print('No IDs with inconsistent types')


def assign_new_ids(data):
    """Reassign new ids according to the original id and type, and the starting frame"""
    # Calculate the starting frame for each (id, type) combination
    start_frames = data.groupby(['id', 'type'])['frame'].min().reset_index()
    start_frames = start_frames.rename(columns={'frame': 'start_frame'})
    # Sort by starting frame, and then by id and type for the same starting frame
    start_frames = start_frames.sort_values(
        by=['start_frame', 'id', 'type'],
        ascending=[True, True, True]
    )
    # Create a mapping from the original (id, type) to the new id
    # Assign consecutive new ids to each combination
    start_frames['new_id'] = range(len(start_frames))
    # Map the new id back to the original dataframe
    # Create a temporary key for merging
    start_frames['temp_key'] = start_frames.apply(lambda x: f"{x['id']}_{x['type']}", axis=1)
    data['temp_key'] = data.apply(lambda x: f"{x['id']}_{x['type']}", axis=1)
    # Merge dataframes
    data = pd.merge(data, start_frames[['temp_key', 'new_id']], on='temp_key', how='left')
    # Assign the new id to the id column
    data['id'] = data['new_id'] + 1

    # Clean up temporary columns
    df = data.drop(columns=['temp_key', 'new_id'])

    return df


def angular_difference_rad(a1_rad, a2_rad):
    """
    Calculate the minimum difference between two angles (considering periodicity)
    """
    diff = abs(a1_rad - a2_rad)
    return min(diff, 2 * np.pi - diff)


def correct_angle_series_bidirectional(angle_series, threshold_rad):
    """
    Correct the angle series bidirectionally to avoid sudden changes
    """
    angles = angle_series.values.copy()
    n = len(angles)
    if n < 2:  # Return directly if there is not enough data
        return pd.Series(angles, index=angle_series.index, name=angle_series.name)
    # Forward correction: find the first stable angle value and fill the previous sudden changes
    first_stable_value = angles[-1]
    idx_to_fill_until = n - 1
    for i in range(n - 1):
        if angular_difference_rad(angles[i], angles[i + 1]) <= threshold_rad:
            first_stable_value = angles[i]
            idx_to_fill_until = i
            break
    angles[:idx_to_fill_until] = first_stable_value
    # Backward correction: find the last stable angle value and fill the subsequent sudden changes
    last_stable_value = angles[0]
    idx_to_fill_from = 0
    for i in range(n - 1, 0, -1):
        if angular_difference_rad(angles[i], angles[i - 1]) <= threshold_rad:
            last_stable_value = angles[i]
            idx_to_fill_from = i
            break
    angles[idx_to_fill_from + 1:] = last_stable_value
    return pd.Series(angles, index=angle_series.index, name=angle_series.name)


def correct_df_angles(df_input, id_col='id', time_col='time', angle_col='angle', threshold_deg=60.0):
    """
    Correct the angle column in the DataFrame bidirectionally
    """
    if df_input.empty:
        df_corrected = df_input.copy()
        df_corrected[angle_col] = pd.Series(dtype=float)
        return df_corrected
    df_corrected = df_input.sort_values(by=[id_col, time_col]).copy()
    threshold_rad = np.deg2rad(threshold_deg)
    # Group by ID and correct the angles
    df_corrected[angle_col] = df_corrected.groupby(id_col, group_keys=False)[angle_col].transform(
        lambda x: correct_angle_series_bidirectional(x, threshold_rad)
    )
    return df_corrected


def drop_fault_object(data):
    """Delete abnormal objects"""
    # 1. If the type is not in valid_types, and the total duration of the id (max time - min time) is less than 1 second, delete the id
    #    适配版词汇：pedestrian / cyclist / motorcyclist 视为短时也可接受的"慢/小目标"
    valid_types = ['pedestrian', 'cyclist', 'motorcyclist']
    # Complete all aggregation calculations in one step
    id_stats = data.groupby('id').agg(
        min_time=('time', 'min'),
        max_time=('time', 'max'),
        has_valid_type=('type', lambda x: x.isin(valid_types).any())
    ).reset_index()

    # Calculate duration
    id_stats['duration'] = id_stats['max_time'] - id_stats['min_time']

    # Filter invalid IDs
    invalid_ids = id_stats[
        (id_stats['duration'] <= 1) &
        (~id_stats['has_valid_type'])
        ]['id']

    # Delete rows corresponding to invalid IDs
    data = data[~data['id'].isin(invalid_ids)]

    # 2. If the type is not in valid_types, and the average confidence is less than 0.5, delete the id
    invalid_ids = (
        data.groupby('id')
        .filter(lambda g:
                ~g['type'].isin(valid_types).any() and
                g['confidence'][~g['confidence'].isna()].mean() < 0.5
                )['id']
        .unique()
    )
    data = data[~data['id'].isin(invalid_ids)]

    # 3. Delete short trajectories
    def calc_traj_length(group):
        """Calculate trajectory length"""
        group = group.sort_values('frame')
        coords = group[['cx_m', 'cy_m']].values
        if len(coords) < 2:
            return 0
        deltas = np.diff(coords, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        return dists.sum()

    # 适配版车辆类型：合并后输出的 small-vehicle / large-vehicle
    type_veh = ['small-vehicle', 'large-vehicle']
    df_type_veh = data[data['type'].isin(type_veh)]
    traj_lengths = df_type_veh.groupby('id').apply(calc_traj_length)
    short_ids = traj_lengths[traj_lengths < 10].index
    df_type_veh_filtered = df_type_veh[~df_type_veh['id'].isin(short_ids)]

    # 4. Correct sudden changes in angles
    df_type_veh_filtered = correct_df_angles(df_type_veh_filtered, id_col='id', time_col='time', angle_col='r_align')

    # For non-type_veh types, keep the original data
    df_other = data[~data['type'].isin(type_veh)]
    # Merge the results into the original data
    data = pd.concat([df_type_veh_filtered, df_other], ignore_index=True)

    return data


def revise_dimensions(data):
    """Revise dimensions"""
    data = data.copy()

    # Group by ID and calculate the median of w_m and h_m
    median_dimensions = data.groupby('id')[['w_m', 'h_m']].median().reset_index()
    median_dimensions.columns = ['id', 'length_med', 'width_med']
    median_dimensions['length_med'] = median_dimensions['length_med'].round(2)
    median_dimensions['width_med'] = median_dimensions['width_med'].round(2)
    data = pd.merge(data, median_dimensions, on='id', how='left')

    # Similarly, group by ID and calculate the median of w and h
    median_dimensions = data.groupby('id')[['w', 'h']].median().reset_index()
    median_dimensions.columns = ['id', 'w_med', 'h_med']
    median_dimensions['w_med'] = median_dimensions['w_med'].round(2)
    median_dimensions['h_med'] = median_dimensions['h_med'].round(2)
    data = pd.merge(data, median_dimensions, on='id', how='left')

    return data


if __name__ == "__main__":
    # Get grouped csv files (按视频基类名合并 obb+hbb)
    grouped_files = group_csv_files_by_prefix(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for group_name, files in grouped_files.items():
        combined_data = pd.DataFrame()
        for file in files:
            data = pd.read_csv(file)
            # If the data has an adjusted_r column, rename it to r_align
            if 'adjusted_r' in data.columns:
                data = data.rename(columns={'adjusted_r': 'r_align'})

            print('Processing:', file)
            id_type_stat(data)

            # 适配版：obb/hbb 已统一为 COCO 映射类别，全部保留(多源整合)
            filtered_data = data
            combined_data = pd.concat([combined_data, filtered_data], ignore_index=True)

        # 防御：整组没有任何数据则跳过，避免空 DataFrame -> groupby('id') 报错
        if combined_data.empty:
            print(f"Group {group_name} has no data; skip.")
            continue

        cols = [col for col in ['frame', 'id', 'cx', 'cy', 'w', 'h', 'r', 'confidence', 'type',
                                'w_m', 'h_m', 'time', 'speed',
                                'utm_x_smooth', 'utm_y_smooth', 'cx_m', 'cy_m',
                                'smooth_cx', 'smooth_cy', 'smooth_r',
                                'ekf_x', 'ekf_y', 'ekf_r',
                                'r_align',
                                'isReal']
                if col in combined_data.columns]

        # Keep only the required columns
        combined_data = combined_data[cols]

        combined_data = drop_fault_object(combined_data)  # Delete abnormal objects
        combined_data = assign_new_ids(combined_data)
        combined_data = revise_dimensions(combined_data)  # New length and width

        # Reorder by frame and id in ascending order
        combined_data = combined_data.sort_values(by=['frame', 'id']).reset_index(drop=True)

        # Keep two decimal places for smooth_cx and smooth_cy columns
        if 'smooth_cx' in combined_data.columns:
            combined_data['smooth_cx'] = combined_data['smooth_cx'].round(2)
            combined_data['smooth_cy'] = combined_data['smooth_cy'].round(2)
        # Keep four decimal places for smooth_r column
        if 'smooth_r' in combined_data.columns:
            combined_data['smooth_r'] = combined_data['smooth_r'].round(4)

        # Save to a new csv file
        output_file = os.path.join(output_path, f"{group_name}_Traj.csv")
        combined_data.to_csv(output_file, index=False)
        print(f"Saved integrated data to: {output_file}")
