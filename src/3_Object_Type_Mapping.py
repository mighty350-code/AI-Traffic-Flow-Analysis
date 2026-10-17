# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/4 12:19
@Author  : Terry CYY
@FileName: 3_Object_Type_Mapping.py
@Software: PyCharm
@Function: Map the 'type' to different categories.
           - 原始逻辑：按文件名里的数据集标记(visdrone/dronevehicle/songdo/codroneLess)
             把整数类别索引反查为类别名。
           - 新增逻辑：对 YOLO26/COCO 流水线(文件名含 hbb/obb/DJI，type 列已是字符串类别名)
             按 COCO 字符串 -> 目标分类做映射，未知类型原样保留，不再整文件跳过。
"""
import os
import pandas as pd

# ---------------- 原始：整数类别索引 -> 类别名（按数据集） ----------------
visdrone_type = {
    0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'van', 4: 'truck',
    5: 'tricycle', 6: 'bus', 7: 'motor'
}
dronevehicle_type = {
    0: 'bus', 1: 'car', 2: 'freight car', 3: 'truck', 4: 'van',
    5: 'bicycle', 6: 'tricycle'
}
songdo_type = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
codroneLess_type = {
    0: 'car', 1: 'truck', 2: 'people', 3: 'motorcycle',
    4: 'bicycle', 5: 'tricycle', 6: 'bus',
}

# ---------------- 新增：COCO 字符串类别名 -> 目标分类 ----------------
# 根据你的交通分析需求自行增删；未列出的类型保持原样(str)。
coco_type = {
    # 小型车辆
    'car': 'small-vehicle',
    'van': 'small-vehicle',
    # 大型车辆
    'truck': 'large-vehicle',
    'bus': 'large-vehicle',
    # 行人 / 骑手
    'person': 'pedestrian',
    'bicycle': 'cyclist',
    'motorcycle': 'motorcyclist',
    'motor': 'motorcyclist',
    # 水面
    'boat': 'ship',
}

# ---------------- 新增：COCO 整数索引 -> 类别名（用于 HBB 这种输出整数 index 的流水线） ----------------
# 只列出交通相关类别；完整 80 类可在此按需补充。未列出的索引 -> 'unknown'。
coco_index = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
}


def _is_coco_file(file):
    """文件名看起来是 YOLO26/COCO 流水线产出的（hbb/obb/DJI 等）。"""
    return any(k in file for k in ('hbb', 'obb', 'DJI', 'yolo', 'coco'))


def _dataset_mapping_for(file):
    """按文件名返回整数索引 -> 类别名的映射表；都不是则返回 None。"""
    if 'visdrone' in file:
        return visdrone_type
    if 'dronevehicle' in file:
        return dronevehicle_type
    if 'songdo' in file:
        return songdo_type
    if 'codroneLess' in file:
        return codroneLess_type
    return None


def map_type(file, folder='input'):
    """
    映射单个文件的 'type' 列。始终返回 DataFrame（不再整文件跳过）。
    """
    path = os.path.join(folder, file)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    data = pd.read_csv(path)
    if 'type' not in data.columns:
        print(f"'type' column not found in file {file}; skip mapping.")
        return data

    # 1) 整数类别索引分支（原 VisDrone/DroneVehicle/... 数据，或 COCO 整数索引）
    numeric = pd.to_numeric(data['type'], errors='coerce')
    if numeric.notna().all():
        mapping = _dataset_mapping_for(file)
        if mapping is not None:
            data['type'] = numeric.fillna(-1).astype(int).map(mapping).fillna('unknown')
        elif _is_coco_file(file):
            # HBB 等流水线输出的是 COCO 类别索引：先反查成名字，再归并到目标分类
            names = numeric.fillna(-1).astype(int).map(coco_index).fillna('unknown')
            data['type'] = names.map(coco_type).fillna(names)
        else:
            # 全为整数但既不是已知数据集也不是 COCO 文件：保持原整数（不强行改名）
            print(f"No integer-class mapping for {file}; keep 'type' as-is.")
        return data

    # 2) 字符串类别名分支（YOLO26/COCO: car/bus/truck/...）
    str_type = data['type'].astype(str).str.strip()
    if 'visdrone' in file or 'dronevehicle' in file or 'songdo' in file or 'codroneLess' in file:
        # 这些文件名原本是整数索引，不该出现字符串；做个告警
        print(f"Warning: {file} looks like a dataset file but 'type' is string; "
              f"treating as COCO string mapping.")
    data['type'] = str_type.map(coco_type).fillna(str_type)
    return data


# 目标路径：第二步的 GPS 结果都输出在 output/，类型重映射作用在其上
folder = 'output'


if __name__ == '__main__':
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    print(csv_files)
    for csv_file in csv_files:
        mapped_data = map_type(csv_file, folder)
        if isinstance(mapped_data, pd.DataFrame):
            output_file = os.path.join(folder, csv_file)
            mapped_data.to_csv(output_file, index=False)
            print(f"Mapped data has been saved to: {output_file}")
        else:
            print(f"Skip processing file {csv_file}")
