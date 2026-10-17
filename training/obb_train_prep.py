#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obb_train_prep.py
--------------------------------------------------------------------------------
把本地 DJI_0475 航拍视频 + 已有 OBB 检测标注（FLUID 管线产出的 *yolo_obb.csv）
整理成 YOLO-OBB 训练数据集，并用 yolo26n-obb.pt 微调。

为什么这么做
------------
FLUID 官方 17GB 训练集无法下载（仅图像在 6~7GB video.zip 内）。
我们用本地 DJI_0475.MP4 + 既有 OBB 检测结果（含朝向 r）作为**自训练真值**，
在 traffic-analysis 自己的数据上微调 yolo26n-obb，得到一个更贴合本项目场景的
检测模型，再跑回 1→6 管线反映到前端。

标注格式转换
------------
输入 OBB CSV 每行: frame,id,cx,cy,w,h,r,confidence,type
  cx,cy  = 旋转框中心点(像素)
  w,h    = 旋转框宽高(像素)
  r      = 旋转角(弧度, OpenCV 约定: x轴正向逆时针)
-> YOLO-OBB 标注(每框一行): class x1 y1 x2 y2 x3 y3 x4 y4  (均为归一化坐标)
  四点 = 旋转矩形四个角, 顺序: 右上, 右下, 左下, 左上

用法
----
  python obb_train_prep.py                # 抽帧+转标注+写 data.yaml
  python obb_train_prep.py --skip-extract # 仅重新生成标注(帧已抽过)
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import math
import numpy as np
import pandas as pd
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VIDEO = os.path.join(ROOT, "data", "input.mp4")

# 汇总所有 OBB 标注源
OBB_SOURCES = [
    os.path.join(ROOT, "third_party", "FLUID", "demo", "data_process", "input", "DJI_0475_yolo_obb.csv"),
    os.path.join(ROOT, "third_party", "FLUID", "demo", "data_process", "input", "DJI_0475_first500_yolo_obb.csv"),
    os.path.join(ROOT, "third_party", "FLUID", "demo", "data_process", "input", "DJI_0475_first1000_yolo_obb.csv"),
]

# 类别映射（本地数据仅有 small/large-vehicle；可在此扩充官方 7 类）
CLASSES = ["small-vehicle", "large-vehicle"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# 输出目录
DATASET = os.path.join(ROOT, "data", "obb_dataset")
IMG_TRAIN = os.path.join(DATASET, "images", "train")
IMG_VAL = os.path.join(DATASET, "images", "val")
LBL_TRAIN = os.path.join(DATASET, "labels", "train")
LBL_VAL = os.path.join(DATASET, "labels", "val")
YAML_PATH = os.path.join(DATASET, "data.yaml")


def load_all_obb():
    frames = []
    for p in OBB_SOURCES:
        if not os.path.isfile(p):
            print(f"  [skip] 缺失标注源: {p}")
            continue
        df = pd.read_csv(p)
        df = df[df["type"].isin(CLASS_IDX)]
        frames.append(df)
    if not frames:
        sys.exit("[error] 没有任何可用 OBB 标注")
    big = pd.concat(frames, ignore_index=True)
    # 按 (frame,id,frame内序号) 去重: 同一 frame+id 在多源可能重复, 用 cx,cy,w,h 近似去重
    big = big.drop_duplicates(subset=["frame", "id", "cx", "cy", "w", "h", "r"])
    big["type"] = big["type"].astype(str)
    print(f"  汇总标注: {len(big)} 框, 帧 {big.frame.min()}~{big.frame.max()}, "
          f"类型 {big.type.value_counts().to_dict()}")
    return big


def obb_corners(cx, cy, w, h, r):
    """旋转矩形四角(像素), 顺序: 右上/右下/左下/左上 (YOLO-OBB 约定)。"""
    cos, sin = math.cos(r), math.sin(r)
    # 半宽半高
    hw, hh = w / 2.0, h / 2.0
    # 局部四角 (x右,y下为正)
    local = np.array([[hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]], dtype=float)
    # 旋转 (逆时针)
    rot = np.array([[cos, -sin], [sin, cos]])
    pts = local @ rot.T + np.array([cx, cy])
    return pts  # shape (4,2)


def extract_frames(df, every=1):
    """抽取标注出现的帧为图片。返回 {frame: img_path}。"""
    needed = sorted(set(int(f) for f in df["frame"].unique()))
    needed = needed[::every]
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(LBL_VAL, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        sys.exit(f"[error] 无法打开视频: {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  视频 {W}x{H}, 总帧 {total}, 需抽 {len(needed)} 帧")

    # 按帧索引定位
    frame_map = {int(f): i for i, f in enumerate(needed)}
    got = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fnum = idx + 1  # 视频第1帧对应 frame=1
        idx += 1
        if fnum in frame_map:
            target = frame_map[fnum]
            # 训练/验证划分: 每 5 帧取 1 帧作 val
            is_val = (len(got) % 5 == 4)
            sub = "val" if is_val else "train"
            name = f"frame_{fnum:06d}.jpg"
            img_dir = IMG_VAL if is_val else IMG_TRAIN
            lbl_dir = LBL_VAL if is_val else LBL_TRAIN
            cv2.imwrite(os.path.join(img_dir, name), frame)
            got[fnum] = (name, sub, W, H)
            del frame_map[fnum]
            if not frame_map:
                break
    cap.release()
    print(f"  实际抽出 {len(got)} 帧图片")
    return got, W, H


def write_labels(df, got):
    per_img = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        if f not in got:
            continue
        name, sub, W, H = got[f]
        base = name[:-4]
        corners = obb_corners(row["cx"], row["cy"], row["w"], row["h"], row["r"])
        cls = CLASS_IDX[row["type"]]
        line = [cls] + [f"{x/W:.6f} {y/H:.6f}" for x, y in corners]
        per_img.setdefault((sub, base), []).append(" ".join(str(v) for v in line))
    for (sub, base), lines in per_img.items():
        lbl_dir = LBL_VAL if sub == "val" else LBL_TRAIN
        with open(os.path.join(lbl_dir, base + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"  写出标注文件 {sum(len(v) for v in per_img.values())} 框 / {len(per_img)} 图")


def write_yaml(n_train, n_val):
    content = f"""# traffic-analysis OBB 自训练数据集 (本地 DJI_0475)
# path 用相对路径, 整个 obb_dataset 文件夹可拷贝到任意机器(含服务器)直接训练
path: .
train: images/train
val: images/val

nc: {len(CLASSES)}
names: [{', '.join(repr(c) for c in CLASSES)}]

# 训练集统计(由 obb_train_prep.py 生成)
n_train_images: {n_train}
n_val_images: {n_val}
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  写 data.yaml: {YAML_PATH}")


def main():
    ap = argparse.ArgumentParser(description="本地视频+OBB标注 -> YOLO-OBB 训练集")
    ap.add_argument("--every", type=int, default=1, help="每隔 N 帧抽一张(默认1)")
    ap.add_argument("--skip-extract", action="store_true", help="不重新抽帧(图片已存在)")
    ap.add_argument("--keep", action="store_true", help="保留已抽取图片不清理")
    args = ap.parse_args()

    print("[*] 加载 OBB 标注")
    df = load_all_obb()

    if args.skip_extract and os.path.isdir(IMG_TRAIN):
        # 仅重建标注: 从已有图片名反推 frame
        got = {}
        for sub, idir in (("train", IMG_TRAIN), ("val", IMG_VAL)):
            if not os.path.isdir(idir):
                continue
            for fn in os.listdir(idir):
                if fn.endswith(".jpg"):
                    fnum = int(fn[len("frame_"):-4])
                    got[fnum] = (fn, sub, 2688, 1512)
        print(f"  复用已抽 {len(got)} 帧")
    else:
        if not args.keep and os.path.isdir(DATASET):
            shutil.rmtree(DATASET)
        print("[*] 抽取视频帧")
        got, W, H = extract_frames(df, every=args.every)
        # 用真实 W,H 重写(extract 内已用真实值)

    print("[*] 生成 OBB 标注")
    write_labels(df, got)
    n_train = len(os.listdir(IMG_TRAIN)) if os.path.isdir(IMG_TRAIN) else 0
    n_val = len(os.listdir(IMG_VAL)) if os.path.isdir(IMG_VAL) else 0
    write_yaml(n_train, n_val)
    print(f"[*] 完成. 训练图 {n_train}, 验证图 {n_val}")
    print(f"    下一步: python obb_train.py  (或 yolo obb train ...)")


if __name__ == "__main__":
    main()
