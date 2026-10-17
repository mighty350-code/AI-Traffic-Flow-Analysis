#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obb_train.py
--------------------------------------------------------------------------------
用本地 DJI_0475 自训练数据集微调 yolo26n-obb 模型。

前置: 先运行 obb_train_prep.py 生成 ./obb_dataset/data.yaml
模型:  models/yolo26n-obb.pt (Ultralytics OBB 预训练权重)
输出:  data/obb_dataset/runs/ 以及 models/ta_obb.pt

说明
----
本地标注为 FLUID 管线产出的 OBB 检测结果(含朝向 r), 作为自训练真值。
数据集较小(572 训练图), 故使用较强数据增强 + 适中 epochs, 以贴合本项目场景。

用法
----
  python obb_train.py
  python obb_train.py --epochs 80 --imgsz 640 --batch 8
"""
from __future__ import annotations
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_YAML = os.path.join(ROOT, "data", "obb_dataset", "data.yaml")
PRETRAIN = os.path.join(ROOT, "models", "yolo26n-obb.pt")  # nano: CPU 可训
MODELS_DIR = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser(description="微调 yolo26n-obb (本地 DJI_0475 自训练)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")  # 空=自动(cpu/cuda)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pretrain", default=PRETRAIN)
    args = ap.parse_args()

    if not os.path.isfile(DATA_YAML):
        sys.exit(f"[error] 找不到 data.yaml: {DATA_YAML}\n请先运行 obb_train_prep.py")
    if not os.path.isfile(args.pretrain):
        sys.exit(f"[error] 找不到预训练权重: {args.pretrain}")

    from ultralytics import YOLO

    print(f"[*] 加载预训练: {args.pretrain}")
    model = YOLO(args.pretrain)

    print(f"[*] 开始训练: epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    results = model.train(
        data=DATA_YAML,
        task="obb",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name="ta_dji0475_obb",
        project=os.path.join(ROOT, "data", "obb_dataset", "runs"),
        exist_ok=True,
        # 小数据集增强
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        scale=0.5, translate=0.1, shear=0.0,
        perspective=0.0005,
        degrees=0.0,            # OBB 不做旋转增强(角度已编码在框内)
        cos_lr=True,
        patience=15,
        save_period=1,          # 每 epoch 存一次, 中断也能拿到 last.pt
        verbose=True,
    )

    # 导出权重到 models/ (优先 best, 退化为 last)
    run_dir = os.path.join(ROOT, "data", "obb_dataset", "runs", "ta_dji0475_obb", "weights")
    best = os.path.join(run_dir, "best.pt")
    last = os.path.join(run_dir, "last.pt")
    src = best if os.path.isfile(best) else last
    if os.path.isfile(src):
        os.makedirs(MODELS_DIR, exist_ok=True)
        dst = os.path.join(MODELS_DIR, "ta_obb.pt")
        shutil_copy(src, dst)
        print(f"[*] 已导出权重 -> {dst}  ({'best' if src==best else 'last'})")

    # 在验证集上评估
    print("[*] 验证集评估:")
    model.val(data=DATA_YAML, split="val", imgsz=args.imgsz)


def shutil_copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
