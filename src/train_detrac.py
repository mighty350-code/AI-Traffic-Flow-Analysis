"""
在 UA-DETRAC 交通数据集上微调 YOLOv8，进一步提升交通场景下的车辆识别精度。

UA-DETRAC 简介（公开学术数据集，无需登录）：
  - 包含约 8 万多帧道路监控图像
  - 车辆类别：car（小轿车）、van（面包车）、bus（公交车）、others（其他）
  - 标注格式：每个序列一个 XML，框为 (cx, cy, w, h) + vehicle_type

使用步骤：
  1) 下载并解压 UA-DETRAC 到 data/UA-DETRAC（InspectedImages/ 与 annotations/）
     官方地址：https://detrac-db.rit.albany.edu/download
  2) 运行本脚本完成格式转换 + 训练：
        python src/train_detrac.py --data data/UA-DETRAC --epochs 50
  3) 训练产物在 outputs/train_detrac/，best.pt 可直接用于 detect_vehicles.py

注意：
  - 全量训练建议在 GPU 上进行；纯 CPU 训练会非常慢，可先用 --sample 仅取少量序列做冒烟测试。
  - 数据集约 8GB+，请确保磁盘空间充足（本机 C 盘仅 0.2GB，务必放到 E 盘）。
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# UA-DETRAC 类别 -> YOLO 类别索引
CLASSES = {"car": 0, "van": 1, "bus": 2, "others": 3}


def convert_with_size(data_root: Path, out_root: Path, sample=False):
    """完整转换（使用图像真实尺寸做归一化）。需要 opencv。"""
    import cv2

    img_root = data_root / "InspectedImages"
    ann_root = data_root / "annotations"
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    seqs = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    if sample:
        seqs = seqs[:2]

    for seq in seqs:
        img_dir = img_root / seq
        ann_file = ann_root / f"{seq}.xml"
        if not ann_file.exists():
            continue
        tree = ET.parse(ann_file)
        root = tree.getroot()
        frame_targets = {}
        for frame in root.iter("frame"):
            fid = int(frame.get("num"))
            for t in frame.iter("target"):
                box = t.find("box")
                attr = t.find("attribute")
                if box is None or attr is None:
                    continue
                vtype = attr.text.strip().lower()
                if vtype not in CLASSES:
                    continue
                cx, cy, w, h = (
                    float(box.get("cx")),
                    float(box.get("cy")),
                    float(box.get("w")),
                    float(box.get("h")),
                )
                frame_targets.setdefault(fid, []).append((vtype, cx, cy, w, h))

        for img_path in sorted(img_dir.glob("*.jpg")):
            fid = int(img_path.stem.replace("img", ""))
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            dst_img = out_root / "images" / f"{seq}_{img_path.name}"
            shutil.copy(img_path, dst_img)
            label_path = out_root / "labels" / f"{seq}_{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for vtype, cx, cy, w, h in frame_targets.get(fid, []):
                    xc = (cx) / iw
                    yc = (cy) / ih
                    nw = w / iw
                    nh = h / ih
                    f.write(
                        f"{CLASSES[vtype]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
                    )
    print(f"[convert] 完成，输出目录: {out_root}")


def make_yaml(yolo_root: Path):
    yaml_path = yolo_root / "dataset.yaml"
    data = {
        "path": str(yolo_root),
        "train": "images",
        "val": "images",
        "names": {v: k for k, v in CLASSES.items()},
    }
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return yaml_path


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/UA-DETRAC")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--sample", action="store_true", help="仅用前 2 个序列做冒烟测试")
    args = ap.parse_args()

    data_root = Path(args.data)
    yolo_root = data_root.parent / "yolo_detrac"
    convert_with_size(data_root, yolo_root, sample=args.sample)
    yaml_path = make_yaml(yolo_root)

    model = YOLO("models/yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=8 if not args.sample else 2,
        project="outputs/train_detrac",
        name="exp",
        exist_ok=True,
    )
    print("[train] 完成，最佳权重位于 outputs/train_detrac/exp/weights/best.pt")


if __name__ == "__main__":
    main()
