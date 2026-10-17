"""
在 VisDrone-DET / UAVDT 航拍数据集上微调 YOLOv8 检测头，提升航拍小目标（车辆）检测精度。
====================================================================================

核心思路（"微调检测头"）：
  - 加载 COCO 预训练的 YOLOv8(m) 权重，仅训练最后一级 **Detect 检测头**
    （冻结 backbone 0-9 + neck 10-21，只训练模块 22）。
  - 当目标数据集类别数（nc=5）与 COCO（nc=80）不同时，ultralytics 会自动用新 nc
    重建检测头（随机初始化），并保留 backbone/neck 的预训练权重——这正符合
    "微调检测头" 的语义：迁移学习、训练快、过拟合风险低。
  - 针对航拍小目标：默认高输入分辨率（imgsz=1280）+ mosaic + 尺度抖动 + 末轮关闭
    mosaic（close_mosaic）以稳定小框。

类别统一映射（兼容 detect_vehicles.py / track_vehicles.py 的车辆词表）：
  统一词表 AERIAL_CLASSES = [car, truck, bus, motorcycle, bicycle]  （索引 0-4）
  与下游脚本的 VEHICLE_CLASSES 键名完全一致，训练出的权重可直接作为 --model 使用。

数据集说明与准备：
  - VisDrone-DET（需注册下载官方包，约 1.5GB）：
      https://github.com/VisDrone/VisDrone-DET  （或 VisDrone 官网）
      解压后形如  data/VisDrone-DET-train/{images,annotations}/
  - UAVDT（GitHub，含图像与标注）：
      https://github.com/ucla-mobility/UAVDT
      解压后形如  data/UAV-benchmark-MOTD_v1.0/MOTD/<seq>/{img1,gt}/
  （本机为 CPU，仅适合 --sample 冒烟；正式训练请在带 CUDA 的机器上跑。）

用法：
  # VisDrone 正式训练（GPU 推荐）
  python src/train_aerial.py --dataset visdrone --data data/VisDrone --epochs 50 --device 0

  # 仅冒烟（CPU 友好）
  python src/train_aerial.py --dataset visdrone --data data/VisDrone --sample --epochs 3 --imgsz 320

  # 同时用 VisDrone + UAVDT（数据更多，小目标更鲁棒）
  python src/train_aerial.py --dataset both --data data --epochs 50 --device 0

  # 只做格式转换，不训练（方便先把数据准备好，再到 GPU 机器训练）
  python src/train_aerial.py --dataset visdrone --data data/VisDrone --skip-train

  # 不只训练头，而是连 neck 一起微调（小目标通常更优；仍是轻量微调）
  python src/train_aerial.py --dataset visdrone --data data/VisDrone --freeze backbone --device 0

产物：
  outputs/train_aerial/exp/weights/best.pt  —— 可直接用于 detect_vehicles.py / track_vehicles.py 的 --model
"""
import argparse
import shutil
import random
import cv2
import yaml
from pathlib import Path

# ===== 把所有缓存/配置锁定到本项目目录（E 盘），避免写入 C 盘 =====
ROOT = Path(__file__).resolve().parent.parent
_ultra_cfg = ROOT / ".ultralytics"
import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultra_cfg))
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from ultralytics import YOLO, settings
try:
    settings.update({"runs_dir": str(ROOT / "outputs")})
except Exception:
    pass

DEFAULT_MODEL = str(ROOT / "models" / "yolov8m.pt")

# 统一车辆词表（索引 = COCO 车辆词表顺序，与 detect/track 脚本完全一致）
AERIAL_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

# VisDrone-DET 类别（标注文件里是 1-based 的 category 编号）：
#   1 pedestrian 2 people 3 bicycle 4 car 5 van 6 truck 7 tricycle 8 awning-tricycle 9 bus 10 motor
# 映射到统一索引；None 表示非车辆、忽略。
VISDRONE_MAP = {
    1: None, 2: None,          # 行人/人群 -> 忽略（非车辆）
    3: 4,                       # bicycle   -> bicycle
    4: 0,                       # car       -> car
    5: 0,                       # van       -> car（COCO 无 van，并入 car）
    6: 1,                       # truck     -> truck
    7: 0,                       # tricycle  -> car（三轮机动车并入 car）
    8: 0,                       # awning-tricycle -> car
    9: 2,                       # bus       -> bus
    10: 3,                      # motor     -> motorcycle
}

# UAVDT 类别（标注文件里是 1-based 的 class id）：
#   注意：不同发行版的 bus/truck 顺序可能互换，请按你的标注核对后调整下面这行。
#   默认 1=car, 2=truck, 3=bus。
UAVDT_MAP = {
    1: 0,   # car
    2: 1,   # truck （若你的标注是 2=bus，则与 3 交换）
    3: 2,   # bus
}


# ----------------------------------------------------------------------------
# VisDrone-DET 转换
# ----------------------------------------------------------------------------
def _find_visdrone_dirs(data_root: Path):
    """兼容两种目录布局，返回 (images_dir, annotations_dir)。"""
    cand = [
        (data_root / "VisDrone-DET-train" / "images",
         data_root / "VisDrone-DET-train" / "annotations"),
        (data_root / "images", data_root / "annotations"),
        (data_root / "VisDrone-DET-train", data_root / "VisDrone-DET-train"),
    ]
    for img, ann in cand:
        if img.exists() and ann.exists():
            return img, ann
    raise FileNotFoundError(
        f"未找到 VisDrone 的 images/annotations 目录。期望布局如:\n"
        f"  {data_root}/VisDrone-DET-train/{{images,annotations}}/\n"
        f"  或 {data_root}/{{images,annotations}}/"
    )


def convert_visdrone(data_root: Path, images_dir: Path, labels_dir: Path,
                     sample=False, prefix="visdrone"):
    """转换 VisDrone-DET 为 YOLO 格式，返回 {seq_name: [img_stem,...]} 用于划分 train/val。"""
    img_root, ann_root = _find_visdrone_dirs(data_root)
    seqs = sorted({p.stem.split("_")[0] for p in ann_root.glob("*.txt")})
    if sample:
        seqs = seqs[:2]

    seq_to_imgs = {}
    for ann in sorted(ann_root.glob("*.txt")):
        seq = ann.stem.split("_")[0]
        if seq not in seqs:
            continue
        # 读取该序列所有框，按图像名归类
        per_img = {}
        with open(ann, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                try:
                    x, y, w, h = (float(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3]))
                    cat = int(parts[5])
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    continue
                tgt = VISDRONE_MAP.get(cat, None)
                if tgt is None:
                    continue
                img_stem = ann.stem  # VisDrone 标注名与图名同 stem
                per_img.setdefault(img_stem, []).append((tgt, x, y, w, h))

        for img_path in sorted(img_root.glob(f"{ann.stem}*")):
            stem = img_path.stem
            if stem not in per_img:
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            ih, iw = im.shape[:2]
            dst_img = images_dir / f"{prefix}_{img_path.name}"
            shutil.copy(img_path, dst_img)
            label_path = labels_dir / f"{prefix}_{stem}.txt"
            with open(label_path, "w", encoding="utf-8") as lf:
                for tgt, x, y, w, h in per_img[stem]:
                    xc = (x + w / 2.0) / iw
                    yc = (y + h / 2.0) / ih
                    nw = w / iw
                    nh = h / ih
                    lf.write(f"{tgt} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            seq_to_imgs.setdefault(f"{prefix}_{seq}", []).append(dst_img.stem)
    print(f"[visdrone] 转换完成：{sum(len(v) for v in seq_to_imgs.values())} 张图")
    return seq_to_imgs


# ----------------------------------------------------------------------------
# UAVDT 转换
# ----------------------------------------------------------------------------
def _find_uavdt_seqs(data_root: Path):
    cand = [
        data_root / "UAV-benchmark-MOTD_v1.0" / "MOTD",
        data_root / "MOTD",
        data_root,
    ]
    for d in cand:
        if d.exists() and any((d / s / "img1").exists() for s in
                              [p.name for p in d.iterdir() if p.is_dir()]):
            return [p for p in sorted(d.iterdir()) if (p / "img1").exists()]
    return []


def convert_uavdt(data_root: Path, images_dir: Path, labels_dir: Path,
                  sample=False, prefix="uavdt"):
    seqs = _find_uavdt_seqs(data_root)
    if not seqs:
        raise FileNotFoundError(
            f"未找到 UAVDT 序列（期望形如 .../MOTD/<seq>/img1 与 <seq>/gt/gt_<seq>.txt）。"
        )
    if sample:
        seqs = seqs[:2]

    seq_to_imgs = {}
    for seq in seqs:
        img_dir = seq / "img1"
        gt_dir = seq / "gt"
        gt_files = sorted(gt_dir.glob(f"gt_{seq.name}.txt")) if gt_dir.exists() else []
        if not gt_files:
            gt_files = sorted(gt_dir.glob("gt*.txt")) if gt_dir.exists() else []
        if not gt_files or not img_dir.exists():
            continue
        gt = gt_files[0]
        # 按帧归类
        per_frame = {}
        with open(gt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                try:
                    frame = int(parts[0])
                    x, y, w, h = (float(parts[2]), float(parts[3]),
                                  float(parts[4]), float(parts[5]))
                    ignore = int(parts[6])
                    cat = int(parts[7])
                except ValueError:
                    continue
                if ignore == 1 or w <= 0 or h <= 0:
                    continue
                tgt = UAVDT_MAP.get(cat, None)
                if tgt is None:
                    continue
                per_frame.setdefault(frame, []).append((tgt, x, y, w, h))

        for frame, boxes in per_frame.items():
            fname = f"{frame:07d}.jpg"
            img_path = img_dir / fname
            if not img_path.exists():
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            ih, iw = im.shape[:2]
            dst_img = images_dir / f"{prefix}_{seq.name}_{fname}"
            shutil.copy(img_path, dst_img)
            label_path = labels_dir / f"{prefix}_{seq.name}_{img_path.stem}.txt"
            with open(label_path, "w", encoding="utf-8") as lf:
                for tgt, x, y, w, h in boxes:
                    xc = (x + w / 2.0) / iw
                    yc = (y + h / 2.0) / ih
                    nw = w / iw
                    nh = h / ih
                    lf.write(f"{tgt} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
            seq_to_imgs.setdefault(f"{prefix}_{seq.name}", []).append(dst_img.stem)
    print(f"[uavdt] 转换完成：{sum(len(v) for v in seq_to_imgs.values())} 张图")
    return seq_to_imgs


# ----------------------------------------------------------------------------
# 划分 train/val（按序列划分，避免同场景泄漏）并写 dataset.yaml
# ----------------------------------------------------------------------------
def split_and_make_yaml(yolo_root: Path, seq_to_imgs: dict, val_frac: float):
    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"
    # 过滤掉没有对应 label 的图像
    valid_seqs = {}
    for seq, stems in seq_to_imgs.items():
        kept = [s for s in stems if (labels_dir / f"{s}.txt").exists()]
        if kept:
            valid_seqs[seq] = kept

    all_seqs = sorted(valid_seqs.keys())
    random.seed(42)
    random.shuffle(all_seqs)
    n_val = max(1, int(len(all_seqs) * val_frac))
    val_seqs = set(all_seqs[:n_val])
    train_seqs = set(all_seqs[n_val:])

    def write_list(path, seq_set):
        with open(path, "w", encoding="utf-8") as f:
            for seq in sorted(seq_set):
                for stem in valid_seqs[seq]:
                    f.write(f"{images_dir / (stem + '.jpg')}\n")

    write_list(yolo_root / "train.txt", train_seqs)
    write_list(yolo_root / "val.txt", val_seqs)
    print(f"[split] 序列 {len(all_seqs)} 个 -> train {len(train_seqs)} / val {len(val_seqs)}")

    data = {
        "path": str(yolo_root),
        "train": "train.txt",
        "val": "val.txt",
        "names": {i: c for i, c in enumerate(AERIAL_CLASSES)},
    }
    yaml_path = yolo_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return yaml_path


# ----------------------------------------------------------------------------
# 解析 freeze 模式
# ----------------------------------------------------------------------------
def resolve_freeze(mode: str):
    """返回传给 ultralytics 的 freeze 列表（模块索引）。

    YOLOv8m 模块索引：0-9 backbone, 10-21 neck, 22 Detect 检测头。
    - head     : 冻结 0-21，只训练检测头（默认，即"微调检测头"）
    - backbone : 冻结 0-9，训练 neck+head（小目标通常更优的轻量微调）
    - none     : 全部训练（完整微调）
    - <int>    : 冻结前 N 个模块
    """
    mode = (mode or "head").lower()
    if mode == "head":
        return list(range(0, 22))      # 冻结 backbone + neck
    if mode == "backbone":
        return list(range(0, 10))      # 只冻结 backbone
    if mode in ("none", "all", "full"):
        return []
    if mode.isdigit():
        return list(range(int(mode)))
    raise ValueError(f"未知 freeze 模式: {mode}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="在 VisDrone / UAVDT 航拍数据集上微调 YOLOv8 检测头（提升小目标精度）")
    ap.add_argument("--dataset", required=True, choices=["visdrone", "uavdt", "both"],
                    help="使用的数据集（both=VisDrone+UAVDT 合并训练）")
    ap.add_argument("--data", required=True,
                    help="数据集根目录（见文件头布局说明）")
    ap.add_argument("--out", default=str(ROOT / "data" / "yolo_aerial"),
                    help="YOLO 格式数据输出根目录（默认 data/yolo_aerial）")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="训练输入分辨率（默认 1280，高分辨率更利于小目标；CPU 冒烟可用 320）")
    ap.add_argument("--batch", type=int, default=16, help="批次大小（GPU 建议 16，CPU 冒烟用 2-4）")
    ap.add_argument("--freeze", default="head",
                    help="冻结策略: head(默认,只训检测头) / backbone(训neck+head) / none(全量) / <int>N")
    ap.add_argument("--device", default="cpu", help="训练设备：cpu 或 0(有GPU时)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 工作进程数（Windows CPU 建议 0）")
    ap.add_argument("--val-frac", type=float, default=0.1, help="验证集占序列比例")
    ap.add_argument("--sample", action="store_true", help="仅取前 2 个序列做冒烟测试")
    ap.add_argument("--skip-train", action="store_true", help="只做格式转换，不启动训练")
    ap.add_argument("--project", default=str(ROOT / "outputs" / "train_aerial"),
                    help="训练产物目录（绝对路径，默认 outputs/train_aerial）")
    ap.add_argument("--name", default="exp", help="训练 run 名称")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="初始权重（默认 models/yolov8m.pt）")
    args = ap.parse_args()

    data_root = Path(args.data)
    yolo_root = Path(args.out)
    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seq_to_imgs = {}
    if args.dataset in ("visdrone", "both"):
        seq_to_imgs.update(convert_visdrone(data_root, images_dir, labels_dir,
                                            sample=args.sample, prefix="visdrone"))
    if args.dataset in ("uavdt", "both"):
        seq_to_imgs.update(convert_uavdt(data_root, images_dir, labels_dir,
                                         sample=args.sample, prefix="uavdt"))

    if not seq_to_imgs:
        raise SystemExit("没有转换出任何标注数据，请检查 --data 路径与数据集布局。")

    yaml_path = split_and_make_yaml(yolo_root, seq_to_imgs, args.val_frac)
    print(f"[dataset] yaml: {yaml_path}")

    if args.skip_train:
        print("[skip-train] 仅完成格式转换，已退出。可用 GPU 机器执行训练。")
        return

    freeze_list = resolve_freeze(args.freeze)
    print(f"[train] 冻结模块索引: {freeze_list}  (head-only 应为 0-21)")
    print(f"[train] 模型={args.model}  epochs={args.epochs}  imgsz={args.imgsz}  "
          f"batch={args.batch}  device={args.device}")

    model = YOLO(args.model)  # 加载 COCO 预训练权重；nc 不一致时 ultralytics 自动重建检测头
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=freeze_list,
        # ---- 小目标友好的增强/配置 ----
        mosaic=1.0,             #  mosaic 提升小目标上下文
        scale=0.9,              #  尺度抖动
        fliplr=0.5,             #  水平翻转
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        close_mosaic=10,        #  末 10 轮关闭 mosaic，稳定小框定位
        cos_lr=True,            #  余弦学习率
        patience=20,            #  早停
        amp=True,               #  自动混合精度（CPU 会自动禁用）
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print("\n[train] 完成。最佳权重位于 "
          f"{Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("[train] 可直接用于检测/跟踪：")
    print(f"   python src/detect_vehicles.py --source 你的图片.jpg "
          f"--model {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"   python src/track_vehicles.py  --source 你的视频.mp4 "
          f"--model {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
