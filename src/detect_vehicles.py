"""
静态照片车辆检测 + 分类 + 计数（高精度版）
=========================================
基于 **YOLO26**（Ultralytics 2026 发布的新一代实时检测模型，默认 yolo26m）识别图像中的车辆，输出：
  - 车辆总数
  - 各类型数量（小汽车 / 卡车 / 公交车 / 摩托车 / 自行车）
  - 带彩色框标注的结果图（按车型区分颜色）
  - 结构化 JSON 结果

精度提升手段：
  1. 默认采用 yolo26m 模型（比轻量版更准，对远处/较小车辆更鲁棒）；
  2. 默认按原图分辨率推理（imgsz 自适应，长边对齐 32，上限 1280），
     比固定 640 更能检出远处/较小的车辆；
  3. 可选 --tta 开启水平翻转测试时增强，合并两次预测，进一步降漏检；
  4. 车辆类白名单 + 按类 NMS 去重，避免重复计数。

缓存与配置默认写入当前项目目录。

用法:
  python src/detect_vehicles.py --source <图片路径> [--conf 0.25] [--imgsz 0] [--tta]
"""
import argparse
import json
import os
from pathlib import Path

# ===== 把所有缓存/配置锁定到本项目目录（E 盘），避免写入 C 盘 =====
ROOT = Path(__file__).resolve().parent.parent
_ultra_cfg = ROOT / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultra_cfg))
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

import cv2
import numpy as np
from ultralytics import YOLO, settings

try:
    settings.update({"runs_dir": str(ROOT / "outputs")})
except Exception:
    pass

DEFAULT_MODEL = "yolo26m.pt"  # 首次使用从 Ultralytics Hub 自动下载

# 仅保留车辆相关类别（COCO 中属于"车"的类）
VEHICLE_CLASSES = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
}

# 按车型区分标注颜色 (B, G, R)
COLORS = {
    "car": (0, 200, 0),
    "truck": (0, 165, 255),
    "bus": (255, 0, 0),
    "motorcycle": (255, 215, 0),
    "bicycle": (255, 0, 255),
}


def _predict(model, img, conf, imgsz):
    """返回 (boxes, names) 其中 boxes 为 Nx6 numpy: x1,y1,x2,y2,conf,cls_idx"""
    res = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 6)), res.names
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = res.boxes.cls.cpu().numpy().reshape(-1, 1)
    boxes = np.hstack([xyxy, confs, clss])
    return boxes, res.names


def _keep_vehicles(boxes, names):
    """只保留车辆类，返回 list[(x1,y1,x2,y2,conf,cls_name)]"""
    out = []
    for b in boxes:
        x1, y1, x2, y2, conf, cls_idx = b
        name = names[int(cls_idx)]
        if name in VEHICLE_CLASSES:
            out.append((x1, y1, x2, y2, float(conf), name))
    return out


def _iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    iw = max(0.0, xb - xa); ih = max(0.0, yb - ya)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(dets, iou_thr=0.5):
    """按类做非极大值抑制，去除重叠重复框"""
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if d[5] != best[5] or _iou(best, d) < iou_thr]
    return keep


def _auto_imgsz(h, w):
    """长边对齐 32，上限 1280，兼顾精度与 CPU 速度"""
    long = max(h, w)
    s = min(long, 1280)
    s = max(32, (s // 32) * 32)
    return s


def detect(source, model_path=DEFAULT_MODEL, conf=0.25, out_dir="outputs",
           imgsz=0, tta=False):
    model = YOLO(model_path)
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {source}")
    h, w = img.shape[:2]

    if imgsz and imgsz > 0:
        infer_size = int(imgsz)
    else:
        infer_size = _auto_imgsz(h, w)

    # 主预测
    boxes, names = _predict(model, img, conf, infer_size)
    dets = _keep_vehicles(boxes, names)

    # 测试时增强：水平翻转后再预测，框映射回原图并合并
    if tta:
        flipped = cv2.flip(img, 1)
        fboxes, fnames = _predict(model, flipped, conf, infer_size)
        for b in fboxes:
            x1, y1, x2, y2, c, ci = b
            name = fnames[int(ci)]
            if name not in VEHICLE_CLASSES:
                continue
            # 水平翻转还原: x' = w - x
            dets.append((w - x2, y1, w - x1, y2, float(c), name))

    dets = _nms(dets, iou_thr=0.5)

    counts = {}
    detections = []
    annotated = img.copy()
    for (x1, y1, x2, y2, c, name) in dets:
        counts[name] = counts.get(name, 0) + 1
        color = COLORS.get(name, (0, 255, 0))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {c:.2f}"
        cv2.putText(annotated, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections.append({
            "type": name,
            "type_cn": VEHICLE_CLASSES[name],
            "confidence": round(float(c), 3),
            "bbox": [float(round(x1, 1)), float(round(y1, 1)),
                     float(round(x2, 1)), float(round(y2, 1))],
            "width": float(round(x2 - x1, 1)),
            "height": float(round(y2 - y1, 1)),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(source).stem + "_detected.jpg")
    cv2.imwrite(str(out_path), annotated)

    summary = {
        "source": str(source),
        "model": model_path,
        "imgsz": infer_size,
        "tta": tta,
        "total_vehicles": sum(counts.values()),
        "counts_by_type": {VEHICLE_CLASSES[k]: v for k, v in counts.items()},
        "detections": detections,
        "annotated_image": str(out_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="静态照片车辆检测 / 分类 / 计数（高精度版）")
    parser.add_argument("--source", required=True, help="输入图片路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO 权重路径（默认 models/yolov8m.pt）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="推理分辨率；0=按原图自适应(上限1280)，可手动指定如 1280")
    parser.add_argument("--tta", action="store_true", help="开启水平翻转测试时增强（更准但更慢）")
    parser.add_argument("--out", default=str(ROOT / "outputs"), help="结果输出目录（默认 E 盘项目 outputs）")
    parser.add_argument("--json", default=None, help="JSON 结果保存路径（可选）")
    args = parser.parse_args()

    summary = detect(args.source, args.model, args.conf, args.out, args.imgsz, args.tta)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[结果已保存] {args.json}")


if __name__ == "__main__":
    main()
