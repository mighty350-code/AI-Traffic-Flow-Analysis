"""
航拍路况视频 -> FLUID data_process 可直接读取的识别结果 .txt
=========================================================
调用 **YOLO26m** 做目标跟踪，拿到 track_id / 车型 / 置信度；
同时调用 **YOLO26 OBB 模型(yolo26m-obb.pt)** 检测「真实倾斜角度」的旋转框，
把两者按中心点匹配后，输出 **两** 份 .txt：

  1) <视频名>_yolo_obb.txt   旋转框(OBB)，列：
     frame,id,cx,cy,w,h,r,confidence,type        （r = 弧度，[-π/2, π/2]）
  2) <视频名>_yolo_hbb.txt   水平框(HBB)，列：
     frame,id,ltx,lty,w,h,confidence,type        （ltx/lty = 左上角像素坐标）

这两份文件**无表头、逗号分隔**，文件名分别含 `obb` / `hbb`，
且带 `_yolo` 后缀 —— 正好对齐 FLUID 官方流程：

  * demo/data_process/1_Structured_Recognition_Results.py
      读 input/ 下含 hbb / obb 的 .txt，补表头 -> 同名 .csv
  * demo/data_process/2_holography_smooth_kinematic_4K_batch.py
      按文件名里 `_yolo` 之前的片段去 calibration/ 匹配标定文件，
      再把像素坐标平滑并转成米制经纬度轨迹

为什么这样设计
--------------
- FLUID 的 OBB `r` 全程按**弧度**处理（脚本2用 sin/cos 平滑、再叠加航向旋转角），
  所以本脚本输出弧度 `[-π/2, π/2]`，而不是常见的「度」。
- `type` 映射到 FLUID 词表（small-vehicle / large-vehicle / pedestrian / moped /
  unknown），和 FLUID 自带 DJI 数据、以及 visualization 的配色逻辑一致。
- HBB 用跟踪模型的轴对齐框（左上角 ltx,lty + 宽高 w,h），OBB 用匹配到的
  旋转框（中心 cx,cy + 宽高 w,h + 弧度 r）。两者 id / frame 一致。

用法
----
  # 默认输出到 outputs/<视频名>_yolo_obb.txt / _yolo_hbb.txt
  python src/video_to_txt.py --source 航拍路况.mp4 --device 0

  # 直接写进 FLUID 的 input/ 目录（免复制）
  python src/video_to_txt.py --source 航拍路况.mp4 \
        --fluid-input /workspace/FLUID/demo/data_process/input

  # 先看一小段
  python src/video_to_txt.py --source 航拍路况.mp4 --device 0 --max-frames 100

  # 同时生成渲染视频肉眼检查（旋转框贴合车身）
  python src/video_to_txt.py --source 航拍路况.mp4 --device 0 --video

  # 指定 OBB 模型（如用更大的 yolo26l-obb.pt）
  python src/video_to_txt.py --source 航拍路况.mp4 --device 0 --obb-model yolo26l-obb.pt

跑完后，把两份 *_yolo_*.txt 放进 FLUID 的 demo/data_process/input/ ，
并确保 demo/data_process/calibration/<视频名>.csv 已准备好（GCP 控制点，
列：point,lon,lat,x,y）—— 然后按顺序跑 1_ / 2_ / 3_ ... 即可。
"""
import argparse
import math
import os
from collections import defaultdict, deque
from pathlib import Path

# ===== 把所有缓存/配置锁定到本项目目录，避免写入 C 盘 =====
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.makedirs(ROOT / ".ultralytics", exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from track_vehicles import DEFAULT_MODEL, VEHICLE_CLASSES, build_tracker_cfg  # noqa: E402

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# 旋转框模型：与 YOLO26m 同家族的 OBB 预训练权重（DOTAv1，含车辆类）
DEFAULT_OBB_MODEL = "yolo26m-obb.pt"

# 把 YOLO26 的 COCO 车型名映射到 FLUID 的词表
# （small-vehicle / large-vehicle / pedestrian / moped / unknown）
FLUID_TYPE_MAP = {
    "car": "small-vehicle",
    "truck": "large-vehicle",
    "bus": "large-vehicle",
    "motorcycle": "moped",
    "bicycle": "moped",
}


def map_to_fluid_type(name: str) -> str:
    """把模型输出的车型名映射到 FLUID 的 type 词表；未知类归为 unknown。"""
    return FLUID_TYPE_MAP.get(str(name).lower(), "unknown")


def _norm_angle_rad(a: float) -> float:
    """把弧度角归一化到 (-π/2, π/2]（旋转框的周期是 π，无需交换 w/h）。"""
    a = a % math.pi
    if a > math.pi / 2:
        a -= math.pi
    return a


def _obb_detections(model_obb, im0, device, conf, classes):
    """对单帧跑 OBB 检测，返回 [(cx, cy, w, h, r_rad, poly), ...]。

    r_rad 为弧度（Ultralytics xywhr 直接给出，范围约 [-π/2, π/2]）。
    poly 为 (4,2) 的 4 个角点，用于准确绘制。
    """
    pred_kwargs = dict(conf=conf, device=device, verbose=False, imgsz=1024)
    if classes:  # 仅当确实筛出车辆类时才限制，避免传空列表报错
        pred_kwargs["classes"] = classes
    res = model_obb.predict(im0, **pred_kwargs)[0]
    obb = getattr(res, "obb", None)
    if obb is None or obb.xywhr is None:
        return []
    xywhr = obb.xywhr.cpu().numpy()          # (N,5): cx, cy, w, h, angle(弧度)
    polys = obb.xyxyxyxy.cpu().numpy()       # (N,4,2): 4 个角点
    out = []
    for i, row in enumerate(xywhr):
        cx, cy, w, h, r_rad = row
        out.append((float(cx), float(cy), float(w), float(h),
                    float(r_rad), polys[i]))
    return out


def video_to_txt(source, model_path=DEFAULT_MODEL, obb_model=DEFAULT_OBB_MODEL,
                 conf=0.3, imgsz=1280, out_dir="outputs",
                 fluid_input=None, device="cpu", gmc="orb", with_reid=True,
                 do_track=True, save_video=False, max_frames=0):
    # ---- 加载 HBB 跟踪模型（提供 id / 车型 / 置信度 / 轴对齐框）----
    src = Path(source)
    stem = src.stem
    model = YOLO(model_path)
    names = model.names
    vehicle_idx = [i for i, n in names.items() if n in VEHICLE_CLASSES]
    tracker_yaml = build_tracker_cfg(gmc, with_reid)

    # ---- 加载 OBB 旋转框模型（提供真实倾斜角度，弧度）----
    model_obb = None
    obb_vehicle_idx = None
    if obb_model not in (None, "", "none", "off"):
        try:
            model_obb = YOLO(obb_model)
            obb_names = model_obb.names
            # 仅保留车辆类（DOTAv1 中 small-vehicle / large-vehicle）
            obb_vehicle_idx = [i for i, n in obb_names.items()
                               if "vehicle" in str(n).lower()]
            print(f"[OBB 模型] {obb_model} 已加载，车辆类索引={obb_vehicle_idx}")
        except Exception as e:
            print(f"[警告] OBB 模型加载失败，将退回运动方向近似角度(弧度): {e}")
            model_obb = None

    history = defaultdict(lambda: deque(maxlen=200))        # tid -> 中心点历史
    rot_state = defaultdict(lambda: {"angle": None, "L": None, "W": None})

    # 视频源探测 fps / 尺寸
    res_fps = 10
    if src.suffix.lower() in _VIDEO_EXTS:
        cap = cv2.VideoCapture(str(src))
        res_fps = cap.get(cv2.CAP_PROP_FPS) or 10
        cap.release()

    stream = model.track(
        source=str(source), tracker=tracker_yaml, conf=conf, imgsz=imgsz,
        device=device, stream=True, verbose=False, persist=True,
        classes=vehicle_idx,
    )

    hbb_lines, obb_lines = [], []

    video_writer = None
    out_w = out_h = None
    frame_no = 0

    for res in stream:
        frame_no += 1
        if max_frames and frame_no > max_frames:
            break

        im0 = res.orig_img
        h, w = im0.shape[:2]
        ann = im0.copy() if save_video else None
        boxes = res.boxes

        # 本帧 OBB 检测（真实倾斜框，弧度）
        obbs = (_obb_detections(model_obb, im0, device, conf, obb_vehicle_idx)
                if model_obb else [])

        if boxes is not None and (do_track and boxes.id is not None or not do_track):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy().reshape(-1)
            clss = boxes.cls.int().cpu().numpy().reshape(-1)
            ids = (boxes.id.int().cpu().numpy().reshape(-1)
                   if (do_track and boxes.id is not None) else [-1] * len(clss))

            for k in range(len(clss)):
                name = names[int(clss[k])]
                if name not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy[k]]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                c = float(confs[k])
                tid = int(ids[k]) if do_track else -1
                ftype = map_to_fluid_type(name)

                # ---- HBB 行：轴对齐框，左上角 ltx/lty + 宽高 w/h（像素）----
                ltx, lty = round(x1, 1), round(y1, 1)
                bw, bh = round(x2 - x1, 1), round(y2 - y1, 1)
                hbb_lines.append(
                    f"{frame_no},{tid},{ltx},{lty},{bw},{bh},{round(c, 3)},{ftype}")

                # ---- OBB 行：优先用 OBB 检测的真实倾斜框（弧度 r）----
                ocx = ocy = ow = oh = r = poly = None
                if obbs:
                    best_d, best = 1e9, None
                    for det in obbs:
                        d = math.hypot(det[0] - cx, det[1] - cy)
                        if d < best_d:
                            best_d, best = d, det
                    # 匹配阈值：不超过 HBB 长边的 0.7，且至少 10px
                    thr = max(0.7 * max(x2 - x1, y2 - y1), 10.0)
                    if best is not None and best_d <= thr:
                        ocx, ocy, ow, oh, r_rad, poly = best
                        r = _norm_angle_rad(float(r_rad))   # 弧度，[-π/2, π/2]

                # ---- 回退：无 OBB 时用运动方向估计角度（弧度）----
                if ocx is None:
                    w0, h0 = x2 - x1, y2 - y1
                    if do_track:
                        hist = history[tid]
                        hist.append((int(cx), int(cy)))
                        st = rot_state[tid]
                        inst = None
                        if len(hist) >= 2:
                            (px, py), (qx, qy) = hist[-2], hist[-1]
                            dx, dy = qx - px, qy - py
                            if math.hypot(dx, dy) > 1.5:
                                inst = math.atan2(dy, dx)
                        if inst is not None:
                            if st["angle"] is None:
                                st["angle"] = inst
                            else:
                                da = ((inst - st["angle"] + math.pi) % (2 * math.pi)
                                      - math.pi)
                                st["angle"] += 0.25 * da
                        elif st["angle"] is None:
                            st["angle"] = 0.0 if w0 >= h0 else math.pi / 2
                        long0, short0 = max(w0, h0), min(w0, h0)
                        if st["L"] is None:
                            st["L"], st["W"] = long0, short0
                        else:
                            st["L"] += 0.1 * (long0 - st["L"])
                            st["W"] += 0.1 * (short0 - st["W"])
                        r = _norm_angle_rad(st["angle"])            # 弧度
                        ocx, ocy, ow, oh = cx, cy, round(st["L"], 1), round(st["W"], 1)
                    else:
                        # 既无 OBB 又未跟踪：轴对齐，r=0（弧度）
                        r = 0.0
                        ocx, ocy, ow, oh = cx, cy, round(w0, 1), round(h0, 1)

                # ---- OBB 行：frame,id,cx,cy,w,h,r(弧度),confidence,type ----
                obb_lines.append(
                    f"{frame_no},{tid},"
                    f"{round(ocx, 1)},{round(ocy, 1)},"
                    f"{round(ow, 1)},{round(oh, 1)},"
                    f"{round(float(r), 6)},{round(c, 3)},{ftype}")

                if save_video:
                    col = (0, 200, 0)
                    if poly is not None:
                        cv2.polylines(ann, [np.int32(poly)], True, col, 2)
                    else:
                        rect = ((ocx, ocy), (ow, oh), math.degrees(r))
                        pts = cv2.boxPoints(rect)
                        cv2.polylines(ann, [np.int32(pts)], True, col, 2)
                    cv2.putText(ann, f"#{tid} {ftype} {c:.2f}",
                                (int(x1), max(int(y1) - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if save_video and ann is not None:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vpath = (Path(fluid_input) if fluid_input
                         else Path(out_dir)) / (stem + "_obb_tracked.mp4")
                video_writer = cv2.VideoWriter(str(vpath), fourcc, res_fps, (w, h))
                print(f"[渲染视频] -> {vpath}")
            video_writer.write(ann)

        if frame_no % 50 == 0:
            print(f"  已处理帧 {frame_no}，HBB {len(hbb_lines)} 条 / OBB {len(obb_lines)} 条",
                  flush=True)

    if video_writer is not None:
        video_writer.release()

    # ---- 写出两份 .txt（无表头，逗号分隔；文件名含 hbb / obb / _yolo）----
    if fluid_input:
        out_dir_path = Path(fluid_input)
    else:
        out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    hbb_path = out_dir_path / f"{stem}_yolo_hbb.txt"
    obb_path = out_dir_path / f"{stem}_yolo_obb.txt"

    hbb_path.write_text("\n".join(hbb_lines) + "\n", encoding="utf-8")
    obb_path.write_text("\n".join(obb_lines) + "\n", encoding="utf-8")

    print(f"[HBB TXT 已保存] {hbb_path}  ({len(hbb_lines)} 条)")
    print(f"[OBB TXT 已保存] {obb_path}  ({len(obb_lines)} 条, r=弧度)")
    if not fluid_input:
        print("  把这两份 *_yolo_*.txt 放进 FLUID 的 demo/data_process/input/ 即可"
              "（并确保 calibration/<视频名>.csv 已就绪）。")
    return str(hbb_path), str(obb_path)


def main():
    p = argparse.ArgumentParser(
        description="航拍路况视频 -> FLUID data_process 识别结果 .txt "
                    "(OBB+弧度 + HBB，YOLO26m 跟踪 + YOLO26-OBB 角度)")
    p.add_argument("--source", required=True, help="航拍路况视频文件")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HBB 跟踪模型（默认 yolo26m.pt，提供 id/车型/置信度）")
    p.add_argument("--obb-model", default=DEFAULT_OBB_MODEL,
                   help="OBB 旋转框模型（默认 yolo26m-obb.pt；传 none 关闭，用运动方向近似角度）")
    p.add_argument("--out-dir", default=str(ROOT / "outputs"),
                   help="输出目录（默认 outputs/），写入 <视频名>_yolo_hbb.txt / _obb.txt")
    p.add_argument("--fluid-input", default=None,
                   help="直接写到 FLUID 的 input/ 目录路径（如 /workspace/FLUID/demo/data_process/input），免复制")
    p.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    p.add_argument("--imgsz", type=int, default=1280, help="HBB 跟踪推理分辨率")
    p.add_argument("--device", default="cpu", help="cpu / 0(若有GPU)")
    p.add_argument("--gmc", default="orb",
                   choices=["orb", "ecc", "sparseOptFlow", "sift", "none"],
                   help="防抖方法，默认 orb")
    p.add_argument("--no-reid", action="store_true", help="关闭 ReID（更快）")
    p.add_argument("--no-track", action="store_true",
                   help="仅检测不跟踪（id 记为 -1）")
    p.add_argument("--video", action="store_true",
                   help="同时输出渲染视频便于肉眼检查")
    p.add_argument("--max-frames", type=int, default=0,
                   help="仅处理前 N 帧（调试用，0=全部）")
    args = p.parse_args()

    video_to_txt(
        source=args.source, model_path=args.model,
        obb_model=args.obb_model, conf=args.conf, imgsz=args.imgsz,
        out_dir=args.out_dir, fluid_input=args.fluid_input, device=args.device,
        gmc=args.gmc, with_reid=not args.no_reid, do_track=not args.no_track,
        save_video=args.video, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
