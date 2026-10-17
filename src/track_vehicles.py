"""
航拍视频目标检测 + 多目标跟踪（YOLO26 + 内置 BoT-SORT）
========================================================
基于 **YOLO26**（Ultralytics 2026 发布的新一代实时检测模型，默认 yolo26m）逐帧检测
无人机/航拍视频中的车辆，再用 **Ultralytics 内置的 BoT-SORT** 跟踪器把同一辆车的轨迹
串联起来。

输入：原始视频（.mp4/.avi/.mov/.mkv/.webm/.flv）、图片序列目录、或单张图片
输出：
  - 渲染视频：每一帧叠加「检测框 + 跟踪 ID + 车型 + 置信度 + 轨迹折线」的可视化图层
  - 结构化 JSON：逐帧检测结果 + 车辆级轨迹（id / 逐帧 bbox / 中心点 / 置信度），
    可直接用于车流量统计、平均速度、轨迹热力图等下游任务

防抖原理：BoT-SORT 的 **GMC 全局运动补偿**估计无人机自身运动并补偿，使目标在
"相机运动时"仍能被稳定关联，避免整帧平移/旋转造成的 ID 跳变（航拍抖动的核心难题）。

缓存与配置默认写入当前项目目录。

用法:
  # 视频（默认 YOLO26m + 内置 BoT-SORT 防抖），输出标注视频 + 轨迹 JSON
  python src/track_vehicles.py --source 航拍视频.mp4 --json outputs/tracks.json
  # 指定模型 / 阈值 / 分辨率
  python src/track_vehicles.py --source 航拍视频.mp4 --model yolo26m.pt --conf 0.3 --imgsz 1280
  # 更快（关闭 ReID，纯运动+防抖）:
  python src/track_vehicles.py --source 航拍视频.mp4 --no-reid
  # 换防抖方法（大幅抖动用 ecc，平滑运动用 sparseOptFlow）:
  python src/track_vehicles.py --source 航拍视频.mp4 --gmc ecc
  # 图像序列目录（逐帧当视频处理）:
  python src/track_vehicles.py --source 图片文件夹/ --fps 10
"""
import argparse
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import ultralytics as _ultra
import yaml

# ===== 把所有缓存/配置锁定到本项目目录，避免写入 C 盘 =====
ROOT = Path(__file__).resolve().parent.parent
_ultra_cfg = ROOT / ".ultralytics"
os.environ["YOLO_CONFIG_DIR"] = str(_ultra_cfg)  # 强制指向项目内（覆盖任何已有值）
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

try:
    settings.update({"runs_dir": str(ROOT / "outputs")})
except Exception:
    pass

# 默认模型：首次使用会从 Ultralytics Hub 自动下载 yolo26m.pt（需联网）
DEFAULT_MODEL = "yolo26m.pt"
# Ultralytics 内置的 BoT-SORT 跟踪配置（我们在此基础上覆盖 gmc / reid 参数）
BUILTIN_BOTSORT = Path(_ultra.__file__).resolve().parent / "cfg" / "trackers" / "botsort.yaml"

# 车辆相关类别（COCO 名称；按「名称」匹配，兼容 YOLO26 的 COCO 类别顺序）
VEHICLE_CLASSES = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
}

# 按车型区分标注颜色 (B, G, R)
TYPE_COLORS = {
    "car": (0, 200, 0),
    "truck": (0, 165, 255),
    "bus": (255, 0, 0),
    "motorcycle": (255, 215, 0),
    "bicycle": (255, 0, 255),
}


def color_for_id(tid: int):
    """为每条轨迹分配稳定且区分度高的颜色 (B,G,R)"""
    np.random.seed(tid * 7 + 13)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def build_tracker_cfg(gmc: str, with_reid: bool):
    """以 Ultralytics 内置 botsort.yaml 为基础，覆盖 GMC / ReID 参数，写到项目缓存返回路径"""
    if not BUILTIN_BOTSORT.exists():
        raise FileNotFoundError(f"未找到内置 BoT-SORT 配置: {BUILTIN_BOTSORT}")
    with open(BUILTIN_BOTSORT, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["gmc_method"] = gmc
    cfg["with_reid"] = bool(with_reid)
    out = ROOT / ".cache" / "botsort_override.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(out)


def _is_video(path: Path):
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _list_frames(source: Path):
    """返回有序图片帧路径列表；若是视频则返回 None（用 cap 处理）"""
    if source.is_dir():
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files.extend(source.glob(e))
        return sorted(files)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return [source]
    return None


def track(source, model_path=DEFAULT_MODEL, conf=0.3, imgsz=1280,
          out_dir="outputs", json_path=None, draw_traj=True, save_video=True,
          device="cpu", gmc="orb", with_reid=True, fps=10, max_frames=0,
          include_frames=True, oriented=True):
    src = Path(source)
    model = YOLO(model_path)
    names = model.names  # 加载权重后为 COCO 类别名

    # 按「类别名」匹配车辆类，兼容任意 YOLO 变体的类别顺序
    vehicle_idx = [i for i, n in names.items() if n in VEHICLE_CLASSES]

    tracker_yaml = build_tracker_cfg(gmc, with_reid)

    # ---- 轨迹 / 逐帧存储 ----
    tracks = defaultdict(lambda: {
        "type": None, "type_cn": None,
        "frames": [], "boxes": [], "centers": [], "confs": [], "oriented": [],
    })
    type_counter = defaultdict(int)
    history = defaultdict(lambda: deque(maxlen=200))  # tid -> 中心点历史(画轨迹)
    rot_state = defaultdict(lambda: {"angle": None, "L": None, "W": None})  # 旋转框状态
    frame_no = 0
    video_writer = None
    ann_img_path = None
    out_size = None
    out_fps = fps
    res_w = res_h = None

    is_img = src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    is_video = _is_video(src)
    frames_list = _list_frames(src)
    is_sequence = frames_list is not None and not is_img

    # ---- 视频源：先取 fps / 尺寸 ----
    if is_video:
        cap = cv2.VideoCapture(str(src))
        out_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        ret, probe = cap.read()
        if ret:
            res_h, res_w = probe.shape[:2]
        cap.release()

    print(f"[加载] 模型={model_path}  跟踪器=内置 BoT-SORT(gmc={gmc}, reid={with_reid})")
    print(f"[加载] 置信度={conf}  推理分辨率={imgsz}  设备={device}")

    stream = model.track(
        source=str(source),
        tracker=tracker_yaml,
        conf=conf,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
        persist=True,
        classes=vehicle_idx,
    )

    frames_json = []  # 逐帧检测结果（结构化数据）

    for res in stream:
        frame_no += 1
        if max_frames and frame_no > max_frames:
            break

        im0 = res.orig_img
        h, w = im0.shape[:2]
        if out_size is None:
            out_size = (w, h)
        if res_w is None:
            res_w, res_h = w, h

        ann = im0.copy()
        dets_this_frame = []
        boxes = res.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().numpy().reshape(-1)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy().reshape(-1)
            clss = boxes.cls.int().cpu().numpy().reshape(-1)
            for tid, (x1, y1, x2, y2), c, ci in zip(ids, xyxy, confs, clss):
                name = names[int(ci)]
                if name not in VEHICLE_CLASSES:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                t = tracks[tid]
                if t["type"] is None:
                    t["type"] = name
                    t["type_cn"] = VEHICLE_CLASSES[name]
                    type_counter[name] += 1
                t["frames"].append(frame_no)
                t["boxes"].append([round(x1, 1), round(y1, 1),
                                   round(x2, 1), round(y2, 1)])
                t["centers"].append([round(cx, 1), round(cy, 1)])
                t["confs"].append(round(float(c), 3))
                history[tid].append((int(cx), int(cy)))

                # ---- 计算「贴合车身朝向」的旋转框 ----
                # 思路：用车辆最近若干帧中心点的运动方向估计车头朝向，
                # 再把检测框渲染成沿该方向旋转、长宽稳定的矩形。
                # 转弯时运动方向随之改变，框便斜着贴住车身。
                w0, h0 = x2 - x1, y2 - y1
                st = rot_state[tid]
                obb = None
                if oriented:
                    hist = history[tid]
                    inst_ang = None
                    if len(hist) >= 2:
                        (px, py), (qx, qy) = hist[-2], hist[-1]
                        dx, dy = qx - px, qy - py
                        if math.hypot(dx, dy) > 1.5:       # 过滤静止抖动
                            inst_ang = math.atan2(dy, dx)  # 运动方向≈车头朝向
                    if inst_ang is not None:
                        if st["angle"] is None:
                            st["angle"] = inst_ang
                        else:
                            # 角度差归一化到 [-pi, pi] 后做 EMA 平滑，避免跳变
                            da = (inst_ang - st["angle"] + math.pi) % (2 * math.pi) - math.pi
                            st["angle"] += 0.25 * da
                    elif st["angle"] is None:
                        # 首帧无运动信息：以长边方向作为初始朝向
                        st["angle"] = 0.0 if w0 >= h0 else math.pi / 2
                    # 稳定长宽：长边=车长，短边=车宽，EMA 平滑避免 45° 退化
                    long0, short0 = max(w0, h0), min(w0, h0)
                    if st["L"] is None:
                        st["L"], st["W"] = long0, short0
                    else:
                        st["L"] += 0.1 * (long0 - st["L"])
                        st["W"] += 0.1 * (short0 - st["W"])
                    angle_deg = math.degrees(st["angle"]) % 180
                    obb = {"cx": round(cx, 1), "cy": round(cy, 1),
                           "w": round(st["L"], 1), "h": round(st["W"], 1),
                           "angle_deg": round(angle_deg, 1)}
                    t["oriented"].append(obb)

                # 记录到逐帧结构化数据
                det_entry = {
                    "track_id": tid,
                    "type": name,
                    "type_cn": VEHICLE_CLASSES[name],
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "conf": round(float(c), 3),
                }
                if obb is not None:
                    det_entry["oriented_bbox"] = obb
                dets_this_frame.append(det_entry)

                # 画框 + ID + 车型 + 置信度
                col = TYPE_COLORS.get(name, (0, 255, 0))
                if obb is not None:
                    rect = ((obb["cx"], obb["cy"]), (obb["w"], obb["h"]), obb["angle_deg"])
                    pts = cv2.boxPoints(rect)
                    cv2.polylines(ann, [np.int32(pts)], True, col, 2)
                    lx, ly = int(pts[0][0]), max(int(pts[0][1]) - 6, 14)
                else:
                    cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    lx, ly = int(x1), max(int(y1) - 6, 14)
                label = f"#{tid} {name} {c:.2f}"
                cv2.putText(ann, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        if include_frames:
            frames_json.append({"frame": frame_no, "detections": dets_this_frame})

        # 画轨迹折线
        if draw_traj:
            for tid, pts in history.items():
                if len(pts) < 2:
                    continue
                col = color_for_id(tid)
                for i in range(1, len(pts)):
                    cv2.line(ann, pts[i - 1], pts[i], col, 2)

        # ---- 写渲染视频 / 标注图 ----
        if is_img:
            # 单张图片：保存标注图
            out_dir_p = Path(out_dir)
            out_dir_p.mkdir(parents=True, exist_ok=True)
            ann_img_path = out_dir_p / (src.stem + "_detected.jpg")
            cv2.imwrite(str(ann_img_path), ann)
        elif save_video:
            if video_writer is None:
                vdir = Path(out_dir)
                vdir.mkdir(parents=True, exist_ok=True)
                vpath = vdir / (src.stem + "_tracked.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(vpath), fourcc, out_fps, out_size)
                track_video_path = vpath
            video_writer.write(ann)

        # 进度
        if frame_no % 20 == 0:
            print(f"  已处理帧 {frame_no}，当前活跃轨迹 {len(history)} 条", flush=True)

    if video_writer is not None:
        video_writer.release()

    # ---- 汇总结构化数据 ----
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    total_tracks = len(tracks)
    avg_len = (sum(len(t["frames"]) for t in tracks.values()) / total_tracks) if total_tracks else 0
    total_detections = sum(len(f["detections"]) for f in frames_json) if include_frames else None

    track_summary = {}
    for tid, t in tracks.items():
        avg_conf = round(sum(t["confs"]) / len(t["confs"]), 3) if t["confs"] else 0.0
        track_summary[str(tid)] = {
            "type": t["type"],
            "type_cn": t["type_cn"],
            "frame_count": len(t["frames"]),
            "first_frame": t["frames"][0] if t["frames"] else None,
            "last_frame": t["frames"][-1] if t["frames"] else None,
            "avg_conf": avg_conf,
            "path": [{"frame": f, "center": c, "bbox": b, "conf": cf,
                      "oriented_bbox": o}
                     for f, c, b, cf, o in zip(t["frames"], t["centers"],
                                               t["boxes"], t["confs"], t["oriented"])],
        }

    summary = {
        "source": str(source),
        "algorithm": "YOLO26 + BoT-SORT (built-in)",
        "model": model_path,
        "model_family": "yolo26",
        "oriented_boxes": bool(oriented),
        "tracker": "botsort.yaml (built-in)",
        "gmc_method": gmc,
        "with_reid": bool(with_reid),
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "fps": round(out_fps, 2) if out_fps else None,
        "resolution": {"width": res_w, "height": res_h},
        "frame_count": frame_no,
        "summary": {
            "total_tracks": total_tracks,
            "avg_track_length_frames": round(avg_len, 1),
            "counts_by_type": {VEHICLE_CLASSES[k]: v for k, v in type_counter.items()},
            "total_detections": total_detections,
        },
        "tracks": track_summary,
        "frames": frames_json if include_frames else None,
        "rendered_video": str(track_video_path) if (save_video and not is_img and video_writer is not None) else None,
        "annotated_image": str(ann_img_path) if is_img else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="航拍视频目标检测 + 跟踪 · YOLO26 + 内置 BoT-SORT")
    parser.add_argument("--source", required=True, help="视频文件 / 图片序列目录 / 单张图片")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO 权重（默认 yolo26m.pt，首次使用自动从 Hub 下载）")
    parser.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值（默认 0.3）")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理分辨率（默认 1280）")
    parser.add_argument("--out", default=str(ROOT / "outputs"), help="结果输出目录")
    parser.add_argument("--json", default=str(ROOT / "outputs" / "tracks.json"),
                        help="结构化结果 JSON 保存路径（默认 outputs/tracks.json）")
    parser.add_argument("--gmc", default="orb",
                        choices=["orb", "ecc", "sparseOptFlow", "sift", "none"],
                        help="防抖方法（全局运动补偿），默认 orb")
    parser.add_argument("--no-reid", action="store_true", help="关闭 ReID 外观分支（更快）")
    parser.add_argument("--no-traj", action="store_true", help="不在视频上绘制轨迹折线")
    parser.add_argument("--no-video", action="store_true", help="不生成标注视频（仅输出 JSON）")
    parser.add_argument("--no-frames", action="store_true", help="JSON 中不写入逐帧数组（tracks 已含逐帧信息）")
    parser.add_argument("--no-oriented", action="store_true",
                        help="关闭旋转框（贴合车身朝向），改用原始正交框")
    parser.add_argument("--device", default="cpu", help="推理设备 cpu / 0(若有GPU)")
    parser.add_argument("--fps", type=float, default=10, help="图像序列输出视频帧率（默认 10）")
    parser.add_argument("--max-frames", type=int, default=0, help="仅处理前 N 帧（调试用，0=全部）")
    args = parser.parse_args()

    summary = track(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        out_dir=args.out,
        json_path=args.json,
        draw_traj=not args.no_traj,
        save_video=not args.no_video,
        device=args.device,
        gmc=args.gmc,
        with_reid=not args.no_reid,
        fps=args.fps,
        max_frames=args.max_frames,
        include_frames=not args.no_frames,
        oriented=not args.no_oriented,
    )

    print("\n===== 航拍检测 + 跟踪结果 =====")
    print(f"视频/序列 : {summary['source']}")
    print(f"算法      : {summary['algorithm']}")
    print(f"处理帧数  : {summary['frame_count']}")
    print(f"车辆轨迹  : {summary['summary']['total_tracks']} 条 "
          f"(平均长度 {summary['summary']['avg_track_length_frames']} 帧)")
    print(f"车型统计  : {summary['summary']['counts_by_type']}")
    if summary.get("rendered_video"):
        print(f"渲染视频  : {summary['rendered_video']}")
    if summary.get("annotated_image"):
        print(f"标注图片  : {summary['annotated_image']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建 JSON 所在目录
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[结构化 JSON 已保存] {args.json}")


if __name__ == "__main__":
    main()
