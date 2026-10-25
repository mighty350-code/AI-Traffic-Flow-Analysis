# -*- coding: utf-8 -*-
"""
在 TI 路口视频的前 N 帧上叠加 FLUID 轨迹，验证「像素坐标 → 世界坐标」变换
与轨迹质量在真实画面上是否自洽。

做法
  1. 用 ffmpeg 从 mp4 抽前 N 帧（默认 1000）到 raw/frames/<video>/
  2. 读 segments.csv / tracks.csv，取落在 [1, N] 帧窗口内的轨迹点
  3. 在每帧上绘制该帧所有目标的 bbox（cx,cy,w,h）与 id
  4. 统计画面内目标数、平均框尺寸、可见帧率，并输出抽样拼图

输出
  reports/frame_overlay_stats.json
  reports/figures/frame_overlay_sample.png

用法: python overlay_frames.py [--frames 1000] [--sample 6]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

VIDEO_DIR = {"20250529_video-1": "video-1",
             "20250529_video-2": "video-2",
             "20250529_video-3": "video-3"}


def collect_points(trj, frame):
    d = trj[(trj.frame == frame) & trj.cx.notna()]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, args.data)
    out_dir = os.path.join(root, args.out)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 轨迹点：从原始 traj 读（像素坐标）
    raw = os.path.join(root, args.raw, "traj")
    seg = pd.read_csv(os.path.join(data_dir, "segments.csv"))

    stats = {}
    sample_panels = []

    for vid, sub in VIDEO_DIR.items():
        fdir = os.path.join(root, args.raw, "frames", sub)
        if not os.path.isdir(fdir):
            stats[vid] = {"status": "no_frames", "frame_dir": fdir}
            continue
        imgs = sorted(f for f in os.listdir(fdir) if f.lower().endswith((".jpg", ".png")))
        if not imgs:
            stats[vid] = {"status": "empty_dir", "frame_dir": fdir}
            continue
        n = min(len(imgs), args.frames)

        trj = pd.read_csv(os.path.join(raw, f"20250529_{sub}_Traj.csv"),
                          low_memory=False)
        trj = trj[trj.frame <= n]
        by_frame = {int(f): g for f, g in trj.groupby("frame")}

        # 逐帧统计
        counts = []
        for i in range(1, n + 1):
            g = by_frame.get(i)
            counts.append(0 if g is None else len(g))
        counts = np.array(counts)

        present = sorted(by_frame.keys())
        # 帧覆盖率：有目标的帧数
        cov = len(present) / n

        # 目标在画面中的位置
        allpts = trj.dropna(subset=["cx", "cy"])
        in_view = ((allpts.cx >= 0) & (allpts.cx <= 3840) &
                   (allpts.cy >= 0) & (allpts.cy <= 2160)).mean()

        # 与 segments 对齐率：segments 覆盖的帧里有多少能在 traj 找到点
        s = seg[(seg.video_id == vid) & (seg.seg_end_frame <= n)]
        covered = set()
        for _, r in s.iterrows():
            covered.update(range(int(r.seg_start_frame), int(r.seg_end_frame) + 1))
        hit = sum(1 for f in covered if f in by_frame)
        seg_cov = hit / len(covered) if covered else 0.0

        stats[vid] = {
            "status": "ok",
            "n_frames_extracted": len(imgs),
            "n_frames_used": n,
            "frame_size": list(Image.open(os.path.join(fdir, imgs[0])).size),
            "mean_targets_per_frame": float(counts.mean()),
            "max_targets_per_frame": int(counts.max()),
            "frames_with_target_ratio": float(cov),
            "points_in_view_ratio": float(in_view),
            "segment_frame_hit_ratio": float(seg_cov),
            "median_box_w_px": float(allpts.w.median()),
            "median_box_h_px": float(allpts.h.median()),
        }

        # 抽样拼图
        picks = np.linspace(1, n, args.sample).astype(int)
        for p in picks:
            fn = imgs[p - 1] if p - 1 < len(imgs) else imgs[-1]
            im = Image.open(os.path.join(fdir, fn)).convert("RGB")
            g = by_frame.get(int(p))
            sample_panels.append((vid, int(p), im, g))

    with open(os.path.join(out_dir, "frame_overlay_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 画抽样图
    if sample_panels:
        cols = 3
        rows = int(np.ceil(len(sample_panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.6 * cols, 3.9 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (vid, fr, im, g) in zip(axes, sample_panels):
            ax.imshow(im)
            if g is not None and len(g):
                for _, r in g.iterrows():
                    ax.add_patch(patches.Rectangle(
                        (r.cx - r.w / 2, r.cy - r.h / 2), r.w, r.h,
                        linewidth=1.1, edgecolor="#C0392B", facecolor="none"))
                    ax.text(r.cx - r.w / 2, r.cy - r.h / 2 - 2,
                            str(int(r.id)), color="#FFD166", fontsize=7,
                            ha="left", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.12",
                                      fc="#1B3A5C", ec="none", alpha=0.75))
            ax.set_title(f"{vid.replace('20250529_','')}  frame {fr}  "
                         f"目标 {0 if g is None else len(g)}", fontsize=11)
            ax.axis("off")
        for ax in axes[len(sample_panels):]:
            ax.axis("off")
        fig.suptitle("TI 路口抽帧轨迹叠加（红框=FLUID 检测框，数字=轨迹 id）",
                     y=1.0, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "frame_overlay_sample.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
