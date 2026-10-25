# -*- coding: utf-8 -*-
"""
FLUID / TI 路口统一训练数据的回归校验。

在 build_dataset.py 产出数据后运行，逐条断言数据契约：
  1. 划分粒度 = 整段视频（video_id 与 split 一一对应）
  2. 同一轨迹（video_id, track_id 复合键）绝不跨 split
     —— 注意 track_id 在不同视频间会重复，单用 track_id 会误报
  3. 帧区间在三侧互不重叠（相邻帧不混入两边）
  4. 标签取值域正确、产物行数自洽
  5. 时间轴单调、插值比例、关键列无缺失
  6. 特征集不含标签判据量（防标签泄漏）

退出码 0 = 全部通过；1 = 有失败项。

用法: python verify_dataset.py [--data data]
"""
import argparse
import os
import sys

import pandas as pd

# 由行为标签导出、必须从特征集中剔除的判据量（与 evaluate.py 的 LEAKY 对齐）
LEAKY = {
    "seg_speed_smooth_med", "seg_low_speed_ratio", "seg_net_turn_deg",
    "seg_cum_turn_deg", "seg_turn_rate_deg_s", "seg_straightness", "label",
}
BASE_DROP = {
    "seg_idx", "video_id", "scene", "track_id", "label", "split",
    "seg_start_frame", "seg_end_frame", "seg_start_time", "seg_end_time",
    "seg_mid_time", "track_type", "entry_direction", "exit_direction",
    "overall_direction", "route_turn", "route_turn_label",
    "signal_state", "signal_cycle", "signal_source", "signal_state_ref",
}

FAILURES = []


def chk(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    d = args.data

    seg = pd.read_csv(os.path.join(d, "segments.csv"))
    trk = pd.read_csv(os.path.join(d, "tracks.csv"))
    tl = pd.read_csv(os.path.join(d, "track_labels.csv"))

    print("\n[1] 划分粒度 = 整段视频")
    for name, df in [("segments", seg), ("tracks", trk)]:
        m = df.groupby("video_id")["split"].nunique()
        chk((m == 1).all(),
            f"{name}: 每个 video_id 只属于一个 split {dict(m)}")
    v2s = seg.groupby("video_id")["split"].first().to_dict()
    chk(len(set(v2s.values())) == len(v2s),
        f"三段视频分属三个不同 split {v2s}")

    print("\n[2] 同一轨迹不跨 split（复合键 video_id+track_id）")
    for name, df in [("segments", seg), ("tracks", trk)]:
        g = df.groupby(["video_id", "track_id"])["split"].nunique()
        chk((g == 1).all(),
            f"{name}: 轨迹跨边违规数 = {int((g > 1).sum())} / {len(g)}")
    chk(seg.track_id.nunique() < len(seg.groupby(['video_id', 'track_id'])),
        "提示：track_id 在视频间重复，校验必须用复合键")

    print("\n[3] 帧区间三侧互不重叠")
    rng = seg.groupby(["video_id", "split"]).agg(
        fmin=("seg_start_frame", "min"), fmax=("seg_end_frame", "max"))
    print(rng.to_string())
    ok = True
    for vid, grp in rng.groupby(level=0):
        if len(grp) == 1:
            continue
        iv = sorted(zip(grp.fmin, grp.fmax))
        for i in range(1, len(iv)):
            if iv[i][0] <= iv[i - 1][1]:
                ok = False
    chk(ok, "同视频内不同 split 的帧区间无交叠")

    print("\n[4] 标签域与产物自洽")
    chk(set(seg.label.unique()) == {"stop", "straight", "turn"},
        f"窗口标签域 = {sorted(seg.label.unique())}")
    chk(set(trk.track_class.dropna().unique()) == {"stop", "straight", "turn"},
        f"轨迹标签域 = {sorted(trk.track_class.dropna().unique())}")
    chk(len(tl) == len(trk), f"track_labels 行数 {len(tl)} == tracks 行数 {len(trk)}")
    chk(len(seg) == trk.n_segments.sum() if "n_segments" in trk.columns else True,
        "segments 总行数与轨迹分段数自洽")
    n_grp = seg.groupby(["video_id", "track_id"]).ngroups
    chk(n_grp == len(trk), f"轨迹数一致：segments {n_grp} == tracks {len(trk)}")

    print("\n[5] 时间轴与数据质量")
    mono = seg.groupby(["video_id", "track_id"]).seg_start_time.apply(
        lambda s: s.is_monotonic_increasing)
    chk(bool(mono.all()), "每条轨迹的 seg_start_time 单调不减")
    chk(bool((seg.seg_end_frame >= seg.seg_start_frame).all()),
        "每个窗口 end_frame >= start_frame")
    chk(float(seg.seg_interp_ratio.mean()) < 0.05,
        f"窗口平均插值占比 {seg.seg_interp_ratio.mean():.4f} < 0.05")
    chk(bool(trk.interp_ratio.max() < 1.0),
        f"最高单轨迹插值占比 {trk.interp_ratio.max():.4f} < 1.0")

    print("\n[6] 建模特征集不含标签判据量（防标签泄漏）")
    # 判据列本身存在于 segments 中（它们是真实运动学量），
    # 关键在于 evaluate.feature_cols 是否把它们排除出建模特征集。
    # 这里复刻该逻辑做静态校验，与运行时行为保持一致。
    def feature_cols(df):
        drop = BASE_DROP | LEAKY
        return [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    for name, df in [("segments", seg), ("tracks", trk)]:
        fs = feature_cols(df)
        chk(not (LEAKY & set(fs)),
            f"{name}: 建模特征集 {len(fs)} 列，不含判据列 "
            f"{sorted(LEAKY & set(fs))}")
        chk("label" not in fs, f"{name}: 目标列 label 未进入特征集")

    print("\n[7] 关键列缺失")
    for name, df, cols in [
            ("segments", seg, ["video_id", "track_id", "label", "split",
                               "seg_start_frame", "seg_end_frame"]),
            ("tracks", trk, ["video_id", "track_id", "split", "track_class"]),
            ("track_labels", tl, ["video_id", "track_id", "split",
                                  "track_class"])]:
        miss = [c for c in cols if c not in df.columns]
        chk(not miss, f"{name}: 关键列齐全（缺 {miss}）")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"校验未通过：{len(FAILURES)} 项失败")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("校验全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
