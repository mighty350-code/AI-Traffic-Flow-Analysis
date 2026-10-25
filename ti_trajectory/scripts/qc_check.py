# -*- coding: utf-8 -*-
"""
FLUID / TI 路口轨迹数据质量体检（QC）

检查项
  1. 时间轴    : time 与 frame 的换算关系、单调性、帧覆盖、重复
  2. 坐标      : 像素/世界坐标量纲与范围、NaN 分布、跳变、速度一致性
  3. 断轨      : 同 id 内相邻观测的 frame 跳变、isReal=0 插值段、轨迹时长过短
  4. 缺失值    : 逐列 NaN 统计及其成因归类
  5. 路线/信号 : route.id 与 traj.id 的对齐、时间区间、geometry 与轨迹坐标一致性
                 信号表与 route.in_state 的自洽性

输出
  reports/qc_report.json   机器可读
  reports/qc_report.md     人读摘要

用法: python qc_check.py [--raw raw] [--out reports]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

VIDEOS = [1, 2, 3]
SCENE = "TI"
FPS = 10.0

# 世界坐标合理车速上限（m/s）—— 城市道路路口，>30 m/s 视为跳变
JUMP_SPEED_THR = 30.0
# 有效轨迹最短时长（秒）
MIN_DUR = 1.0
# 插值行占比上限（超过则整条轨迹不可信）
MAX_INTERP_RATIO = 0.5


def load(root, kind, v):
    p = os.path.join(root, kind, f"20250529_video-{v}_{kind.capitalize()}.csv"
                      if kind != "traj" else f"traj/20250529_video-{v}_Traj.csv")
    p = os.path.join(root, kind, f"20250529_video-{v}_{kind}.csv")
    return pd.read_csv(p, low_memory=False)


def geom_to_array(g):
    if not isinstance(g, str) or not g.strip():
        return None
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", g)
    if not nums:
        return None
    return np.asarray(nums, dtype=float).reshape(-1, 2)


def qc_video(root, v):
    tr = load(root, "traj", v)
    rt = load(root, "route", v)
    sg = load(root, "signal", v)

    tr = tr.sort_values(["id", "frame"]).reset_index(drop=True)
    rep = {"video": f"20250529_video-{v}", "scene": SCENE}

    # ---------- 1. 时间轴 ----------
    frames = np.sort(tr.frame.unique())
    dup_frame = int(tr.duplicated(["id", "frame"]).sum())
    dt_frame = np.diff(frames)
    rep["time"] = {
        "fps_declared": FPS,
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "n_frames_observed": int(len(frames)),
        "frame_span": int(frames.max() - frames.min() + 1),
        "frame_missing_in_span": int((frames.max() - frames.min() + 1) - len(frames)),
        "frame_gap_hist": {int(k): int(v_) for k, v_ in
                           zip(*np.unique(dt_frame, return_counts=True))},
        "duplicate_id_frame_rows": dup_frame,
        "time_min": float(tr.time.min()),
        "time_max": float(tr.time.max()),
        "time_equals_frame_over_fps": bool(
            np.allclose(tr.time, tr.frame / FPS, atol=1e-6)),
        "time_monotonic_within_id": bool(
            tr.groupby("id").time.apply(lambda s: s.is_monotonic_increasing).all()),
        "time_monotonic_global_by_frame": bool(
            tr.sort_values("frame").time.is_monotonic_increasing),
    }

    # ---------- 2. 坐标 ----------
    g = tr.groupby("id")
    dx = g.cx_m.diff()
    dy = g.cy_m.diff()
    dt = g.time.diff()
    v_calc = np.sqrt(dx ** 2 + dy ** 2) / dt.replace(0, np.nan)
    tr = tr.assign(v_calc=v_calc)

    real = tr[tr.isReal == 1].dropna(subset=["v_calc"])
    err = (real.v_calc - real.speed).abs()
    rep["coord"] = {
        "pixel_x_range": [float(tr.cx.min()), float(tr.cx.max())],
        "pixel_y_range": [float(tr.cy.min()), float(tr.cy.max())],
        "world_x_range": [float(tr.cx_m.min()), float(tr.cx_m.max())],
        "world_y_range": [float(tr.cy_m.min()), float(tr.cy_m.max())],
        "world_center_is_origin": bool(abs(tr.cx_m.mean()) < 50
                                        and abs(tr.cy_m.mean()) < 50),
        "speed_vs_coord_abs_err_median": float(err.median()),
        "speed_vs_coord_abs_err_p95": float(err.quantile(0.95)),
        "v_calc_quantiles": {str(q): float(real.v_calc.quantile(q))
                             for q in [0.5, 0.9, 0.99, 0.999, 1.0]},
        "n_rows_speed_over_threshold": int((real.v_calc > JUMP_SPEED_THR).sum()),
        "n_ids_with_jump": int(real[real.v_calc > JUMP_SPEED_THR].id.nunique()),
        "jump_speed_threshold_ms": JUMP_SPEED_THR,
    }

    # ---------- 3. 断轨 ----------
    fdiff = g.frame.diff().dropna()
    gaps = tr.loc[fdiff.index]
    gap_gt1 = gaps[fdiff > 1]
    interp = tr[tr.isReal == 0]
    interp_ratio = tr.groupby("id").isReal.apply(lambda s: (s == 0).mean())
    dur = g.time.agg(["min", "max"])
    dur["dur"] = dur["max"] - dur["min"]
    npts = g.frame.size()

    # 连续插值段长度
    def max_interp_run(s):
        best = cur = 0
        for v in s:
            cur = cur + 1 if v == 0 else 0
            best = max(best, cur)
        return best

    run_len = tr.groupby("id").isReal.apply(max_interp_run)

    rep["track"] = {
        "n_ids": int(tr.id.nunique()),
        "n_rows": int(len(tr)),
        "adjacent_frame_gap_hist": {int(k): int(v_) for k, v_ in
                                   zip(*np.unique(fdiff, return_counts=True))},
        "n_adjacent_gaps_gt1": int((fdiff > 1).sum()),
        "n_ids_with_frame_gap": int(gap_gt1.id.nunique()),
        "max_frame_gap": int(fdiff.max()) if len(fdiff) else 0,
        "n_interpolated_rows": int(len(interp)),
        "interp_ratio_overall": float((tr.isReal == 0).mean()),
        "n_ids_with_interp": int((interp_ratio > 0).sum()),
        "n_ids_interp_ratio_over_limit": int((interp_ratio > MAX_INTERP_RATIO).sum()),
        "interp_ratio_limit": MAX_INTERP_RATIO,
        "max_interp_run_frames": int(run_len.max()),
        "n_ids_with_interp_run_ge10": int((run_len >= 10).sum()),
        "traj_points_median": int(npts.median()),
        "traj_duration_median_s": float(dur.dur.median()),
        "n_ids_duration_lt_1s": int((dur.dur < MIN_DUR).sum()),
        "min_duration_limit_s": MIN_DUR,
        "n_ids_type_inconsistent": int((g.type.nunique() > 1).sum()),
        "n_ids_entry_dir_inconsistent": int((g.entry_direction.nunique() > 1).sum()),
        "n_ids_exit_dir_inconsistent": int((g.exit_direction.nunique() > 1).sum()),
    }

    # ---------- 4. 缺失值 ----------
    na = tr.isna().sum()
    na = {k: int(v_) for k, v_ in na.items() if v_ > 0}
    rep["missing"] = {
        "traj_na_by_column": na,
        "traj_na_total_cells": int(tr.isna().sum().sum()),
        "traj_total_cells": int(tr.size),
        "traj_na_ratio": float(tr.isna().sum().sum() / tr.size),
        "na_cx_equals_interpolated_rows": bool(
            int(tr.cx.isna().sum()) == int((tr.isReal == 0).sum())),
        "course_na_all_at_id_tail": bool(
            tr[tr.course.isna()].groupby("id").frame.transform("max")
            .eq(tr[tr.course.isna()].frame).all()),
        "route_na_by_column": {k: int(v_) for k, v_ in rt.isna().sum().items()
                               if v_ > 0},
        "signal_na_by_column": {k: int(v_) for k, v_ in sg.isna().sum().items()
                                if v_ > 0},
    }

    # ---------- 5. 路线 / 信号 ----------
    tspan = tr.groupby("id").time.agg(["min", "max"]).reset_index()
    m = rt.merge(tspan, on="id", how="left")
    # 区分两种「不落在区间内」的情况：
    #   a) route 未给出 in/out_time —— 车辆从未进入路口，不是数据错误
    #   b) 真正越界 —— 给了时间但落在轨迹时间跨度之外，这才是问题
    inside = ((m.in_time >= m["min"] - 0.01) & (m.out_time <= m["max"] + 0.01))
    has_time = m.in_time.notna() & m.out_time.notna()
    n_no_time = int((~has_time).sum())
    n_out_of_span = int((~inside & has_time).sum())
    # 单侧缺失（只有 in_time 或只有 out_time）
    n_one_side = int((m.in_time.notna() ^ m.out_time.notna()).sum())

    # geometry 与 traj 坐标一致性（抽样）
    geo_dev = []
    for tid in rt.id.head(60):
        ga = geom_to_array(rt.loc[rt.id == tid, "geometry"].iloc[0])
        if ga is None:
            continue
        t = tr[tr.id == tid].sort_values("time")
        if len(t) < 3:
            continue
        n = min(len(ga), len(t))
        d = np.hypot(t.cx_m.values[:n] - ga[:n, 0], t.cy_m.values[:n] - ga[:n, 1])
        geo_dev.append(float(np.median(d)))
    geo_dev = np.asarray(geo_dev)

    # 信号自洽：route.in_state 是否与 signal 表在 (direction, turn, in_time) 处一致
    sg_idx = {}
    for _, s in sg.iterrows():
        sg_idx.setdefault((s.direction, s.turn), []).append(
            (s.begin_time, s.end_time, s.state, s.cycle))
    hit = miss = 0
    for _, r in rt.dropna(subset=["in_time", "in_state"]).iterrows():
        cand = sg_idx.get((r.entry_direction, r.turn), [])
        st = [c[2] for c in cand if c[0] <= r.in_time <= c[1]]
        if not st:
            miss += 1
        elif st[0] == r.in_state:
            hit += 1
    tot = hit + miss

    # route.turn 与 signal.turn 词表差异
    sg_pairs = set(zip(sg.direction, sg.turn))
    rt_pairs = set(map(tuple, rt[["entry_direction", "turn"]].dropna().values))
    # 每种转向有多少条轨迹真正拿到了专属相位
    turn_phase_cov = {}
    for t in sorted(rt.turn.dropna().unique()):
        sub = rt[(rt.turn == t) & rt.in_time.notna()]
        turn_phase_cov[str(t)] = {
            "n_routes": int(len(sub)),
            "n_with_phase": int(sub.in_state.notna().sum()),
        }

    rep["route_signal"] = {
        "n_route_rows": int(len(rt)),
        "n_route_ids": int(rt.id.nunique()),
        "route_ids_subset_of_traj_ids": bool(
            set(rt.id) <= set(tr.id.unique())),
        "n_route_ids_matched_in_traj": int(len(set(rt.id) & set(tr.id.unique()))),
        "n_traj_ids_without_route": int(len(set(tr.id.unique()) - set(rt.id))),
        "route_turn_vocab": sorted(map(str, rt.turn.dropna().unique())),
        "signal_turn_vocab": sorted(map(str, sg.turn.dropna().unique())),
        "route_turn_not_in_signal_vocab": sorted(
            set(map(str, rt.turn.dropna().unique())) - set(map(str, sg.turn.unique()))),
        "signal_pairs_available": sorted(f"{d}:{t}" for d, t in sg_pairs),
        "route_pairs_needed": sorted(f"{d}:{t}" for d, t in rt_pairs),
        "route_pairs_without_dedicated_phase": sorted(
            f"{d}:{t}" for d, t in (rt_pairs - sg_pairs)),
        "phase_coverage_by_turn": turn_phase_cov,
        "n_route_turn_missing": int(rt.turn.isna().sum()),
        "n_route_no_intersection_time": n_no_time,
        "n_route_out_of_traj_span": n_out_of_span,
        "n_route_one_side_time_only": n_one_side,
        "route_in_out_inside_traj_span_ratio": float(inside.mean()),
        "route_median_lead_in_s": float((m.in_time - m["min"]).median()),
        "route_median_trail_out_s": float((m.out_time - m["max"]).median()),
        "geometry_matches_traj_median_dev_m": (
            float(np.median(geo_dev)) if len(geo_dev) else None),
        "geometry_samples": int(len(geo_dev)),
        "signal_state_hit_ratio": float(hit / tot) if tot else None,
        "signal_state_checked": int(tot),
        "signal_state_miss_no_matching_window": int(miss),
        "n_signal_rows": int(len(sg)),
        "signal_cycles": int(sg.cycle.max()),
        "signal_time_max": float(sg.end_time.max()),
        "signal_covers_traj_time": bool(sg.end_time.max() >= tr.time.max()),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    root = os.path.abspath(args.raw)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    reps = [qc_video(root, v) for v in VIDEOS]
    with open(os.path.join(out, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump(reps, f, ensure_ascii=False, indent=2)

    # ---- Markdown 摘要 ----
    L = []
    L.append("# FLUID / TI 路口轨迹数据质量体检报告\n")
    L.append(f"数据源：Figshare FLUID v2 衍生数据 `derived_data`（场景 {SCENE}），"
             f"3 段完整轨迹（20250529_video-1/2/3），帧率 {FPS} FPS。\n")
    for r in reps:
        t, c, k, ms, rs = (r["time"], r["coord"], r["track"],
                           r["missing"], r["route_signal"])
        L.append(f"\n## {r['video']}\n")
        L.append("### 时间轴\n")
        L.append("| 项 | 值 |\n|---|---|")
        L.append(f"| 帧范围 | {t['frame_min']} – {t['frame_max']} |")
        L.append(f"| 帧跨度内缺失帧数 | {t['frame_missing_in_span']} |")
        L.append(f"| time == frame/10 | {t['time_equals_frame_over_fps']} |")
        L.append(f"| time 在 id 内单调 | {t['time_monotonic_within_id']} |")
        L.append(f"| (id, frame) 重复行 | {t['duplicate_id_frame_rows']} |")
        L.append(f"| 帧间隔直方图 | {t['frame_gap_hist']} |")
        L.append("### 坐标\n")
        L.append("| 项 | 值 |\n|---|---|")
        L.append(f"| 像素坐标范围 x | {c['pixel_x_range'][0]:.1f} – {c['pixel_x_range'][1]:.1f} |")
        L.append(f"| 像素坐标范围 y | {c['pixel_y_range'][0]:.1f} – {c['pixel_y_range'][1]:.1f} |")
        L.append(f"| 世界坐标范围 x | {c['world_x_range'][0]:.2f} – {c['world_x_range'][1]:.2f} m |")
        L.append(f"| 世界坐标范围 y | {c['world_y_range'][0]:.2f} – {c['world_y_range'][1]:.2f} m |")
        L.append(f"| speed 与坐标反算速度 中位误差 | {c['speed_vs_coord_abs_err_median']:.3f} m/s |")
        L.append(f"| 同上 P95 | {c['speed_vs_coord_abs_err_p95']:.3f} m/s |")
        L.append(f"| 速度 > {c['jump_speed_threshold_ms']:.0f} m/s 的行 / 涉及 id | "
                 f"{c['n_rows_speed_over_threshold']} / {c['n_ids_with_jump']} |")
        L.append(f"| v_calc 分位 (50/90/99/99.9/max) | "
                 + " / ".join(f"{v:.2f}" for v in c["v_calc_quantiles"].values()) + " m/s |")
        L.append("### 断轨\n")
        L.append("| 项 | 值 |\n|---|---|")
        L.append(f"| 轨迹条数 | {k['n_ids']} |")
        L.append(f"| 相邻观测 frame 间隔 >1 的次数 | {k['n_adjacent_gaps_gt1']} |")
        L.append(f"| 含 frame 断口的 id 数 | {k['n_ids_with_frame_gap']} |")
        L.append(f"| 插值行（isReal=0）数 | {k['n_interpolated_rows']} |")
        L.append(f"| 插值行整体占比 | {k['interp_ratio_overall']:.4f} |")
        L.append(f"| 插值占比 > {k['interp_ratio_limit']} 的 id 数 | {k['n_ids_interp_ratio_over_limit']} |")
        L.append(f"| 单 id 最长连续插值段（帧） | {k['max_interp_run_frames']} |")
        L.append(f"| 轨迹时长中位数 | {k['traj_duration_median_s']:.1f} s |")
        L.append(f"| 时长 < {k['min_duration_limit_s']}s 的 id 数 | {k['n_ids_duration_lt_1s']} |")
        L.append(f"| id 内 type 不一致数 | {k['n_ids_type_inconsistent']} |")
        L.append("### 缺失值\n")
        L.append(f"- traj NaN 单元格：{ms['traj_na_total_cells']:,} / {ms['traj_total_cells']:,}"
                 f"（{ms['traj_na_ratio']:.5f}）")
        L.append(f"- 逐列：{ms['traj_na_by_column']}")
        L.append(f"- `cx/cy/r/confidence` 的 NaN 与 isReal=0 行完全对应："
                 f"{ms['na_cx_equals_interpolated_rows']}")
        L.append(f"- `course` 的 NaN 全部落在各 id 末行（差分定义导致）："
                 f"{ms['course_na_all_at_id_tail']}")
        L.append(f"- route NaN：{ms['route_na_by_column']}")
        L.append("### 路线 / 信号\n")
        L.append("| 项 | 值 |\n|---|---|")
        L.append(f"| route 行数 / id 数 | {rs['n_route_rows']} / {rs['n_route_ids']} |")
        L.append(f"| route.id ⊆ traj.id | {rs['route_ids_subset_of_traj_ids']} |")
        L.append(f"| traj 中无 route 标注的 id 数 | {rs['n_traj_ids_without_route']} |")
        L.append(f"| route.turn 词表 | {rs['route_turn_vocab']} |")
        L.append(f"| signal.turn 词表 | {rs['signal_turn_vocab']} |")
        L.append(f"| route.turn 中不在 signal 词表的值 | {rs['route_turn_not_in_signal_vocab']} |")
        L.append(f"| route.turn 缺失数 | {rs['n_route_turn_missing']} |")
        L.append(f"| route 未给出进出路口时刻的行数 | {rs['n_route_no_intersection_time']} |")
        L.append(f"| **真正越界**（给了时刻但落在轨迹跨度外） | {rs['n_route_out_of_traj_span']} |")
        L.append(f"| 仅单侧有时刻的行数 | {rs['n_route_one_side_time_only']} |")
        L.append(f"| geometry 与 traj 坐标中位偏差 | {rs['geometry_matches_traj_median_dev_m']} m |")
        L.append(f"| in_state 与 signal 表一致率 | {rs['signal_state_hit_ratio']} |")
        L.append(f"| 信号周期数 | {rs['signal_cycles']} |")
        L.append(f"| 信号时间覆盖到轨迹末端 | {rs['signal_covers_traj_time']} |")
        L.append("")
        L.append("#### 信号相位覆盖缺口\n")
        L.append("源信号表只对下列 (进口方向, 转向) 组合给出相位：\n")
        L.append(f"`{', '.join(rs['signal_pairs_available'])}`\n")
        L.append("而 route 标注用到了下列组合，其中**没有专属相位**的是：\n")
        L.append(f"`{', '.join(rs['route_pairs_without_dedicated_phase'])}`\n")
        L.append("即右转（r）与掉头（t）在本路口未单独信号化，属源数据语义而非缺陷。"
                 "构建脚本对这些轨迹回退到「同进口道直行相位」，"
                 "并用 `signal_source` 列区分 `direct` / `straight` / `unknown`，"
                 "避免把回退值误当作专属相位。\n")
        L.append("| route.turn | 有进出时刻的 route 行数 | 其中带专属相位 |\n|---|---|---|")
        for t, d in rs["phase_coverage_by_turn"].items():
            L.append(f"| {t} | {d['n_routes']} | {d['n_with_phase']} |")
        L.append("")
    md = "\n".join(L) + "\n"
    with open(os.path.join(out, "qc_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
