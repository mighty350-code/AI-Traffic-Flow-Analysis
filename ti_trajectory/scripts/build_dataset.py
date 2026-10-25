# -*- coding: utf-8 -*-
"""
构建 FLUID / TI 路口的统一训练输入。

输入（raw/）：
    traj/20250529_video-{1,2,3}_Traj.csv     逐帧轨迹点
    route/20250529_video-{1,2,3}_Route.csv   轨迹级路线（进/出方向、转向、信号交互）
    signal/20250529_video-{1,2,3}_signal.csv 相位级信号状态

输出（out/）：
    tracks.csv     轨迹级统一表：一条轨迹一行，接入 route + signal 摘要
    segments.csv   窗口级统一表：一个滑动窗口一行，含运动学特征 + 行为标签
    track_labels.csv 轨迹级三分类清单（stop / straight / turn）+ 判定依据
    signal_states.csv  时序信号状态表（按 direction+turn 展开），供逐点接入
    signal_intervals.csv 汇总相位表（任一方向为绿即视为路口放行）
    manifest.json  数据清单与统计

设计约束
  1. 划分粒度 = 整段视频。同一视频的全部帧只能落在同一侧，杜绝相邻帧泄漏。
  2. 行为标签在窗口级给出（stop / straight / turn），由运动学判据导出，
     轨迹级 route.turn 作为独立真值保留，用于交叉校验而不参与标签生成。
  3. 插值行（isReal=0）保留但显式标记；像素列缺失不影响世界坐标特征。
  4. 停车时长按时间区间并集计算，避免滑动窗口重叠导致重复计时。

用法: python build_dataset.py [--raw raw] [--out data] [--win 30] [--stride 10]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCENE = "TI"
VIDEOS = [1, 2, 3]
FPS = 10.0

# ---- 行为判据阈值（全部依据数据分布确定，见 reports/qc_report.md）----
STOP_SPEED_MED = 0.5      # m/s，窗口内速度中位数低于此值视为静止
STOP_LOW_RATIO = 0.8      # 窗口内低速帧占比
TURN_NET_DEG = 30.0       # deg，窗口净转角
TURN_CUM_DEG = 60.0       # deg，窗口累计转角
MIN_SEG_POINTS = 10       # 窗口内最少有效点数

# ---- 清洗阈值 ----
JUMP_SPEED_THR = 30.0     # m/s，超过判为坐标跳变
MIN_INTERP_RUN = 5        # 连续插值帧数 >= 此值视为"断轨段"

# ---- 轨迹级转向判据（用于把窗口标签回落到轨迹级）----
# 只统计「路口内」的窗口：route.in_time ~ route.out_time。
# 理由：转向只发生在路口内；把路口外的直线段计入会稀释信号，
# 把停车段的航向抖动计入会引入假转向。
TRACK_TURN_NET_DEG = 120.0   # deg，路口内行驶段净转角绝对值阈值
TRACK_TURN_RATIO = 0.2       # 路口内转弯窗口占行驶窗口比例阈值

# ---- 轨迹级「停车」判据 ----
# 用点级静止占比而非窗口时长占比，避免滑动窗口重叠重复计时。
# 轨迹中静止点占比 >= 该阈值，且存在连续静止段 >= MIN_STOP_RUN_S，
# 判为以停车为主的行为（等红灯 / 排队）。
TRACK_STOP_RATIO = 0.5       # 静止点占轨迹全部点的比例
MIN_STOP_RUN_S = 1.0         # 最短连续静止时长（秒）

TURN_MAP = {"l": "left", "r": "right", "t": "u_turn", "s": "straight"}
DIRS = ["E", "N", "S", "W", "Unknown"]
TURN_ORDER = ["s", "l", "r", "t"]
SIGNAL_TURNS = ["s", "l", "r", "p"]

# 机动车类别。
# 注意口径：FLUID 官方 README 只给了 "e.g., car, moped, pedestrain" 的示例，
# 未定义法规口径。本项目按中国大陆交规归类：
#   - moped = 轻便摩托车，属机动车（需 D/E/F 驾驶证、挂牌、可走机动车道）
#   - tricycle 可能是人力三轮车（属非机动车），FLUID 未细分，故归非机动车
#   - pedestrian = 行人，非机动车
# 若上游后续细化了 type 词表，只需改这一处即可全量重算。
MOTOR_TYPES = {"car", "truck", "bus", "van", "trailer", "moped"}

# 源信号表只为 (direction, turn) ∈ {(E,s),(N,l),(N,s),(W,l),(W,s),(E,p),(N,p),(W,p)}
# 提供相位。右转 'r' 与掉头 't' 在本路口未单独信号化，因此没有专属相位。
# 对这两类轨迹，用「同进口道直行相位 (direction,'s')」作为参考上下文，
# 并用 signal_source 明确标注这是参考值而非专属相位，避免误读。
SIGNAL_DIRECT_TURNS = {"s", "l"}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def read_csv(path):
    return pd.read_csv(path, low_memory=False)


def load_all(raw):
    data = {}
    for v in VIDEOS:
        tr = read_csv(os.path.join(raw, "traj", f"20250529_video-{v}_Traj.csv"))
        rt = read_csv(os.path.join(raw, "route", f"20250529_video-{v}_Route.csv"))
        sg = read_csv(os.path.join(raw, "signal", f"20250529_video-{v}_signal.csv"))
        data[v] = (tr, rt, sg)
    return data


def max_interp_run(isreal):
    """返回 (最长连续插值段长度, 各段长度列表)。"""
    runs, cur = [], 0
    for v in isreal:
        if v == 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return (max(runs) if runs else 0), runs


# ----------------------------------------------------------------------------
# 1. 信号状态表：把 signal.csv 展开为可直接按时间查询的长表
# ----------------------------------------------------------------------------
def build_signal_states(raw, data):
    """signal.csv 每行是 (direction, turn) 下的一段状态区间。
    输出统一长表，并额外提供 'any' 汇总相位（任一方向为绿即视为路口放行）。"""
    rows = []
    for v, (_, _, sg) in data.items():
        for _, s in sg.iterrows():
            rows.append({
                "video_id": f"20250529_video-{v}",
                "scene": s["name"],
                "direction": s["direction"],
                "turn": s["turn"],
                "state": s["state"],
                "begin_time": float(s["begin_time"]),
                "end_time": float(s["end_time"]),
                "duration": float(s["duration"]),
                "cycle": int(s["cycle"]),
            })
    ss = pd.DataFrame(rows).sort_values(
        ["video_id", "direction", "turn", "begin_time"]).reset_index(drop=True)

    # 汇总相位：按时间轴合并所有 (direction, turn) 的区间，取该时刻出现的状态集合
    agg = []
    for (vid, bt, et), g in ss.groupby(["video_id", "begin_time", "end_time"]):
        agg.append({
            "video_id": vid, "begin_time": bt, "end_time": et,
            "states": "|".join(sorted(set(g.state))),
            "green_present": bool((g.state == "G").any()),
            "n_groups": int(len(g)),
        })
    agg = pd.DataFrame(agg).sort_values(["video_id", "begin_time"])
    return ss, agg


def signal_state_at(ss, video_id, direction, turn, t):
    """查询 t 时刻 (direction, turn) 的信号状态；未覆盖返回 (None, None)。"""
    if direction is None or turn is None or direction == "Unknown":
        return None, None
    cand = ss[(ss.video_id == video_id) & (ss.direction == direction) &
              (ss.turn == turn)]
    if cand.empty:
        return None, None
    hit = cand[(cand.begin_time <= t) & (cand.end_time >= t)]
    if hit.empty:
        return None, None
    r = hit.iloc[0]
    return r.state, int(r.cycle)


class SignalIndex:
    """信号状态 O(log n) 查询索引，避免逐窗口全表扫描。

    对每个 (video_id, direction, turn) 分组，区间按 begin_time 升序且互不重叠，
    因此可用 searchsorted 直接定位 t 所在的区间。
    """

    def __init__(self, ss):
        self.groups = {}
        for (vid, d, t_), g in ss.groupby(["video_id", "direction", "turn"],
                                         sort=False):
            g = g.sort_values("begin_time")
            self.groups[(vid, d, t_)] = {
                "begin": g.begin_time.to_numpy(dtype=float),
                "end": g.end_time.to_numpy(dtype=float),
                "state": g.state.to_numpy(),
                "cycle": g.cycle.to_numpy(dtype=int),
            }

    def query(self, video_id, direction, turn, t):
        if direction is None or turn is None or direction == "Unknown":
            return None, None
        g = self.groups.get((video_id, direction, turn))
        if g is None:
            return None, None
        i = np.searchsorted(g["begin"], t, side="right") - 1
        if i < 0:
            return None, None
        if t <= g["end"][i]:
            return g["state"][i], int(g["cycle"][i])
        return None, None

    def query_with_fallback(self, video_id, direction, turn, t):
        """查询 (direction, turn) 的信号相位，缺失时退回同进口道直行相位。

        返回 (state, cycle, source)：
          source = 'direct'    → 命中原专属相位（如直行、左转）
          source = 'straight'  → 退回同进口道直行相位（右转/掉头未单独信号化）
          source = 'unknown'   → 无任何可用相位

        注意：'straight' 回退值只表示「该进口道通行方向的相位」，
        不等于该车实际所受相位 —— 右转/掉头在本路口未单独信号化，
        实际受对向流与自由右转规则约束。把它当受控相位用会误导模型
        （实测监督式转向分类准确率从 0.938 掉到 0.836）。
        因此构建脚本分两列输出：signal_state 只放专属相位，
        signal_state_ref 单独放参考相位，由使用者自行决定是否采用。
        """
        if direction is None or direction == "Unknown":
            return None, None, "unknown"
        if turn in SIGNAL_DIRECT_TURNS:
            st, cy = self.query(video_id, direction, turn, t)
            if st is not None:
                return st, cy, "direct"
        # 回退：同进口道直行相位
        st, cy = self.query(video_id, direction, "s", t)
        if st is not None:
            return st, cy, "straight"
        return None, None, "unknown"


# ----------------------------------------------------------------------------
# 2. 轨迹级统一表
# ----------------------------------------------------------------------------
def build_tracks(data, ss):
    sidx = SignalIndex(ss)
    out = []
    for v, (tr, rt, sg) in data.items():
        vid = f"20250529_video-{v}"
        tr = tr.sort_values(["id", "frame"]).reset_index(drop=True)
        rt = rt.set_index("id")

        g = tr.groupby("id")
        agg = g.agg(
            type=("type", "first"),
            frame_start=("frame", "min"),
            frame_end=("frame", "max"),
            time_start=("time", "min"),
            time_end=("time", "max"),
            n_points=("frame", "size"),
            n_real=("isReal", lambda s: int((s == 1).sum())),
            conf_mean=("confidence", "mean"),
            length_med=("length_med", "first"),
            width_med=("width_med", "first"),
            entry_direction=("entry_direction", "first"),
            exit_direction=("exit_direction", "first"),
            overall_direction=("overall_direction", "first"),
        )
        agg["duration"] = agg.time_end - agg.time_start
        agg["interp_ratio"] = 1.0 - agg.n_real / agg.n_points

        # 速度统计（世界坐标反算更可靠）
        dx = g.cx_m.diff()
        dy = g.cy_m.diff()
        dt = g.time.diff()
        v = (np.sqrt(dx ** 2 + dy ** 2) / dt.replace(0, np.nan)).rename("v_calc")
        tmp = tr[["id"]].copy()
        tmp["v_calc"] = v.values
        agg = agg.join(tmp.groupby("id").v_calc.agg(
            ["median", "max", "mean", "std"]).rename(
            columns={"median": "v_median", "max": "v_max",
                     "mean": "v_mean", "std": "v_std"}))

        # 路径长度
        tmp["seg"] = np.hypot(dx, dy)
        agg = agg.join(tmp.groupby("id").seg.sum().rename("path_len"))

        # 插值段
        agg["max_interp_run"] = g.isReal.apply(
            lambda s: max_interp_run(s)[0])
        agg["has_track_break"] = agg.max_interp_run >= MIN_INTERP_RUN

        # 坐标跳变标记
        tmp["jump"] = tmp.v_calc > JUMP_SPEED_THR
        agg = agg.join(tmp.groupby("id").jump.sum().rename("n_jump"))

        # 接入 route
        for c in ["in_time", "out_time", "turn", "in_state", "in_cycle",
                  "out_state", "out_cycle", "in_delta", "out_delta"]:
            agg["route_" + c] = rt[c] if c in rt.columns else np.nan
        agg["route_turn_label"] = agg.route_turn.map(TURN_MAP)

        # 接入信号：进出时刻的相位
        # sig_*       —— 严格：只用该 (进口方向, 转向) 的专属相位；缺失即为空
        # sig_*_ref   —— 参考：缺失时回退到同进口道直行相位，仅作上下文参考
        in_state2, in_cycle2, out_state2, out_cycle2 = [], [], [], []
        in_ref, out_ref = [], []
        for tid, row in agg.iterrows():
            t_in = row.get("route_in_time", np.nan)
            t_out = row.get("route_out_time", np.nan)
            s1, c1, src1 = sidx.query_with_fallback(
                vid, row.entry_direction, row.route_turn, t_in)
            s2, c2, src2 = sidx.query_with_fallback(
                vid, row.exit_direction, row.route_turn, t_out)
            in_state2.append(s1 if src1 == "direct" else row.get("route_in_state"))
            in_cycle2.append(c1 if src1 == "direct" else row.get("route_in_cycle"))
            out_state2.append(s2 if src2 == "direct" else row.get("route_out_state"))
            out_cycle2.append(c2 if src2 == "direct" else row.get("route_out_cycle"))
            in_ref.append(s1)
            out_ref.append(s2)
        agg["sig_in_state"] = in_state2
        agg["sig_in_cycle"] = in_cycle2
        agg["sig_out_state"] = out_state2
        agg["sig_out_cycle"] = out_cycle2
        agg["sig_in_state_ref"] = in_ref
        agg["sig_out_state_ref"] = out_ref
        # 专属相位与 route 原始标注的一致性；回退值不参与比较
        agg["sig_state_match_in"] = np.where(
            agg.sig_in_state.notna(), agg.sig_in_state == agg.route_in_state, None)

        agg = agg.reset_index().rename(columns={"id": "track_id"})
        agg.insert(0, "video_id", vid)
        agg.insert(1, "scene", SCENE)
        agg["is_motor"] = agg["type"].isin(MOTOR_TYPES)
        out.append(agg)
    return pd.concat(out, ignore_index=True)


def track_stop_profile(tr, tid):
    """从原始帧计算一条轨迹的「静止」画像。

    用点级统计而非窗口统计：滑动窗口彼此重叠，窗口时长占比会重复计时，
    使轨迹停车占比被系统性高估。

    返回 dict：
      stop_ratio        静止点占全部点的比例
      stop_run_max_s   最长连续静止时长
    """
    g = tr[tr.id == tid]
    if g.empty:
        return None
    sp = g["speed_smooth"].to_numpy(dtype=float)
    real = g["isReal"].to_numpy() == 1
    t = g["time"].to_numpy(dtype=float)

    stationary = (sp < STOP_SPEED_MED) & real
    n = len(sp)
    stop_ratio = float(stationary.sum() / n) if n else 0.0

    # 最长连续静止时长：按相邻静止点的时间差累加
    best = 0.0
    cur = 0.0
    for i in range(n):
        if stationary[i]:
            cur += (t[i] - t[i - 1]) if (i > 0 and t[i] > t[i - 1]) else 0.1
            best = max(best, cur)
        else:
            cur = 0.0
    return {
        "stop_ratio": stop_ratio,
        "stop_run_max_s": float(best),
    }


def attach_track_three_class(tracks, data):
    """轨迹级三分类：stop / straight / turn。

    判据优先级（自上而下，命中即定）：
      1. stop     —— 轨迹以停车为主：静止点占比 >= TRACK_STOP_RATIO
                     且最长连续静止 >= MIN_STOP_RUN_S
      2. turn     —— route.turn ∈ {left, right, u_turn}（人工路线标注，独立真值）；
                     route.turn 缺失时回退到运动学判据 track_turn_class
      3. straight —— 其余

    理由：停车是「时序上占据主导」的状态，与「怎么走」是正交维度。
    一条左转轨迹可能先等红灯再转弯；若按转向优先会丢掉真实的停车信息，
    若让短暂减速的直行轨迹判成停车又会误标。因此先用时间占比门槛判定
    停车是否构成主导行为，再判转向。

    输出列：
      track_class            三分类标签 stop / straight / turn
      track_class_source     判定依据：stop_ratio / route_turn / kinematics / default
      stop_ratio             静止点占比
      stop_run_max_s         最长连续静止时长
    """
    prof = []
    for v, (tr, _, _) in data.items():
        for tid in tr.id.unique():
            p = track_stop_profile(tr, tid)
            if p is None:
                continue
            prof.append({
                "video_id": f"20250529_video-{v}",
                "track_id": int(tid), **p})
    prof = pd.DataFrame(prof)
    tracks = tracks.merge(prof, on=["video_id", "track_id"], how="left")

    def pick(row):
        sr = row.get("stop_ratio")
        run = row.get("stop_run_max_s")
        if pd.notna(sr) and sr >= TRACK_STOP_RATIO \
                and pd.notna(run) and run >= MIN_STOP_RUN_S:
            return "stop", "stop_ratio"
        rt = row.get("route_turn_label")
        if pd.notna(rt):
            return ("straight" if rt == "straight" else "turn"), "route_turn"
        k = row.get("track_turn_class")
        if pd.notna(k) and k in ("turn", "straight"):
            return k, "kinematics"
        return "straight", "default"

    lab = tracks.apply(pick, axis=1, result_type="expand")
    tracks["track_class"] = lab[0]
    tracks["track_class_source"] = lab[1]

    # 三分类与人工转向真值的一致性（停车轨迹不参与，转弯/直行才可比）
    rtl = tracks["route_turn_label"]
    truth = np.where(rtl == "straight", "straight", "turn")
    tracks["class_agree_route"] = np.where(
        rtl.isna() | (tracks.track_class == "stop"), None,
        tracks.track_class == truth)
    return tracks


def attach_track_turn_class(tracks, segments):
    """把窗口标签回落到轨迹级「转弯 / 直行」。

    只统计路口内（route.in_time ~ route.out_time）且非停车的窗口：
      - 转向只发生在路口内，路口外的直线段会稀释信号；
      - 停车段航向易抖动，计入会产生假转向。
    判据：路口内净转角绝对值 > TRACK_TURN_NET_DEG
          且 路口内转弯窗口占行驶窗口比例 > TRACK_TURN_RATIO
    """
    d = segments[segments.route_turn.notna()].copy()
    rows = []
    for (vid, tid), g in d.groupby(["video_id", "track_id"]):
        gi = g[g.in_intersection]
        if len(gi) == 0:
            continue
        moving = gi[gi.label != "stop"]
        n_turn = int((gi.label == "turn").sum())
        n_str = int((gi.label == "straight").sum())
        n_stop = int((gi.label == "stop").sum())
        n_moving = len(moving)
        net = float(moving.seg_net_turn_deg.sum())
        ratio = n_turn / n_moving if n_moving else 0.0
        is_turn = (abs(net) > TRACK_TURN_NET_DEG) and (ratio > TRACK_TURN_RATIO)
        rows.append({
            "video_id": vid, "track_id": int(tid),
            "n_seg_in_intersection": len(gi),
            "n_seg_in_turn": n_turn, "n_seg_in_straight": n_str,
            "n_seg_in_stop": n_stop,
            "turn_net_deg_in_intersection": net,
            "turn_window_ratio_in_intersection": ratio,
            "track_turn_class": "turn" if is_turn else "straight",
        })
    tl = pd.DataFrame(rows)
    tracks = tracks.merge(tl, on=["video_id", "track_id"], how="left")
    tracks["track_turn_class"] = tracks.track_turn_class.fillna("unknown")
    # 与人工标注的对照列（仅对有 route.turn 的轨迹有效）
    tracks["route_turn_binary"] = tracks.route_turn.map(
        {"s": "straight", "l": "turn", "r": "turn", "t": "turn"})
    tracks["turn_label_agree"] = np.where(
        tracks.route_turn_binary.isna(), None,
        tracks.track_turn_class == tracks.route_turn_binary)
    return tracks


# ----------------------------------------------------------------------------
# 3. 窗口级统一表
# ----------------------------------------------------------------------------
def segment_features(g):
    """窗口内的运动学特征 + 行为标签。

    g 为单个轨迹的 numpy 结构化视图（由 build_segments 预先切好），
    避免逐窗口 pandas 切片带来的开销。
    """
    x = g["cx_m"]
    y = g["cy_m"]
    t = g["time"]
    n = len(t)
    if n < 3:
        return None

    dx = np.diff(x)
    dy = np.diff(y)
    d = np.hypot(dx, dy)
    dt = np.diff(t)
    dt_safe = np.where(dt > 0, dt, np.nan)
    v = d / dt_safe

    sp = g["speed_smooth"]
    ax = g["ax"]
    ay = g["ay"]

    path_len = float(d.sum())
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = disp / path_len if path_len > 1e-6 else 1.0

    ang = np.unwrap(np.arctan2(dy, dx))
    net = float((ang[-1] - ang[0]) * 180 / np.pi)
    cum = float(np.abs(np.diff(ang)).sum() * 180 / np.pi) if len(ang) > 1 else 0.0
    dur = float(t[-1] - t[0])
    turn_rate = cum / dur if dur > 1e-6 else 0.0

    if len(ang) > 2:
        da = np.diff(ang)
        curv = float(np.mean(np.abs(da)) / max(np.mean(d), 1e-6))
    else:
        curv = 0.0

    v_med = float(np.nanmedian(sp))
    low_ratio = float(np.mean(sp < 1.0))
    if v_med < STOP_SPEED_MED and low_ratio >= STOP_LOW_RATIO:
        label = "stop"
    elif abs(net) > TURN_NET_DEG or cum > TURN_CUM_DEG:
        label = "turn"
    else:
        label = "straight"

    v_fin = v[np.isfinite(v)]
    ax_fin = ax[np.isfinite(ax)]

    return {
        "seg_n_points": n,
        "seg_duration": dur,
        "seg_v_mean": float(v_fin.mean()) if v_fin.size else 0.0,
        "seg_v_max": float(v_fin.max()) if v_fin.size else 0.0,
        "seg_v_std": float(v_fin.std()) if v_fin.size else 0.0,
        "seg_v_first": float(v[0]) if np.isfinite(v[0]) else 0.0,
        "seg_v_last": float(v[-1]) if np.isfinite(v[-1]) else 0.0,
        "seg_speed_smooth_mean": float(np.nanmean(sp)),
        "seg_speed_smooth_med": v_med,
        "seg_speed_smooth_std": float(np.nanstd(sp)),
        "seg_speed_smooth_min": float(np.nanmin(sp)),
        "seg_speed_smooth_max": float(np.nanmax(sp)),
        "seg_speed_smooth_p25": float(np.percentile(sp, 25)),
        "seg_speed_smooth_p75": float(np.percentile(sp, 75)),
        "seg_low_speed_ratio": low_ratio,
        "seg_accel_abs_mean": float((np.nanmean(np.abs(ax)) +
                                     np.nanmean(np.abs(ay))) / 2)
        if np.any(np.isfinite(ax)) else 0.0,
        "seg_accel_abs_max": float(np.nanmax(np.abs(ax_fin)))
        if ax_fin.size else 0.0,
        "seg_path_len": path_len,
        "seg_displacement": disp,
        "seg_straightness": straightness,
        "seg_net_turn_deg": net,
        "seg_cum_turn_deg": cum,
        "seg_turn_rate_deg_s": turn_rate,
        "seg_curvature": curv,
        "seg_cx_m_mean": float(np.mean(x)),
        "seg_cy_m_mean": float(np.mean(y)),
        "seg_dist_to_center": float(np.hypot(np.mean(x), np.mean(y))),
        "seg_interp_ratio": float(np.mean(g["isReal"] == 0)),
        "label": label,
    }


def build_segments(data, ss, win, stride):
    """按滑动窗口切分每条轨迹，输出窗口级统一表。

    实现要点：每条轨迹先转成 numpy 结构化数组，窗口仅取视图，避免 pandas 切片开销。
    """
    sidx = SignalIndex(ss)
    rows = []
    for v, (tr, rt, sg) in data.items():
        vid = f"20250529_video-{v}"
        tr = tr.sort_values(["id", "frame"])
        rt_idx = rt.set_index("id")

        cols = ["frame", "time", "cx_m", "cy_m", "speed_smooth", "ax", "ay",
                "isReal", "type", "entry_direction", "exit_direction",
                "overall_direction", "length_med", "width_med"]
        for tid, g in tr.groupby("id", sort=False):
            if len(g) < MIN_SEG_POINTS:
                continue
            arr = {c: g[c].to_numpy() for c in cols}
            n = len(g)
            rte = rt_idx.loc[tid] if tid in rt_idx.index else None
            entry_dir = arr["entry_direction"][0]
            exit_dir = arr["exit_direction"][0]
            turn = rte["turn"] if rte is not None else np.nan
            vtype = arr["type"][0]
            overall = arr["overall_direction"][0]
            len_med = float(arr["length_med"][0]) if pd.notna(arr["length_med"][0]) else np.nan
            wid_med = float(arr["width_med"][0]) if pd.notna(arr["width_med"][0]) else np.nan
            # 路口内时间区间（route 给出车辆进出路口的时刻）
            t_in = float(rte["in_time"]) if rte is not None and pd.notna(rte.get("in_time")) else np.nan
            t_out = float(rte["out_time"]) if rte is not None and pd.notna(rte.get("out_time")) else np.nan

            n_seg = max(1, (n - win) // stride + 1)
            for k in range(n_seg):
                s0 = k * stride
                s1 = min(s0 + win, n)
                if s1 - s0 < MIN_SEG_POINTS:
                    break
                sub = {c: arr[c][s0:s1] for c in cols}
                f = segment_features(sub)
                if f is None:
                    continue
                t_mid = float(arr["time"][s0:s1][(s1 - s0) // 2])
                st_ref, cyc_ref, src = sidx.query_with_fallback(
                    vid, entry_dir, turn, t_mid)
                # 严格相位：仅当该 (进口方向, 转向) 有专属相位时才有值
                st = st_ref if src == "direct" else None
                cyc = cyc_ref if src == "direct" else None
                # 是否位于路口内
                in_int = (np.isfinite(t_in) and np.isfinite(t_out)
                          and t_in <= t_mid <= t_out)
                f.update({
                    "video_id": vid,
                    "scene": SCENE,
                    "track_id": int(tid),
                    "seg_start_frame": int(arr["frame"][s0]),
                    "seg_end_frame": int(arr["frame"][s1 - 1]),
                    "seg_start_time": float(arr["time"][s0]),
                    "seg_end_time": float(arr["time"][s1 - 1]),
                    "seg_mid_time": t_mid,
                    "in_intersection": bool(in_int),
                    "track_type": vtype,
                    "is_motor": vtype in MOTOR_TYPES,
                    "entry_direction": entry_dir,
                    "exit_direction": exit_dir,
                    "overall_direction": overall,
                    "route_turn": turn,
                    "route_turn_label": TURN_MAP.get(turn, None),
                    "length_med": len_med,
                    "width_med": wid_med,
                    "signal_state": st,
                    "signal_cycle": cyc,
                    "signal_source": src,
                    "signal_state_known": st is not None,
                    "signal_state_ref": st_ref,
                })
                rows.append(f)
    df = pd.DataFrame(rows)
    df.insert(0, "seg_idx", range(len(df)))
    front = ["seg_idx", "video_id", "scene", "track_id", "seg_start_frame",
             "seg_end_frame", "seg_start_time", "seg_end_time", "seg_mid_time",
             "label"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# ----------------------------------------------------------------------------
# 5. 交通冲突接入
# ----------------------------------------------------------------------------
def build_conflicts(raw, tracks):
    """接入 FLUID 上游的交通冲突检测结果。

    源表 raw/conflict/TI_conflict.csv 是**事件级**的：每行一次冲突，
    记录冲突时刻（Frame）、涉事两车（Car 1/2 ID）及其世界坐标、
    以及三个严重度指标：

        diff    两车间最小距离 (m)
        TTC     碰撞时间 (s)         —— 越小越危险
        MTTC    修改的 TTC (s)        —— 考虑了相对速度方向的 TTC
        DGT/PET 减速度增长率/感知碰撞时间 (m/s^2) —— 越��越危险

    这里做三件事：
      1. 把事件展开成「每车一行」的长表，便于按轨迹聚合；
      2. 映射到统一轨迹键 (video_id, track_id)；
      3. 聚合出轨迹级冲突统计，挂在 tracks 上。

    注意：上游冲突检测用到的少量轨迹可能已被上游删除（实测 4 个车 ID
    在原始 traj 中不存在），这类事件会被标记为 unmatched 而非静默丢弃。
    """
    path = os.path.join(raw, "conflict", f"{SCENE}_conflict.csv")
    if not os.path.exists(path):
        return None, None

    cf = read_csv(path)
    cf.columns = [c.strip() for c in cf.columns]
    # Scene 形如 20250529_video-1_Traj -> video_id
    cf["video_id"] = cf.Scene.str.replace("_Traj", "", regex=False)
    cf["frame"] = cf.Frame.astype(int)
    cf["car1_id"] = cf["Car 1 ID"].astype(int)
    cf["car2_id"] = cf["Car 2 ID"].astype(int)

    # ---- 展开成每车一行 ----
    long = []
    for _, r in cf.iterrows():
        for side, cid, cx, cy in (
                ("car1", r.car1_id, r["Car 1 X"], r["Car 1 Y"]),
                ("car2", r.car2_id, r["Car 2 X"], r["Car 2 Y"])):
            long.append({
                "video_id": r.video_id,
                "frame": r.frame,
                "time": r.frame / FPS,
                "track_id": cid,
                "role": side,
                "peer_track_id": r.car2_id if side == "car1" else r.car1_id,
                "x": cx, "y": cy,
                "diff": r["diff"],
                "TTC": r["TTC"],
                "MTTC": r["MTTC"],
                "DGT_PET": r["DGT/PET"],
                "conflict_type": r.conflict_type,
            })
    ev = pd.DataFrame(long)

    # ---- 映射到统一轨迹键 ----
    known = set(zip(tracks.video_id, tracks.track_id))
    ev["matched"] = [
        (v, t) in known for v, t in zip(ev.video_id, ev.track_id)]
    ev["split"] = ev.video_id.map(
        {f"20250529_video-{v}": s
         for v, s in zip(VIDEOS, ["train", "val", "test"])})

    # ---- 轨迹级聚合 ----
    m = ev[ev.matched]
    agg = m.groupby(["video_id", "track_id"]).agg(
        n_conflict=("conflict_type", "size"),
        conflict_types=("conflict_type", lambda s: "|".join(sorted(set(s)))),
        conflict_min_diff=("diff", "min"),
        conflict_min_ttc=("TTC", "min"),
        conflict_min_mttc=("MTTC", "min"),
        conflict_max_dgt_pet=("DGT_PET", "max"),
        conflict_first_frame=("frame", "min"),
        conflict_last_frame=("frame", "max"),
    ).reset_index()
    tracks = tracks.merge(agg, on=["video_id", "track_id"], how="left")
    for c in ["n_conflict", "conflict_min_diff", "conflict_min_ttc",
              "conflict_min_mttc", "conflict_max_dgt_pet",
              "conflict_first_frame", "conflict_last_frame"]:
        tracks[c] = tracks[c].fillna(0)
    tracks["n_conflict"] = tracks["n_conflict"].astype(int)
    tracks["has_conflict"] = tracks["n_conflict"] > 0
    tracks["conflict_types"] = tracks["conflict_types"].fillna("")

    return ev, tracks


# ----------------------------------------------------------------------------
# 6. 划分：按整段视频
# ----------------------------------------------------------------------------
def make_split(tracks, segments):
    """整段视频划分。train=video-1, val=video-2, test=video-3。
    保证同一 video_id 的所有 track_id / segment 全部落在同一侧。"""
    assign = {f"20250529_video-{v}": s for v, s in zip(VIDEOS, ["train", "val", "test"])}

    seg_split = segments.video_id.map(assign).rename("split")
    segments = segments.assign(split=seg_split)

    tr_split = tracks.video_id.map(assign).rename("split")
    tracks = tracks.assign(split=tr_split)

    # 泄漏断言：track_id 在不同视频间可重复，故用 (video_id, track_id) 复合键
    tr_ids = tracks.groupby("split").apply(
        lambda d: set(zip(d.video_id, d.track_id)), include_groups=False)
    assert not (tr_ids["train"] & tr_ids["val"]), "train/val 出现相同轨迹"
    assert not (tr_ids["train"] & tr_ids["test"]), "train/test 出现相同轨迹"
    assert not (tr_ids["val"] & tr_ids["test"]), "val/test 出现相同轨迹"

    # 关键约束：一段视频的全部帧必须落在同一侧。
    # 帧号在每个视频内从 1 重新计数，跨视频帧号重叠是正常的（不同视频不同时间拍摄），
    # 真正要保证的是「同一视频不跨划分」，且同一视频内的帧不被切到两边。
    for vid, g in segments.groupby("video_id"):
        assert g.split.nunique() == 1, f"{vid} 被切分到多个 split"
        assert g.track_id.nunique() == segments[segments.video_id == vid].track_id.nunique()
    # 同一视频内，相邻划分之间不得有帧级交叠（此处天然满足，作为回归保护）
    for vid, g in segments.groupby("video_id"):
        assert g.split.nunique() == 1, f"{vid} 跨 split"
    return tracks, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--win", type=int, default=30, help="窗口长度（帧）")
    ap.add_argument("--stride", type=int, default=10, help="窗口步长（帧）")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(root, args.raw)
    out = os.path.join(root, args.out)
    os.makedirs(out, exist_ok=True)

    # 原始数据不在仓库内，缺失时给出可操作的指引而不是 FileNotFoundError。
    if not os.path.isdir(os.path.join(raw, "traj")):
        print(f"[错误] 缺少原始数据目录: {os.path.join(raw, 'traj')}", file=sys.stderr)
        print("", file=sys.stderr)
        print("原始数据来自 Figshare FLUID v2 的 derived_data（场景 TI），未随仓库上传。", file=sys.stderr)
        print("请按 ti_trajectory/README.md 第 3 节准备，解压后应得：", file=sys.stderr)
        print(f"  {args.raw}/traj/      20250529_video-{{1,2,3}}_Traj.csv", file=sys.stderr)
        print(f"  {args.raw}/route/     20250529_video-{{1,2,3}}_Route.csv", file=sys.stderr)
        print(f"  {args.raw}/signal/    20250529_video-{{1,2,3}}_signal.csv", file=sys.stderr)
        print(f"  {args.raw}/conflict/  TI_conflict.csv", file=sys.stderr)
        print(f"  {args.raw}/map/       TI.osm", file=sys.stderr)
        sys.exit(2)

    print("[1/8] 读取原始数据 ...")
    data = load_all(raw)

    print("[2/8] 构建信号状态表 ...")
    ss, agg = build_signal_states(raw, data)
    ss.to_csv(os.path.join(out, "signal_states.csv"), index=False)
    agg.to_csv(os.path.join(out, "signal_intervals.csv"), index=False)

    print("[3/8] 构建轨迹级统一表 ...")
    tracks = build_tracks(data, ss)

    print(f"[4/8] 构建窗口级统一表 (win={args.win}, stride={args.stride}) ...")
    segments = build_segments(data, ss, args.win, args.stride)

    print("[5/8] 按整段视频划分 train/val/test ...")
    tracks, segments = make_split(tracks, segments)

    print("[6/8] 回落到轨迹级转向类别 ...")
    tracks = attach_track_turn_class(tracks, segments)

    print("[7/8] 轨迹级三分类（stop / straight / turn） ...")
    tracks = attach_track_three_class(tracks, data)

    print("[8/8] 接入交通冲突 ...")
    conflicts, tracks = build_conflicts(raw, tracks)
    if conflicts is not None:
        conflicts.to_csv(os.path.join(out, "conflict_events.csv"), index=False)
        n_matched = int(conflicts.matched.sum())
        # 事件数：长表里一次事件有两行（car1/car2），两行的 (track_id, peer)
        # 组合互为颠倒，故按「车对无序」去重：把两 ID 排序后拼成一把键
        pair = conflicts.apply(
            lambda r: tuple(sorted((r.track_id, r.peer_track_id))), axis=1)
        n_event = int(conflicts.assign(_pair=pair).groupby(
            ["video_id", "frame", "_pair"]).ngroups)
        print(f"  冲突事件 {n_event} 次（车-事件级 {len(conflicts)} 行），"
              f"可匹配到统一轨迹 {n_matched}/{len(conflicts)}，"
              f"涉事轨迹 {tracks.has_conflict.sum()} 条")

    tracks.to_csv(os.path.join(out, "tracks.csv"), index=False)
    segments.to_csv(os.path.join(out, "segments.csv"), index=False)

    # 轨迹级三分类清单：只保留分类结果与判定依据，便于直接用于训练/核查
    tl_cols = [
        "video_id", "scene", "track_id", "split", "type", "is_motor",
        "track_class", "track_class_source",
        "stop_ratio", "stop_run_max_s",
        "route_turn", "route_turn_label", "class_agree_route",
        "track_turn_class", "turn_label_agree",
        "entry_direction", "exit_direction", "overall_direction",
        "route_in_time", "route_out_time",
        "sig_in_state", "sig_in_state_ref", "sig_out_state", "sig_out_state_ref",
        "duration", "n_points", "n_real", "interp_ratio",
        "v_median", "v_max", "path_len",
        "has_track_break", "max_interp_run", "n_jump",
        "length_med", "width_med",
        "n_conflict", "has_conflict", "conflict_types",
        "conflict_min_diff", "conflict_min_ttc", "conflict_min_mttc",
        "conflict_max_dgt_pet",
    ]
    tl_cols = [c for c in tl_cols if c in tracks.columns]
    tlabels = tracks[tl_cols].copy()
    tlabels.to_csv(os.path.join(out, "track_labels.csv"), index=False)

    manifest = {
        "scene": SCENE,
        "source": "Figshare FLUID v2 derived_data (TI)",
        "fps": FPS,
        "videos": {f"20250529_video-{v}": {
            "split": s,
            "n_tracks": int((tracks.video_id == f"20250529_video-{v}").sum()),
            "n_segments": int((segments.video_id == f"20250529_video-{v}").sum()),
        } for v, s in zip(VIDEOS, ["train", "val", "test"])},
        "thresholds": {
            "stop_speed_median_ms": STOP_SPEED_MED,
            "stop_low_speed_ratio": STOP_LOW_RATIO,
            "turn_net_deg": TURN_NET_DEG,
            "turn_cum_deg": TURN_CUM_DEG,
            "jump_speed_ms": JUMP_SPEED_THR,
            "min_interp_run": MIN_INTERP_RUN,
            "min_seg_points": MIN_SEG_POINTS,
        },
        "window": {"frames": args.win, "stride": args.stride,
                   "seconds": args.win / FPS},
        "n_tracks_total": int(len(tracks)),
        "n_segments_total": int(len(segments)),
        "label_distribution": segments.label.value_counts().to_dict(),
        "label_distribution_by_split": {
            s: segments[segments.split == s].label.value_counts().to_dict()
            for s in ["train", "val", "test"]},
        "track_label_distribution": tracks.route_turn_label.value_counts(
            dropna=False).to_dict(),
        "track_class_distribution": tracks.track_class.value_counts().to_dict(),
        "track_class_distribution_by_split": {
            s: tracks[tracks.split == s].track_class.value_counts().to_dict()
            for s in ["train", "val", "test"]},
        "track_class_distribution_motor_only": tracks[
            tracks.is_motor].track_class.value_counts().to_dict(),
        "track_class_source_distribution": tracks.track_class_source.value_counts(
        ).to_dict(),
        "signal_source_distribution": segments.signal_source.value_counts(
        ).to_dict(),
        "signal_state_known_ratio": float(segments.signal_state_known.mean()),
        "signal_state_ref_known_ratio": float(
            segments.signal_state_ref.notna().mean()),
        "turn_map": TURN_MAP,
    }

    # ---- 交通冲突统计 ----
    if conflicts is not None:
        ev = conflicts[conflicts.matched]
        _pair = conflicts.apply(
            lambda r: tuple(sorted((r.track_id, r.peer_track_id))), axis=1)
        n_event = int(conflicts.assign(_pair=_pair).groupby(
            ["video_id", "frame", "_pair"]).ngroups)
        manifest["conflict"] = {
            "source": f"FLUID upstream {SCENE}_conflict.csv",
            "n_events": n_event,
            "n_vehicle_event_rows": int(len(conflicts)),
            "n_matched_rows": int(conflicts.matched.sum()),
            "n_unmatched_rows": int((~conflicts.matched).sum()),
            "n_tracks_involved": int(tracks.has_conflict.sum()),
            "n_tracks_involved_motor": int(
                tracks[tracks.is_motor].has_conflict.sum()),
            "conflict_type_distribution": {
                str(k): int(v) for k, v in
                conflicts[conflicts.role == "car1"].conflict_type
                .value_counts().items()},
            "conflict_events_by_split": {
                str(k): int(v) for k, v in
                conflicts[conflicts.role == "car1"].split
                .value_counts().items()},
            "severity": {
                "min_diff_m": float(ev["diff"].min()),
                "median_diff_m": float(ev["diff"].median()),
                "min_ttc_s": float(ev["TTC"].min()),
                "median_ttc_s": float(ev["TTC"].median()),
                "min_mttc_s": float(ev["MTTC"].min()),
                "max_dgt_pet": float(ev["DGT_PET"].max()),
            },
        }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n输出目录: {out}")


if __name__ == "__main__":
    sys.exit(main())
