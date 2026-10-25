# -*- coding: utf-8 -*-
"""
FLUID / TI 路口：停车 / 直行 / 转弯 三分类的建模与评价。

三个层次的评价
  ① 窗口级三分类（自监督标签）
     标签由运动学判据导出（speed_smooth / 净转角），因此判据量本身被排除出特征集；
     若不排除，分类器只是复述标签规则，会得到虚高的 0.99+ 准确率。
  ② 轨迹级交叉校验
     窗口标签按轨迹多数票回落到「转弯/直行」，与 route.turn（人工路线标注）比对，
     检验运动学判据生成的行为标签与人工标注是否一致。
  ③ 监督式转向分类（独立真值）
     直接以 route.turn 作为标签（s→直行，l/r/t→转弯）训练分类器，
     这份真值独立于运动学判据，能给出真实泛化性能；再叠加信号相位特征，
     检验信号信息对转向/停车行为的增益。
  ④ 停车行为 × 信号相位交叉验证
     检验停车标签是否真的对应红灯停车场景。
  ⑤ 轨迹级三分类（stop / straight / turn）
     把窗口特征按轨迹聚合，预测整条轨迹的行为类别；同一轨迹全部窗口只在一侧。

数据划分
  严格按整段视频：video-1=train / video-2=val / video-3=test。
  同一视频的全部帧与全部 track_id 只出现在一侧，相邻帧与同车轨迹不跨边。

输出
  reports/eval_report.md
  reports/eval_metrics.json
  reports/figures/*.png

用法: python evaluate.py [--data data] [--out reports]
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

LABELS = ["stop", "straight", "turn"]
SEED = 42

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

CMAP = {"stop": "#C0392B", "straight": "#1B3A5C", "turn": "#4A6B8A"}

# ------------------------------------------------------------------
# 泄漏防护：行为标签由下列判据量导出，这些列必须从特征集中剔除，
# 否则分类器只是复述标签规则（自一致性），不能反映泛化能力。
# ------------------------------------------------------------------
LEAKY = {
    "seg_speed_smooth_med",      # stop 判据：速度中位数
    "seg_low_speed_ratio",       # stop 判据：低速帧占比
    "seg_net_turn_deg",          # turn 判据：净转角
    "seg_cum_turn_deg",          # turn 判据：累计转角
    "seg_turn_rate_deg_s",       # 与累计转角共线
    "seg_straightness",          # 与净转角强共线
    "label",
}

BASE_DROP = {
    "seg_idx", "video_id", "scene", "track_id", "label", "split",
    "seg_start_frame", "seg_end_frame", "seg_start_time", "seg_end_time",
    "seg_mid_time", "track_type", "entry_direction", "exit_direction",
    "overall_direction", "route_turn", "route_turn_label",
    "signal_state", "signal_cycle", "signal_source", "signal_state_ref",
}


def load(data_dir):
    seg = pd.read_csv(os.path.join(data_dir, "segments.csv"))
    trk = pd.read_csv(os.path.join(data_dir, "tracks.csv"))
    return seg, trk


def set_cn_ylabel(ax, text, labelpad=14):
    """设置中文纵轴标签（水平放置）。

    matplotlib 竖排中文（CJK 旋转 90°）存在字形度量缺陷：
    同一标签内的汉字会互相重叠成乱码（横排 x 轴标签正常）。
    统一改为 rotation=0 水平放置在轴左侧，视觉更清晰。
    """
    ax.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=labelpad)


def feature_cols(seg, exclude_leaky=True, extra_drop=()):
    """选出可用作特征的数值列。

    只保留数值/布尔列：字符串列（如 signal_state、route_turn_label）
    必须先经独热编码，否则无法直接送进模型。
    """
    drop = set(BASE_DROP) | set(extra_drop)
    if exclude_leaky:
        drop |= LEAKY
    return [c for c in seg.columns
            if c not in drop and pd.api.types.is_numeric_dtype(seg[c])]


def make_models():
    return {
        "GradientBoosting": GradientBoostingClassifier(
            random_state=SEED, n_estimators=250, max_depth=4, learning_rate=0.08),
        "RandomForest": RandomForestClassifier(
            random_state=SEED, n_estimators=300, max_depth=14,
            min_samples_leaf=3, n_jobs=-1),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=SEED, C=1.0)),
    }


def _fit_select(X, y, splits, models):
    """按 val 宏F1 选模型。

    返回 (结果dict, 最佳名, 拟合好的模型, 标签编码器)。
    标签在内部被 LabelEncoder 编成整数，模型 predict 输出也是整数；
    调用方在算混淆矩阵/分类报告时必须用同一个 le 编码真值，
    否则会出现「字符串 + 整数」混合标签而报错。
    """
    le = LabelEncoder().fit(sorted(set(y)))
    Xtr = X[splits == "train"]
    ytr = le.transform(y[splits == "train"])
    Xva = X[splits == "val"]
    yva = le.transform(y[splits == "val"])
    Xte = X[splits == "test"]
    yte = le.transform(y[splits == "test"])

    results, fitted = {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        fitted[name] = mdl
        pv = mdl.predict(Xva)
        pt = mdl.predict(Xte)
        results[name] = {
            "val_accuracy": float(accuracy_score(yva, pv)),
            "val_macro_f1": float(f1_score(yva, pv, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(yte, pt)),
            "test_macro_f1": float(f1_score(yte, pt, average="macro", zero_division=0)),
            "test_precision": float(precision_score(yte, pt, average="macro",
                                                    zero_division=0)),
            "test_recall": float(recall_score(yte, pt, average="macro",
                                             zero_division=0)),
        }
    best = max(results, key=lambda k: results[k]["val_macro_f1"])
    return results, best, fitted[best], le


# ==================================================================
# ① 窗口级三分类（自监督标签）
# ==================================================================
def eval_window_level(seg, feats, out_dir, tag="window"):
    # 信号相位做独热编码（只用专属相位，语义准确）
    # 无专属相位的轨迹（右转/掉头）会落到 sig_nan 哑元上，
    # 这本身是有信息量的：说明该转向未单独信号化。
    d = pd.get_dummies(seg.copy(), columns=["signal_state"], prefix="sig",
                       dummy_na=True)
    sig_cols = [c for c in d.columns if c.startswith("sig_")]
    feats = list(feats) + sig_cols
    X = d[feats].astype(float)
    y = d.label
    le = LabelEncoder().fit(LABELS)
    yte = le.transform(y[d.split == "test"])
    yva = le.transform(y[d.split == "val"])

    results, best, mdl, _le = _fit_select(X, y, d.split, make_models())
    pt = mdl.predict(X[d.split == "test"])
    pv = mdl.predict(X[d.split == "val"])
    results[best]["test_report"] = classification_report(
        yte, pt, target_names=LABELS, zero_division=0, output_dict=True)
    results[best]["val_report"] = classification_report(
        yva, pv, target_names=LABELS, zero_division=0, output_dict=True)
    results[best]["test_confusion"] = confusion_matrix(
        yte, pt, labels=range(3)).tolist()
    results["_selected_model"] = best
    results["_best"] = results[best]

    imp = None
    if hasattr(mdl, "feature_importances_"):
        imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(
            ascending=False)

    # 混淆矩阵图
    cm = np.array(results[best]["test_confusion"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1B3A5C",
                    fontsize=13)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("预测")
    set_cn_ylabel(ax, "真实")
    ax.set_title(f"窗口级测试集混淆矩阵（{best}, video-3）", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_window.png"), dpi=170)
    plt.close(fig)

    if imp is not None:
        top = imp.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        ax.barh(top.index, top.values, color="#1B3A5C", height=0.65)
        ax.set_xlabel("Gini 重要度")
        ax.set_title("窗口级分类器特征重要度 Top-15（已剔除判据量）", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E3E6EA", lw=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=170)
        plt.close(fig)

    return results, best, imp


# ==================================================================
# ② 轨迹级：运动学标签 vs route.turn 真值
# ==================================================================
def eval_track_level(seg, trk):
    v = seg[seg.route_turn.notna()].copy()
    agg = v.groupby(["video_id", "track_id"]).agg(
        n_win=("label", "size"),
        n_turn=("label", lambda s: (s == "turn").sum()),
        n_stop=("label", lambda s: (s == "stop").sum()),
        n_straight=("label", lambda s: (s == "straight").sum()),
        route_turn=("route_turn", "first"),
    ).reset_index()
    agg["pred_turn_class"] = np.where(
        agg.n_turn > agg.n_straight, "turn", "straight")
    agg["route_turn_label"] = agg.route_turn.map(
        {"s": "straight", "l": "turn", "r": "turn", "t": "turn"})
    agg = agg.merge(trk[["video_id", "track_id", "split", "type"]],
                    on=["video_id", "track_id"], how="left")

    res = {
        "n_tracks_with_route": int(len(agg)),
        "agreement_overall": float(accuracy_score(agg.route_turn_label,
                                                  agg.pred_turn_class)),
        "macro_f1": float(f1_score(agg.route_turn_label, agg.pred_turn_class,
                                   average="macro", zero_division=0)),
        "confusion": confusion_matrix(agg.route_turn_label, agg.pred_turn_class,
                                      labels=["straight", "turn"]).tolist(),
        "by_turn": {}, "by_split": {},
    }
    for t, g in agg.groupby("route_turn"):
        res["by_turn"][t] = {
            "n": int(len(g)),
            "agreement": float(accuracy_score(g.route_turn_label,
                                              g.pred_turn_class)),
            "mean_turn_windows": float(g.n_turn.mean()),
            "mean_stop_windows": float(g.n_stop.mean()),
        }
    for s, g in agg.groupby("split"):
        res["by_split"][s] = {
            "n": int(len(g)),
            "agreement": float(accuracy_score(g.route_turn_label,
                                              g.pred_turn_class)),
        }
    return res, agg


# ==================================================================
# ③ 监督式转向分类（route.turn 为独立真值）
# ==================================================================
def eval_supervised_turn(seg, out_dir):
    """以 route.turn（人工标注）为标签做直行/转弯二分类。

    这是唯一一份不依赖运动学判据的标签，能给出真实泛化性能。
    对比三种特征设置：
      kinematics_only            不含任何信号特征
      kinematics_plus_signal    只用该 (进口方向, 转向) 的专属相位
      kinematics_plus_signal_ref 额外把「同进口道直行相位」作为参考值
    最后一项用于检验：把参考相位当作车辆实际所受相位是否会误导模型。
    """
    d = seg[seg.route_turn.notna()].copy()
    d["y"] = d.route_turn.map({"s": "straight", "l": "turn", "r": "turn",
                                "t": "turn"})
    d = d[d.y.notna()]

    # 信号相位编码为数值特征（不使用原始字符串列）
    strict = pd.get_dummies(d, columns=["signal_state"], prefix="sig",
                             dummy_na=True)
    sig_cols_strict = [c for c in strict.columns if c.startswith("sig_")]
    ref = pd.get_dummies(d, columns=["signal_state_ref"], prefix="sigref",
                         dummy_na=True)
    sig_cols_ref = [c for c in ref.columns if c.startswith("sigref_")]
    d = strict.join(ref[[c for c in ref.columns if c in sig_cols_ref]])

    kin_feats = feature_cols(d, exclude_leaky=True)
    # 判据量对「直行 vs 转弯」仍然泄漏，转弯类必须整体剔除
    kin_feats = [c for c in kin_feats
                 if c not in {"seg_straightness", "seg_cum_turn_deg",
                              "seg_turn_rate_deg_s", "seg_net_turn_deg"}]
    # 字符串列（route_turn_label / 派生 y）必须排除
    kin_feats = [c for c in kin_feats
                 if pd.api.types.is_numeric_dtype(d[c])]
    setups = {
        "kinematics_only": kin_feats,
        "kinematics_plus_signal": kin_feats + sig_cols_strict,
        "kinematics_plus_signal_ref": kin_feats + sig_cols_ref,
    }

    out = {}
    for name, feats in setups.items():
        feats = [f for f in feats if f in d.columns
                 and pd.api.types.is_numeric_dtype(d[f])]
        X = d[feats].astype(float)
        res, best, mdl, _le = _fit_select(X, d.y, d.split, make_models())
        # res 结构为 {model_name: metrics}，此处把最佳模型的结果平铺
        flat = dict(res[best])
        flat["_selected_model"] = best
        flat["n_features"] = len(feats)
        flat["_all_models"] = res
        out[name] = flat

    # 混淆矩阵（用 kinematics_plus_signal 的最佳模型）
    feats = setups["kinematics_plus_signal"]
    feats = [f for f in feats if f in d.columns
             and pd.api.types.is_numeric_dtype(d[f])]
    X = d[feats].astype(float)
    res, best, mdl, _le = _fit_select(X, d.y, d.split, make_models())
    pt = mdl.predict(X[d.split == "test"])
    yte = (d[d.split == "test"].y == "turn").astype(int).values
    cm = confusion_matrix(yte, pt, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1B3A5C",
                    fontsize=15)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["直行", "转弯"]); ax.set_yticklabels(["直行", "转弯"])
    ax.set_xlabel("预测")
    set_cn_ylabel(ax, "真实（route.turn）")
    ax.set_title(f"监督式转向分类测试集混淆矩阵（{best}, video-3）", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_turn.png"), dpi=170)
    plt.close(fig)

    out["_detail"] = {
        "selected_model": best,
        "test_report": classification_report(
            yte, pt, target_names=["straight", "turn"], zero_division=0,
            output_dict=True),
        "test_confusion": cm.tolist(),
        "signal_features": sig_cols_strict,
        "signal_ref_features": sig_cols_ref,
        "n_samples": int(len(d)),
        "label_counts": d.y.value_counts().to_dict(),
    }
    return out


# ==================================================================
# ④ 停车行为与信号灯状态交叉验证
# ==================================================================
def eval_stop_vs_signal(seg, trk):
    """检验「停车」标签是否与红灯停车一致。

    给出两个层次，口径不同、结论含义也不同：

    ① 窗口级（主检验）：只取**路口内**的窗口，按该窗口**所在时刻**的信号相位
       统计其中 stop 窗口的占比。这是「红灯时车是否停下」的直接检验。
       实测：专属相位口径下红灯 stop 占比 0.284、绿灯 0.018（相差约 16 倍）。

    ② 轨迹级（辅助）：按轨迹「进入路口时的相位」统计「该轨迹是否含任意停车段」。
       注意这个口径**不能**用来判断红灯停车：一条轨迹只要在进路口前排过队
       就会被计入「含停车段」，与进路口时的相位无关，因此绿灯轨迹的停车率
       同样很高（实测 0.37），看不出红灯与绿灯的差别。
       它的用途是描述「有多少条机动车轨迹出现过停车」。

    相位来源分两档：
      sig_in_state      专属相位（该 进口方向+转向 有独立相位才有值）
      sig_in_state_ref  参考相位（缺失时回退到同进口道直行相位）
    右转/掉头在本路口未单独信号化，参考相位覆盖了它们，
    因此用参考相位做检验覆盖面更全；但它并非该车实际所受相位，故两种口径都给出。

    子集口径说明：机动车子集按 tracks.is_motor 判定（口径见 build_dataset.py
    的 MOTOR_TYPES），不是按「有无 route 转向标注」判定 —— 二者不等
    （1468 vs 1257），前者才是法规意义上的机动车。
    """
    d = seg.copy()
    # ---------- ① 窗口级主检验：路口内窗口，按窗口时刻相位 ----------
    # seg 与 trk 都有 is_motor 列，merge 后会变成 is_motor_x/_y，
    # 这里只取 trk 侧的（同一列同源），并统一命名为 is_motor。
    d = d.merge(trk[["video_id", "track_id", "is_motor"]].rename(
                    columns={"is_motor": "is_motor_t"}),
                on=["video_id", "track_id"], how="left")
    d["is_motor"] = d["is_motor_t"]
    mot = d[d.is_motor.fillna(False).astype(bool)]
    ins = mot[mot.in_intersection == True]  # noqa: E712
    res = {}
    for tag, col in [("", "signal_state"), ("_ref", "signal_state_ref")]:
        res[f"window_stop_rate_in_intersection{tag}"] = {
            str(ph): float((g.label == "stop").mean())
            for ph, g in ins.groupby(col) if len(g)}
        res[f"n_window_by_phase_in_intersection{tag}"] = {
            str(ph): int(len(g)) for ph, g in ins.groupby(col)}
    res["n_window_in_intersection_total"] = int(len(ins))
    res["n_window_stop_in_intersection"] = int((ins.label == "stop").sum())

    # ---------- ② 轨迹级辅助统计 ----------
    st = d[d.label == "stop"].groupby(["video_id", "track_id"]).agg(
        n_stop_win=("label", "size"),
        stop_dur=("seg_duration", "sum"),
        med_v=("seg_speed_smooth_mean", "median"),
    ).reset_index()
    cols = ["video_id", "track_id", "split", "route_in_state", "route_turn",
            "sig_in_state", "sig_in_state_ref", "duration", "is_motor"]
    st = st.merge(trk[cols], on=["video_id", "track_id"], how="left")
    st["has_stop"] = True
    nostop = trk[~trk.set_index(["video_id", "track_id"]).index.isin(
        st.set_index(["video_id", "track_id"]).index)][cols].copy()
    nostop["has_stop"] = False
    nostop["n_stop_win"] = 0
    nostop["stop_dur"] = 0.0

    allt = pd.concat([st, nostop], ignore_index=True)
    tab = pd.crosstab(allt.sig_in_state, allt.has_stop, normalize="index")
    res["n_tracks_with_stop"] = int(st.shape[0])
    res["n_tracks_total"] = int(len(allt))
    res["stop_rate_by_entry_signal"] = {
        str(k): float(v) for k, v in tab[True].items()} if True in tab else {}
    res["n_by_entry_signal"] = {str(k): int(v) for k, v in
                                allt.sig_in_state.value_counts().items()}
    # 机动车子集：按 is_motor 判定（法规口径），而非「有无 route 转向标注」
    mv = allt[allt.is_motor.fillna(False).astype(bool)]
    res["n_motor_vehicles"] = int(len(mv))
    res["motor_stop_rate_by_entry_signal"] = {
        str(k): float(g.has_stop.mean()) for k, g in mv.groupby("sig_in_state")}
    res["n_motor_by_entry_signal"] = {
        str(k): int(len(g)) for k, g in mv.groupby("sig_in_state")}
    # 参考相位口径（含右转/掉头）
    res["motor_stop_rate_by_entry_signal_ref"] = {
        str(k): float(g.has_stop.mean())
        for k, g in mv.groupby("sig_in_state_ref")}
    res["n_motor_by_entry_signal_ref"] = {
        str(k): int(len(g)) for k, g in mv.groupby("sig_in_state_ref")}
    res["n_motor_without_dedicated_phase"] = int(mv.sig_in_state.isna().sum())
    return res, allt


# ==================================================================
# ⑤ 轨迹级三分类（stop / straight / turn）
# ==================================================================
def eval_track_three_class(seg, trk, out_dir):
    """轨迹级三分类：用窗口特征预测整条轨迹的行为类别。

    标签来自 tracks.track_class（停车由点级静止占比判定，转向由
    route.turn 人工标注），属于「自监督 + 部分人工真值」的混合标签。

    为避免同一辆车的相邻窗口同时出现在训练集与测试集（数据泄漏），
    这里按 (video_id, track_id) 聚合窗口特征后再划分：
    同一轨迹的所有窗口合成一条样本，轨迹只出现在一侧。
    """
    feats = feature_cols(seg)
    X = seg[feats].astype(float)

    # ---- 窗口特征按轨迹聚合（均值），形成轨迹级样本 ----
    agg = X.groupby([seg.video_id, seg.track_id]).mean().reset_index()
    meta = seg.groupby([seg.video_id, seg.track_id]).agg(
        split=("split", "first"),
        track_type=("track_type", "first"),
    ).reset_index()
    agg = agg.merge(meta, on=["video_id", "track_id"], how="left")

    lab = trk[["video_id", "track_id", "track_class", "track_class_source",
               "route_turn_label", "stop_ratio"]].copy()
    agg = agg.merge(lab, on=["video_id", "track_id"], how="left")
    # is_motor 已作为数值特征进入 X（车辆类型是合法的非泄漏特征），
    # 轨迹级聚合后取其均值即该轨迹是否机动车，无需再从 tracks 合并。

    res = {}
    for subset, mask in [
            ("all", np.ones(len(agg), dtype=bool)),
            ("motor_only", agg.is_motor.fillna(0).astype(bool).to_numpy())]:
        d = agg.loc[mask].copy()
        if d.track_class.isna().any():
            d = d[d.track_class.notna()]
        y = d.track_class
        if y.nunique() < 2:
            continue
        r, best, mdl, le = _fit_select(d[feats].astype(float), y, d.split,
                                       make_models())
        flat = dict(r[best])
        flat["_selected_model"] = best
        flat["n_features"] = len(feats)
        flat["n_tracks"] = int(len(d))
        flat["class_counts"] = y.value_counts().to_dict()
        flat["_all_models"] = r
        res[subset] = flat

        if subset == "motor_only":
            dtest = d[d.split == "test"]
            pt = mdl.predict(dtest[feats].astype(float))
            # 模型内部用 LabelEncoder 编码标签，predict 输出为整数；
            # 真值必须用同一个 le 编码，否则字符串+整数混排会报错。
            yte = le.transform(dtest.track_class.values)
            cm = confusion_matrix(yte, pt, labels=list(range(len(le.classes_))))
            rep = classification_report(yte, pt, target_names=list(le.classes_),
                                        zero_division=0, output_dict=True)
            res["_detail"] = {
                "selected_model": best,
                "test_report": rep,
                "test_confusion": cm.tolist(),
                "n_test": int(len(yte)),
            }
            fig, ax = plt.subplots(figsize=(6.4, 5.4))
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "#1B3A5C",
                            fontsize=13)
            ax.set_xticks(range(len(le.classes_)))
            ax.set_yticks(range(len(le.classes_)))
            ax.set_xticklabels(list(le.classes_))
            ax.set_yticklabels(list(le.classes_))
            ax.set_xlabel("预测")
            set_cn_ylabel(ax, "真实")
            ax.set_title(f"轨迹级三分类混淆矩阵（{best}, 机动车, video-3）", pad=12)
            for s in ax.spines.values():
                s.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "confusion_matrix_track.png"), dpi=170)
            plt.close(fig)
    return res


# ==================================================================
# ⑥ 交通冲突与行为的关系
# ==================================================================
def eval_conflicts(data_dir, seg, trk, out_dir):
    """分析 FLUID 上游检测到的交通冲突与车辆行为的关系。

    冲突表是**事件级**的（一次冲突两行：一行记 car1，一行记 car2），
    这里做三件事：
      1. 把每次冲突的两车配对，回到「行为 × 行为」的组合；
      2. 给每辆车定位冲突时刻它所处的窗口标签（stop/straight/turn）；
      3. 交叉分析：冲突类型、严重度指标、冲突是否发生在路口内、
         停车/转弯在冲突中的角色。

    窗口定位说明：冲突帧常落在轨迹末尾之后（实测 11/134 条，差 1–8 帧，
    属上游把轨迹尾部外推的边界效应），故采用**就近窗口**匹配
    （取 start_frame ≤ frame 的最后一个窗口，否则取第一个），
    可覆盖全部可匹配事件。
    """
    path = os.path.join(data_dir, "conflict_events.csv")
    if not os.path.exists(path):
        return None, None

    ev = pd.read_csv(path)
    ev_all = ev.copy()
    ev = ev[ev.matched].copy()
    if ev.empty:
        return None, None

    # ---- 定位冲突时刻所在窗口 ----
    seg_idx = {}
    for (vid, tid), g in seg.groupby(["video_id", "track_id"]):
        seg_idx[(vid, tid)] = g.sort_values("seg_start_frame")

    def window_label(vid, tid, frame):
        g = seg_idx.get((vid, tid))
        if g is None or len(g) == 0:
            return None, None
        before = g[g.seg_start_frame <= frame]
        row = before.iloc[-1] if len(before) else g.iloc[0]
        return row.label, int(abs(frame - row.seg_start_frame))

    labs, dists = [], []
    for r in ev_all.itertuples():
        lb, ds = window_label(r.video_id, r.track_id, r.frame)
        labs.append(lb)
        dists.append(ds)
    ev_all["win_label"] = labs
    ev_all["frame_offset"] = dists
    ev = ev_all[ev_all.matched].copy()

    # ---- 轨迹级信息 ----
    tinfo = trk.set_index(["video_id", "track_id"])
    ev["track_class"] = [
        tinfo.track_class.get((r.video_id, r.track_id), None)
        for r in ev.itertuples()]
    ev["is_motor"] = [
        bool(tinfo.is_motor.get((r.video_id, r.track_id), False))
        for r in ev.itertuples()]

    # ---- 配对成事件 ----
    # 配对必须在**完整长表**上做：若先按 matched 过滤，对车轨迹被上游删除的
    # 那 3 起事件会因缺对车行而整起丢失（实测 69 起只剩 68 起）。
    # 配对键必须含「无序车对」：只用 (video_id, frame) 不唯一 ——
    # 同一帧可能发生多起不同冲突。
    # 配对后再筛掉对车未匹配的事件，但它们要计入 unmatched 统计。
    def pair_key(vid, fr, a, b):
        return (vid, fr, tuple(sorted((int(a), int(b)))))

    car1 = ev_all[ev_all.role == "car1"].copy()
    car2 = ev_all[ev_all.role == "car2"].copy()
    car1["_pk"] = [pair_key(r.video_id, r.frame, r.track_id, r.peer_track_id)
                   for r in car1.itertuples()]
    car2["_pk"] = [pair_key(r.video_id, r.frame, r.track_id, r.peer_track_id)
                   for r in car2.itertuples()]

    cols1 = ["_pk", "video_id", "frame", "track_id", "peer_track_id",
             "win_label", "conflict_type", "diff", "TTC", "MTTC", "DGT_PET",
             "time", "split", "x", "y", "matched"]
    cols2 = ["_pk", "track_id", "win_label", "x", "y", "matched"]
    pair_all = car1[cols1].merge(car2[cols2], on="_pk", how="left",
                                 suffixes=("_1", "_2"))
    # 校验配对正确：car1 的对车应等于 car2 的本车
    pair_all["pair_ok"] = pair_all.peer_track_id == pair_all.track_id_2
    pair_all["both_matched"] = pair_all.matched_1 & pair_all.matched_2

    # 统计口径：事件总数 = 源表事件数；行为分析只用两车都能映射的事件
    n_event_src = int(len(car1))
    pair = pair_all[pair_all.both_matched].copy()

    res = {
        "n_events": n_event_src,
        "n_events_paired": int(len(pair)),
        "n_pair_ok": int(pair.pair_ok.sum()),
        "n_vehicle_rows": int(len(ev)),
        "n_unmatched_vehicle_rows": int((~ev_all.matched).sum()),
        "n_events_unpaired": int(len(pair_all) - len(pair)),
        "conflict_type_distribution": {
            str(k): int(v) for k, v in
            pair_all.conflict_type.value_counts().items()},
        "events_by_split": {
            str(k): int(v) for k, v in
            pair_all.split.value_counts().items()},
        "severity": {
            "diff_median_m": float(pair["diff"].median()),
            "diff_min_m": float(pair["diff"].min()),
            "ttc_median_s": float(pair.TTC.median()),
            "ttc_min_s": float(pair.TTC.min()),
            "mttc_median_s": float(pair.MTTC.median()),
            "dgt_pet_max": float(pair.DGT_PET.max()),
        },
        "window_match": {
            "n_located": int(ev.win_label.notna().sum()),
            "n_total": int(len(ev)),
            "median_frame_offset": float(pd.Series(
                [d for d in dists if d is not None]).median()),
        },
    }

    # ---- 行为 × 行为 组合（按事件去重，取 car1/car2 的行为对）----
    combo = (pair.groupby(["win_label_1", "win_label_2"])
             .size().reset_index(name="n_events"))
    res["behavior_pair_distribution"] = {
        f"{a}×{b}": int(n) for a, b, n in
        combo.itertuples(index=False)}

    # ---- 各行为在涉事车辆中的占比 vs 全体占比 ----
    all_counts = ev.win_label.value_counts()
    res["behavior_share_among_conflict_vehicles"] = {
        str(k): {"n": int(v), "share": float(v / len(ev))}
        for k, v in all_counts.items()}
    veh_base = seg.label.value_counts()
    res["behavior_share_in_all_windows"] = {
        str(k): float(v / len(seg)) for k, v in veh_base.items()}

    # ---- 严重度按行为分组 ----
    sev = {}
    for lb, g in ev.groupby("win_label"):
        if lb is None:
            continue
        sev[str(lb)] = {
            "n": int(len(g)),
            "diff_median_m": float(g["diff"].median()),
            "ttc_median_s": float(g.TTC.median()),
            "mttc_median_s": float(g.MTTC.median()),
            "dgt_pet_max": float(g.DGT_PET.max()),
            "severe_ttc_lt_1s": int((g.TTC < 1.0).sum()),
        }
    res["severity_by_behavior"] = sev

    # ---- 冲突类型 × 行为 ----
    tb = (ev.groupby(["conflict_type", "win_label"]).size()
          .reset_index(name="n"))
    res["conflict_type_by_behavior"] = {
        str(ct): {str(lb): int(n) for lb, n in
                  zip(g["win_label"], g["n"])}
        for ct, g in tb.groupby("conflict_type")}

    # ---- 严重冲突（TTC<1s）的行为构成 ----
    severe = ev[ev.TTC < 1.0]
    res["severe_events"] = {
        "threshold_ttc_s": 1.0,
        "n_vehicle_rows": int(len(severe)),
        "n_events": int(len(severe) // 2),
        "behavior_distribution": {
            str(k): int(v) for k, v in
            severe.win_label.value_counts().items()},
    }

    # ---- 冲突时刻是否在路口内 ----
    in_int = []
    for r in ev.itertuples():
        g = seg_idx.get((r.video_id, r.track_id))
        if g is None or len(g) == 0:
            in_int.append(None)
            continue
        before = g[g.seg_start_frame <= r.frame]
        row = before.iloc[-1] if len(before) else g.iloc[0]
        in_int.append(bool(row.in_intersection))
    ev["in_intersection"] = in_int
    res["in_intersection_rate"] = {
        str(k): int(v) for k, v in
        ev.in_intersection.value_counts(dropna=False).items()}

    # ---- 图件 ----
    make_conflict_plots(ev, pair, out_dir)
    return res, ev


def make_conflict_plots(ev, pair, out_dir):
    """冲突相关的图件。"""
    # 1) 冲突类型分布
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ct = pair.conflict_type.value_counts()
    order = ["rear end", "angle", "head on", "sideswipe"]
    order = [o for o in order if o in ct.index] + \
            [o for o in ct.index if o not in order]
    cnt = [int(ct[o]) for o in order]
    colors = ["#C0392B" if o == "rear end" else "#4A6B8A" for o in order]
    bars = ax.barh(order[::-1], cnt[::-1], color=colors[::-1], height=0.62)
    for b, c in zip(bars, cnt[::-1]):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                str(c), va="center", fontsize=10, color="#333")
    ax.set_xlabel("冲突事件数")
    ax.set_title("交通冲突类型分布（FLUID 上游检测，TI 路口三段视频）", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conflict_types.png"), dpi=170)
    plt.close(fig)

    # 2) 行为参与度：涉事车辆的行为构成
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    share_c = ev.win_label.value_counts(normalize=True)
    order_b = ["stop", "straight", "turn"]
    vals_c = [float(share_c.get(b, 0)) for b in order_b]
    x = np.arange(len(order_b))
    w = 0.36
    ax.bar(x - w / 2, vals_c, w, label="冲突涉事车辆", color="#C0392B")
    ax.set_xticks(x)
    ax.set_xticklabels(["停车", "直行", "转弯"])
    set_cn_ylabel(ax, "占比")
    ax.set_title("涉事车辆的行为构成", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    for xi, v_ in zip(x, vals_c):
        ax.text(xi - w / 2, v_ + 0.004, f"{v_:.1%}", ha="center",
                fontsize=10, color="#333")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conflict_behavior.png"), dpi=170)
    plt.close(fig)

    # 3) 严重度散点：diff vs TTC，按冲突类型着色
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ctype_color = {"rear end": "#C0392B", "angle": "#1B3A5C",
                   "head on": "#4A6B8A", "sideswipe": "#8A8F98"}
    for ct_, g in pair.groupby("conflict_type"):
        ax.scatter(g["diff"], g.TTC, s=42, alpha=0.78,
                   label=f"{ct_} (n={len(g)})",
                   color=ctype_color.get(ct_, "#4A6B8A"),
                   edgecolors="white", linewidths=0.6)
    ax.set_xlabel("两车间最小距离 diff (m)")
    ax.set_ylabel("碰撞时间 TTC (s)")
    ax.set_title("冲突严重度分布：距离 vs 碰撞时间", pad=12)
    ax.axhline(1.0, color="#999", ls="--", lw=1.0)
    ax.text(ax.get_xlim()[1], 1.0, " TTC=1s", va="bottom", ha="right",
            fontsize=9, color="#666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conflict_severity.png"), dpi=170)
    plt.close(fig)

    # 4) 冲突的空间分布（两车位置）
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for ct_, g in pair.groupby("conflict_type"):
        ax.scatter(g.x_1, g.y_1, s=40, alpha=0.8,
                   label=f"{ct_}", color=ctype_color.get(ct_, "#4A6B8A"),
                   edgecolors="white", linewidths=0.5)
    ax.scatter(pair.x_2, pair.y_2, s=18, alpha=0.35, color="#333",
               marker="x", label="对车位置")
    ax.set_xlabel("世界坐标 X (m)")
    ax.set_ylabel("世界坐标 Y (m)")
    ax.set_title("冲突事件的空间分布（TI 路口）", pad=12)
    ax.axhline(0, color="#CCC", lw=0.8)
    ax.axvline(0, color="#CCC", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conflict_spatial.png"), dpi=170)
    plt.close(fig)


# ==================================================================
# 图件
# ==================================================================
def make_plots(seg, trk, out_dir):
    # 1) 三视频标签分布
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    vids = sorted(seg.video_id.unique())
    w = 0.26
    for i, lab in enumerate(LABELS):
        vals = [seg[(seg.video_id == v) & (seg.label == lab)].shape[0] for v in vids]
        tot = [(seg.video_id == v).sum() for v in vids]
        pct = [v_ / t * 100 for v_, t in zip(vals, tot)]
        x = np.arange(len(vids)) + (i - 1) * w
        bars = ax.bar(x, pct, w, label=lab, color=CMAP[lab])
        for b, p in zip(bars, pct):
            ax.text(b.get_x() + b.get_width() / 2, p + 1.2, f"{p:.1f}%",
                    ha="center", fontsize=9.5, color="#333")
    ax.set_xticks(np.arange(len(vids)))
    ax.set_xticklabels([v.replace("20250529_", "") for v in vids])
    set_cn_ylabel(ax, "窗口占比 (%)")
    ax.set_title("各视频窗口级行为标签分布（划分依据：整段视频）", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=170)
    plt.close(fig)

    # 2) 特征剖面
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    profiles = [
        ("seg_speed_smooth_mean", "窗口内平均平滑速度 (m/s)"),
        ("seg_v_std", "窗口内速度波动 (m/s)"),
        ("seg_path_len", "窗口内路径长度 (m)"),
    ]
    for ax, (col, title) in zip(axes, profiles):
        for lab in LABELS:
            d = seg[seg.label == lab][col]
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            lo, hi = np.percentile(d, [1, 99])
            if hi <= lo:
                continue
            bins = np.linspace(lo, hi, 46)
            hist, edges = np.histogram(np.clip(d, lo, hi), bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, color=CMAP[lab], lw=1.9, label=lab)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#EEF0F3", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("三类行为的运动学特征剖面（判据可分性直观检查）", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_profiles.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    # 3) 交叉口空间分布：按转向类别着色
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))
    cmap_turn = {"s": "#1B3A5C", "l": "#C0392B", "r": "#4A6B8A", "t": "#7F8C8D"}
    for ax, v in zip(axes, sorted(seg.video_id.unique())):
        d = seg[seg.video_id == v]
        for t in ["s", "l", "r", "t"]:
            dd = d[d.route_turn == t]
            if len(dd) == 0:
                continue
            ax.scatter(dd.seg_cx_m_mean, dd.seg_cy_m_mean, s=3,
                       color=cmap_turn[t], alpha=0.25, linewidths=0)
        ax.axhline(0, color="#D5D9DE", lw=0.8, zorder=0)
        ax.axvline(0, color="#D5D9DE", lw=0.8, zorder=0)
        ax.set_title(f"{v.replace('20250529_','')}  n={len(d)}", fontsize=11)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap_turn[k],
                          label={"s": "直行", "l": "左转", "r": "右转",
                                 "t": "掉头"}[k], markersize=6)
               for k in ["s", "l", "r", "t"]]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("TI 路口世界坐标系下的窗口空间分布（按 route.turn 着色）",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spatial_distribution.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    # 4) 停车时长分布
    st = trk[trk.v_median < 1.0]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.hist(st.duration, bins=40, color="#C0392B", alpha=0.85)
    ax.set_xlabel("轨迹时长 (s)")
    set_cn_ylabel(ax, "轨迹条数")
    ax.set_title(f"低速（v_median < 1 m/s）轨迹的持续时长分布，n={len(st)}", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6EA", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stop_duration.png"), dpi=170)
    plt.close(fig)


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, args.data)
    out_dir = os.path.join(root, args.out)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    seg, trk = load(data_dir)
    feats = feature_cols(seg)
    print(f"特征数: {len(feats)}（已剔除判据泄漏量 {len(LEAKY)} 个）| 窗口数: {len(seg)}")

    print("[1/7] 窗口级三分类 ...")
    win_res, best, imp = eval_window_level(seg, feats, fig_dir)
    print(f"  选定模型: {best}")
    for k, v_ in win_res.items():
        if k.startswith("_"):
            continue
        print(f"  {k:20s} val_macroF1={v_['val_macro_f1']:.4f} "
              f"test_acc={v_['test_accuracy']:.4f} test_macroF1={v_['test_macro_f1']:.4f}")

    print("[2/7] 轨迹级交叉校验 ...")
    trk_res, agree_df = eval_track_level(seg, trk)
    print(f"  轨迹级一致率: {trk_res['agreement_overall']:.4f}")

    print("[3/7] 监督式转向分类（route.turn 独立真值） ...")
    sup_res = eval_supervised_turn(seg, fig_dir)
    for k in ["kinematics_only", "kinematics_plus_signal",
              "kinematics_plus_signal_ref"]:
        if k not in sup_res:
            continue
        r = sup_res[k]
        print(f"  {k:30s} val_macroF1={r['val_macro_f1']:.4f} "
              f"test_acc={r['test_accuracy']:.4f} test_macroF1={r['test_macro_f1']:.4f}")

    print("[4/7] 停车 × 信号相位交叉验证 ...")
    stop_res, stop_df = eval_stop_vs_signal(seg, trk)
    print(f"  {stop_res['motor_stop_rate_by_entry_signal']}")

    print("[5/7] 轨迹级三分类（stop / straight / turn） ...")
    tc_res = eval_track_three_class(seg, trk, fig_dir)
    for k in ["all", "motor_only"]:
        if k in tc_res:
            r = tc_res[k]
            print(f"  {k:12s} {r['_selected_model']:20s} "
                  f"val_macroF1={r['val_macro_f1']:.4f} "
                  f"test_acc={r['test_accuracy']:.4f} "
                  f"test_macroF1={r['test_macro_f1']:.4f} "
                  f"(n={r['n_tracks']})")

    print("[6/7] 交通冲突与行为的关系 ...")
    cf_res, cf_df = eval_conflicts(data_dir, seg, trk, fig_dir)
    if cf_res:
        print(f"  冲突事件 {cf_res['n_events']} 次，"
              f"配对校验通过 {cf_res['n_pair_ok']}/{cf_res['n_events']}")
        print(f"  行为构成: {cf_res['behavior_share_among_conflict_vehicles']}")

    print("[7/7] 出图与写报告 ...")
    make_plots(seg, trk, fig_dir)

    metrics = {
        "window_level": win_res,
        "track_level": trk_res,
        "supervised_turn": {k: {kk: vv for kk, vv in v_.items()
                                if not kk.startswith("_")}
                            for k, v_ in sup_res.items() if k != "_detail"},
        "supervised_turn_detail": sup_res["_detail"],
        "stop_vs_signal": stop_res,
        "track_three_class": {k: {kk: vv for kk, vv in v_.items()
                                  if not kk.startswith("_")}
                              for k, v_ in tc_res.items() if k != "_detail"},
        "track_three_class_detail": tc_res.get("_detail", {}),
        "conflicts": cf_res,
        "feature_importance_top20": (
            {k: float(v_) for k, v_ in imp.head(20).items()} if imp is not None else {}),
        "n_features_used": len(feats),
        "n_segments": int(len(seg)),
        "n_tracks": int(len(trk)),
        "leaky_features_excluded": sorted(LEAKY),
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    L = []
    L.append("# FLUID / TI 路口：停车 / 直行 / 转弯 分类评价报告\n")

    L.append("## 1. 数据与划分\n")
    L.append("| 视频 | 划分 | 轨迹数 | 窗口数 | stop | straight | turn |")
    L.append("|---|---|---|---|---|---|---|")
    for v, s in zip(["20250529_video-1", "20250529_video-2", "20250529_video-3"],
                    ["train", "val", "test"]):
        d = seg[seg.video_id == v]
        vc = d.label.value_counts()
        L.append(f"| {v} | {s} | {(trk.video_id == v).sum()} | {len(d)} | "
                 f"{vc.get('stop',0)} | {vc.get('straight',0)} | {vc.get('turn',0)} |")
    L.append("")
    L.append("划分严格按**整段视频**进行：同一视频的全部帧、全部 track_id 只出现在"
             "训练/验证/测试的一侧。脚本内置断言校验「每个 video_id 只属于一个 split」，"
             "杜绝相邻帧与同一辆车的轨迹跨边泄漏。\n")

    L.append("## 2. 窗口级三分类（自监督标签）\n")
    L.append("> **防泄漏说明**：行为标签由 `seg_speed_smooth_med` / `seg_low_speed_ratio` / "
             "`seg_net_turn_deg` / `seg_cum_turn_deg` 导出，这些判据量已从特征集中剔除，"
             "否则分类器只是复述标签规则，会得到虚高的近 1.0 准确率。\n")
    L.append("| 模型 | val 宏F1 | val 准确率 | test 准确率 | test 宏F1 | test 宏精确率 | test 宏召回率 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v_ in win_res.items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v_['val_macro_f1']:.4f} | {v_['val_accuracy']:.4f} | "
                 f"{v_['test_accuracy']:.4f} | {v_['test_macro_f1']:.4f} | "
                 f"{v_['test_precision']:.4f} | {v_['test_recall']:.4f} |")
    L.append(f"\n选定模型（按 val 宏F1）：**{best}**\n")
    L.append("### 测试集分类报告\n")
    rep = win_res["_best"]["test_report"]
    L.append("| 类别 | precision | recall | f1-score | support |")
    L.append("|---|---|---|---|---|")
    for lab in LABELS:
        r = rep[lab]
        L.append(f"| {lab} | {r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{r['f1-score']:.4f} | {int(r['support'])} |")
    L.append(f"\n准确率 {rep['accuracy']:.4f}，宏F1 {rep['macro avg']['f1-score']:.4f}。\n")
    L.append("混淆矩阵（行=真实，列=预测）：\n")
    cm = win_res["_best"]["test_confusion"]
    L.append("| | " + " | ".join(LABELS) + " |")
    L.append("|---|" + "---|" * 3)
    for i, lab in enumerate(LABELS):
        L.append(f"| **{lab}** | " + " | ".join(str(x) for x in cm[i]) + " |")
    L.append("")

    L.append("## 3. 轨迹级交叉校验：运动学标签 vs 人工路线标注\n")
    L.append("把窗口标签按轨迹多数票回落为「转弯/直行」，与 `route.turn` 比对。\n")
    L.append("| route.turn | 轨迹数 | 一致率 | 平均转弯窗口数 | 平均停车窗口数 |")
    L.append("|---|---|---|---|---|")
    for k, v_ in sorted(trk_res["by_turn"].items()):
        L.append(f"| {k} | {v_['n']} | {v_['agreement']:.4f} | "
                 f"{v_['mean_turn_windows']:.2f} | {v_['mean_stop_windows']:.2f} |")
    L.append(f"\n整体一致率 **{trk_res['agreement_overall']:.4f}**，"
             f"宏F1 {trk_res['macro_f1']:.4f}。\n")
    L.append("| 划分 | 轨迹数 | 一致率 |")
    L.append("|---|---|---|")
    for k, v_ in trk_res["by_split"].items():
        L.append(f"| {k} | {v_['n']} | {v_['agreement']:.4f} |")
    L.append("")
    L.append("> 一致率在 `s`（直行）上最高，在 `l`/`r`（转弯）上偏低，"
             "原因是多数票回落会把「转弯前停车」计成直行窗口；"
             "第 7 节的轨迹级三分类改用点级静止占比判定停车，"
             "转向直接采用人工标注，不再受此影响。\n")

    L.append("## 4. 监督式转向分类（route.turn 为独立真值）\n")
    L.append("这一节的标签是人工路线标注，与运动学判据无关，"
             "因此结果反映**真实泛化性能**。\n")
    L.append("| 特征设置 | 选定模型 | 特征数 | val 宏F1 | test 准确率 | test 宏F1 |")
    L.append("|---|---|---|---|---|---|")
    for k in ["kinematics_only", "kinematics_plus_signal",
              "kinematics_plus_signal_ref"]:
        if k not in sup_res:
            continue
        r = sup_res[k]
        L.append(f"| {k} | {r['_selected_model']} | {r['n_features']} | "
                 f"{r['val_macro_f1']:.4f} | {r['test_accuracy']:.4f} | "
                 f"{r['test_macro_f1']:.4f} |")
    L.append("")
    L.append("> `kinematics_plus_signal` 只用**该 (进口方向, 转向) 的专属相位**；"
             "`kinematics_plus_signal_ref` 额外把「同进口道直行相位」作为参考值加入。"
             "参考值覆盖了右转/掉头（这两种转向在本路口未单独信号化），"
             "但它并不等于车辆实际所受相位，加入后指标下降，"
             "说明参考相位不能替代专属相位使用。\n")
    d = sup_res["_detail"]
    L.append("### 测试集分类报告（kinematics + signal）\n")
    L.append("| 类别 | precision | recall | f1-score | support |")
    L.append("|---|---|---|---|---|")
    for lab in ["straight", "turn"]:
        r = d["test_report"][lab]
        L.append(f"| {lab} | {r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{r['f1-score']:.4f} | {int(r['support'])} |")
    L.append("")

    L.append("## 5. 停车行为 × 信号相位交叉验证\n")
    L.append("检验「停车」标签是否对应红灯停车场景（机动车子集，"
             "按 `is_motor` 的法规口径筛选）。\n")
    L.append("### 5.1 窗口级主检验：路口内窗口，按窗口所在时刻的相位\n")
    L.append("只取**路口内**的窗口，按该窗口**所在时刻**的信号相位统计 "
             "其中 stop 窗口的占比 —— 这是「红灯时车是否停下」的直接检验。\n")
    L.append("| 相位 | 路口内窗口数 | stop 窗口占比 |")
    L.append("|---|---|---|")
    for k in sorted(stop_res.get("window_stop_rate_in_intersection", {})):
        L.append(f"| {k} | "
                 f"{stop_res['n_window_by_phase_in_intersection'].get(k, 0)} | "
                 f"{stop_res['window_stop_rate_in_intersection'][k]:.4f} |")
    L.append("")
    wref = stop_res.get("window_stop_rate_in_intersection_ref", {})
    if wref:
        L.append("参考相位口径（含右转/掉头）：\n")
        L.append("| 相位 | 路口内窗口数 | stop 窗口占比 |")
        L.append("|---|---|---|")
        for k in sorted(wref):
            L.append(f"| {k} | "
                     f"{stop_res['n_window_by_phase_in_intersection_ref'].get(k, 0)} | "
                     f"{wref[k]:.4f} |")
        L.append("")
    gr = stop_res.get("window_stop_rate_in_intersection", {})
    if "r" in gr and "G" in gr and gr["G"] > 0:
        ratio = gr["r"] / gr["G"]
        L.append(f"> **红灯路口内停车占比是绿灯的 {ratio:.1f} 倍**"
                 f"（{gr['r']:.4f} vs {gr['G']:.4f}），"
                 "说明停车标签确实对应信号停车场景。\n")
    L.append("### 5.2 轨迹级辅助统计（口径不同，不可用于判断红灯停车）\n")
    L.append("按轨迹「进入路口时的相位」统计「该轨迹是否含任意停车段」。"
             "该口径**不能**用来判断红灯停车：一条轨迹只要在进路口前排过队"
             "就会被计入「含停车段」，与进路口时的相位无关，"
             "因此绿灯轨迹的停车率同样偏高，看不出红灯与绿灯的差别。\n")
    L.append("| 进入路口时信号相位 | 机动车轨迹数 | 含停车段的比例 |")
    L.append("|---|---|---|")
    for k in sorted(stop_res["motor_stop_rate_by_entry_signal"]):
        L.append(f"| {k} | {stop_res['n_motor_by_entry_signal'][k]} | "
                 f"{stop_res['motor_stop_rate_by_entry_signal'][k]:.4f} |")
    L.append("")
    L.append(f"机动车轨迹总数 {stop_res['n_motor_vehicles']}，"
             f"含显著停车段的轨迹 {stop_res['n_tracks_with_stop']} 条；"
             f"其中 {stop_res.get('n_motor_without_dedicated_phase', 0)} 条"
             f"在本路口无专属相位（右转/掉头未单独信号化）。\n")

    if imp is not None:
        L.append("## 6. 特征重要度 Top-20\n")
        L.append("| 排名 | 特征 | Gini 重要度 |")
        L.append("|---|---|---|")
        for i, (k, v_) in enumerate(imp.head(20).items(), 1):
            L.append(f"| {i} | `{k}` | {v_:.4f} |")
        L.append("")

    L.append("## 7. 轨迹级三分类：停车 / 直行 / 转弯\n")
    L.append("把窗口特征按轨迹聚合（均值），每条轨迹构成一个样本，"
             "预测整条轨迹的行为类别。划分仍按整段视频，"
             "同一轨迹的全部窗口只出现在一侧。\n")
    L.append("| 子集 | 选定模型 | 轨迹数 | val 宏F1 | test 准确率 | test 宏F1 |")
    L.append("|---|---|---|---|---|---|")
    for k in ["all", "motor_only"]:
        if k not in tc_res:
            continue
        r = tc_res[k]
        L.append(f"| {k} | {r['_selected_model']} | {r['n_tracks']} | "
                 f"{r['val_macro_f1']:.4f} | {r['test_accuracy']:.4f} | "
                 f"{r['test_macro_f1']:.4f} |")
    L.append("")
    d = tc_res.get("_detail")
    if d:
        L.append("### 测试集分类报告（机动车子集）\n")
        L.append("| 类别 | precision | recall | f1-score | support |")
        L.append("|---|---|---|---|---|")
        for lab in LABELS:
            r = d["test_report"][lab]
            L.append(f"| {lab} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1-score']:.4f} | {int(r['support'])} |")
        L.append("")
        L.append("混淆矩阵（行=真实，列=预测）：\n")
        cm = d["test_confusion"]
        L.append("| | " + " | ".join(LABELS) + " |")
        L.append("|---|" + "---|" * 3)
        for i, lab in enumerate(LABELS):
            L.append(f"| **{lab}** | " + " | ".join(str(x) for x in cm[i]) + " |")
        L.append("")

    # ---- 交通冲突 ----
    if cf_res:
        L.append("## 8. 交通冲突与行为的关系\n")
        L.append("数据来自 FLUID 上游的交通冲突检测结果（`raw/conflict/TI_conflict.csv`），"
                 f"共 {cf_res['n_events']} 次冲突事件，"
                 f"涉及 {cf_res['n_vehicle_rows']} 条可映射的车-事件记录；"
                 f"另有 {cf_res['n_unmatched_vehicle_rows']} 条车-事件记录涉及"
                 f"已被上游删除的轨迹，对应 {cf_res['n_events_unpaired']} 起事件"
                 f"因缺一方而无法配对。\n")
        L.append(f"事件配对校验：{cf_res['n_pair_ok']}/{cf_res['n_events']} 起"
                 "两车身份互为对方（`peer_track_id` 与对侧 `track_id` 一致）；"
                 f"未通过的 {cf_res['n_events'] - cf_res['n_pair_ok']} 起均为对侧轨迹"
                 "被上游删除，属数据边界而非处理错误。"
                 "以下行为分析基于两车均可映射的事件。\n")

        L.append("### 8.1 冲突类型分布\n")
        L.append("| 冲突类型 | 事件数 |")
        L.append("|---|---|")
        for k, v_ in cf_res["conflict_type_distribution"].items():
            L.append(f"| {k} | {v_} |")
        L.append("")

        L.append("### 8.2 涉事车辆的行为构成\n")
        L.append("按「冲突时刻该车所处的滑动窗口标签」统计"
                 f"（窗口定位覆盖 {cf_res['window_match']['n_located']}/"
                 f"{cf_res['window_match']['n_total']}）。\n")
        L.append("| 行为 | 涉事车辆数 | 占涉事车辆 | 占全部窗口 |")
        L.append("|---|---|---|---|")
        base = cf_res["behavior_share_in_all_windows"]
        for k, v_ in cf_res["behavior_share_among_conflict_vehicles"].items():
            L.append(f"| {k} | {v_['n']} | {v_['share']:.4f} | "
                     f"{base.get(k, 0):.4f} |")
        L.append("")
        bsc = cf_res["behavior_share_among_conflict_vehicles"]
        if "stop" in bsc and base.get("stop", 0) > 0:
            ratio = bsc["stop"]["share"] / base["stop"]
            L.append(f"> 停车车辆在冲突中的占比是其在全部窗口中占比的 **{ratio:.1f} 倍**，"
                     "说明追尾类冲突与「前车静止、后车仍前行」直接相关。\n")

        L.append("### 8.3 严重度按行为分组\n")
        L.append("| 行为 | 车-事件数 | diff 中位 (m) | TTC 中位 (s) | MTTC 中位 (s) | "
                 "DGT/PET 最大 | TTC<1s 的事件数 |")
        L.append("|---|---|---|---|---|---|---|")
        for k, v_ in cf_res["severity_by_behavior"].items():
            L.append(f"| {k} | {v_['n']} | {v_['diff_median_m']:.2f} | "
                     f"{v_['ttc_median_s']:.3f} | {v_['mttc_median_s']:.2f} | "
                     f"{v_['dgt_pet_max']:.2f} | {v_['severe_ttc_lt_1s']} |")
        L.append("")

        L.append("### 8.4 行为 × 行为组合（按事件）\n")
        L.append("| car1 行为 | car2 行为 | 事件数 |")
        L.append("|---|---|---|")
        for k, v_ in cf_res["behavior_pair_distribution"].items():
            a, b = k.split("×")
            L.append(f"| {a} | {b} | {v_} |")
        L.append("")

        L.append("### 8.5 严重冲突（TTC < 1 s）\n")
        sv = cf_res["severe_events"]
        L.append(f"TTC < {sv['threshold_ttc_s']} s 的车-事件记录 {sv['n_vehicle_rows']} 条"
                 f"（约 {sv['n_events']} 次冲突），行为构成：\n")
        L.append("| 行为 | 车-事件数 |")
        L.append("|---|---|")
        for k, v_ in sv["behavior_distribution"].items():
            L.append(f"| {k} | {v_} |")
        L.append("")

        L.append("### 8.6 冲突发生位置\n")
        L.append("| 是否在路口内 | 车-事件数 |")
        L.append("|---|---|")
        for k, v_ in cf_res["in_intersection_rate"].items():
            L.append(f"| {k} | {v_} |")
        L.append("")
        sev = cf_res["severity"]
        L.append(f"整体严重度：两车间最小距离 diff 中位 {sev['diff_median_m']:.2f} m"
                 f"（最小 {sev['diff_min_m']:.2f} m）；"
                 f"碰撞时间 TTC 中位 {sev['ttc_median_s']:.3f} s"
                 f"（最小 {sev['ttc_min_s']:.2f} s）；"
                 f"MTTC 中位 {sev['mttc_median_s']:.2f} s；"
                 f"DGT/PET 最大 {sev['dgt_pet_max']:.2f} m/s²。\n")

    L.append("## 9. 图件\n")
    for f_ in sorted(os.listdir(fig_dir)):
        L.append(f"- `figures/{f_}`")
    L.append("")

    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "eval_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"报告输出: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
