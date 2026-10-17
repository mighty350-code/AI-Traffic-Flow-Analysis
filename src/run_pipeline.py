#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键流水线编排器：视频 → 轨迹 → 安全指标 → 前端可视化

把原本靠手写命令串起来的各环节固化成一份配置（pipeline_config.yaml），
做到"一条命令复现全流程"。不重写任何算法，只做衔接、日志与断点。

常用用法（都在项目根目录执行）：

    # 查看所有阶段
    python src/run_pipeline.py --list

    # 只打印将要执行的命令，不真跑（验证配置与路径）
    python src/run_pipeline.py --dry-run

    # 全量跑（含检测，耗时最长）
    python src/run_pipeline.py

    # 已有 input CSV，从清洗阶段接着跑
    python src/run_pipeline.py --from clean

    # 只跑安全指标这一段
    python src/run_pipeline.py --from ssm --to conflict

日志落在 logs/pipeline_<时间戳>.log；final/ 下的结果在写入前自动备份到 backups/<时间戳>/。
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time

# GitHub 发布版把主代码放在 src/；第三方 FLUID 由使用者另行下载。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(ROOT, "third_party")
DP = os.path.join(TP, "FLUID", "demo", "data_process")
WEB = os.path.join(TP, "FLUID", "web")
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.yaml")
LOG_DIR = os.path.join(ROOT, "logs")
BACKUP_DIR = os.path.join(ROOT, "backups")

# 配置读取失败时的兜底（保证脚本永远能跑起来）
FALLBACK_STAGES = [
    ("detect", "检测与跟踪", "src/video_to_txt.py", "{root}"),
    ("structure", "给 HBB/OBB TXT 添加表头", "third_party/FLUID/demo/data_process/1_Structured_Recognition_Results.py", "{dp}"),
    ("holography", "平滑+单应变换", "third_party/FLUID/demo/data_process/2_holography_smooth_kinematic_4K_batch.py", "{dp}"),
    ("clean", "多源清洗", "third_party/FLUID/demo/data_process/4_Multi-source_Data_Integration_and_Cleaning.py", "{dp}"),
    ("motion", "运动校正与行为标注", "third_party/FLUID/demo/data_process/5_Motion_Correction_and_Behavior_Labeling.py", "{dp}"),
    ("kinematic", "卡尔曼+RTS", "third_party/FLUID/demo/data_process/6_Kinematic_Parameter_Processing.py", "{dp}"),
    ("ssm", "TTC/DRAC/MTTC", "src/2_droneTwoDimSSM.py", "{dp}"),
    ("pet", "PET/DGT", "src/2_droneDGTBoundary.py", "{dp}"),
    ("conflict", "冲突统计", "src/4_Conflict_Full_Statistics.py", "{dp}"),
    ("convert", "转前端数据", "third_party/FLUID/web/tools/convert_data.py", "{web}"),
]


def load_config(path):
    """读 YAML 配置；没有 PyYAML 或文件缺失时退回内置默认。"""
    if os.path.exists(path):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["stages"] = cfg.get("stages") or []
            return cfg
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读取配置失败（{e}），改用内置默认阶段表")
    cfg = {"source": "", "name": "DJI_0475", "device": "cpu", "imgsz": 1280,
           "conf": 0.3, "gmc": "orb", "obb_model": "", "max_frames": 0,
           "port": 8123, "serve": False}
    cfg["stages"] = [{"id": i, "desc": d, "script": s, "cwd": c, "enabled": True, "args": []}
                     for i, d, s, c in FALLBACK_STAGES]
    return cfg


def context(cfg):
    return {
        "root": ROOT, "dp": DP, "web": WEB,
        "source": str(cfg.get("source", "")),
        "name": str(cfg.get("name", "DJI_0475")),
        "device": str(cfg.get("device", "cpu")),
        "imgsz": str(cfg.get("imgsz", 1280)),
        "conf": str(cfg.get("conf", 0.3)),
        "gmc": str(cfg.get("gmc", "orb")),
        "obb_model": str(cfg.get("obb_model", "")),
        "max_frames": str(cfg.get("max_frames", 0)),
        "port": str(cfg.get("port", 8123)),
    }


def expand(text, ctx):
    out = str(text)
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", v)
    return out


def child_env():
    """把缓存与配置锁在项目目录，延续"不写 C 盘"的工程约定。"""
    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = os.path.join(ROOT, ".ultralytics")
    env["ULTRALYTICS_CONFIG_DIR"] = os.path.join(ROOT, ".ultralytics")
    env["TORCH_HOME"] = os.path.join(ROOT, ".cache", "torch")
    env["HF_HOME"] = os.path.join(ROOT, ".cache", "hf")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def backup_final(tag):
    """写阶段会覆写 final/，先把现有结果另存一份。"""
    final = os.path.join(DP, "final")
    if not os.path.isdir(final):
        return
    dst = os.path.join(BACKUP_DIR, tag)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for fn in os.listdir(final):
        if fn.lower().endswith(".csv"):
            shutil.copy2(os.path.join(final, fn), os.path.join(dst, fn))
            n += 1
    if n:
        print(f"[backup] final/ 下 {n} 个 CSV 已备份到 {dst}")


def resolve_script(script, cwd):
    """脚本路径可能是相对 cwd 的，也可能是相对项目根的，两处都找一遍。"""
    if os.path.isabs(script):
        return script
    p1 = os.path.join(cwd, script)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(ROOT, script)
    if os.path.exists(p2):
        return p2
    return p1


def run_stage(st, ctx, logf, dry=False):
    sid = st.get("id", "?")
    desc = st.get("desc", "")
    cwd = expand(st.get("cwd", "{root}"), ctx)
    script = resolve_script(expand(st.get("script", ""), ctx), cwd)
    args = [expand(a, ctx) for a in st.get("args", []) or []]
    cmd = [sys.executable, script] + args

    print(f"\n[{sid}] {desc}")
    print("   cwd : " + cwd)
    print("   $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    logf.write(f"\n[{sid}] {desc}\n$ {' '.join(cmd)}\n")
    logf.flush()
    if dry:
        return True

    if not os.path.exists(script):
        print(f"   [error] 脚本不存在：{script}")
        logf.write(f"[error] script missing: {script}\n")
        return False

    if sid in ("clean", "motion", "kinematic"):
        backup_final(time.strftime("%Y%m%d_%H%M%S"))

    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=child_env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        print("   " + line.rstrip())
        logf.write(line)
    rc = p.wait()
    dt = time.time() - t0
    print(f"   [{'ok' if rc == 0 else 'FAIL'}] rc={rc}  用时 {dt:.1f}s")
    logf.write(f"[{sid}] rc={rc} elapsed={dt:.1f}s\n")
    logf.flush()
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description="一键流水线：视频 → 轨迹 → 安全指标 → 前端")
    ap.add_argument("--config", default=CONFIG, help="配置文件路径")
    ap.add_argument("--source", default=None, help="覆盖配置里的输入视频")
    ap.add_argument("--device", default=None, help="覆盖 device（cpu / 0 / cuda:0）")
    ap.add_argument("--max-frames", type=int, default=None, help="覆盖最大帧数（小样本验证用）")
    ap.add_argument("--name", default=None, help="覆盖数据集名")
    ap.add_argument("--from", dest="from_stage", default=None, help="从哪个阶段开始")
    ap.add_argument("--to", dest="to_stage", default=None, help="到哪个阶段结束")
    ap.add_argument("--only", default=None, help="只跑指定阶段")
    ap.add_argument("--list", action="store_true", help="列出所有阶段后退出")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--serve", action="store_true", help="跑完后起前端服务")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["source"] = args.source
    if args.device:
        cfg["device"] = args.device
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.name:
        cfg["name"] = args.name
    ctx = context(cfg)

    stages = [s for s in cfg["stages"] if s.get("enabled", True)]
    ids = [s.get("id") for s in stages]

    if args.list:
        print(f"{'阶段':<12}{'状态':<8}说明")
        print("-" * 72)
        for s in cfg["stages"]:
            flag = "启用" if s.get("enabled", True) else "跳过"
            print(f"{s.get('id',''):<12}{flag:<8}{s.get('desc','')}")
        print(f"\n可执行顺序： {' → '.join(ids)}")
        return

    if args.only:
        stages = [s for s in stages if s.get("id") == args.only]
        if not stages:
            sys.exit(f"[error] 没有阶段 {args.only}；可用：{', '.join(ids)}")
    else:
        if args.from_stage:
            if args.from_stage not in ids:
                sys.exit(f"[error] --from {args.from_stage} 不存在；可用：{', '.join(ids)}")
            stages = stages[ids.index(args.from_stage):]
        if args.to_stage:
            sub = [s.get("id") for s in stages]
            if args.to_stage not in sub:
                sys.exit(f"[error] --to {args.to_stage} 不在剩余阶段中；可用：{', '.join(sub)}")
            stages = stages[:sub.index(args.to_stage) + 1]

    os.makedirs(LOG_DIR, exist_ok=True)
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logp = os.path.join(LOG_DIR, f"pipeline_{tag}.log")

    print("=" * 72)
    print(f"  流水线启动  {tag}")
    print(f"  输入视频：{ctx['source'] or '(未指定)'}")
    print(f"  执行阶段：{' → '.join(s.get('id') for s in stages)}")
    print(f"  日志：{logp}")
    print("=" * 72)

    with open(logp, "w", encoding="utf-8") as logf:
        t0 = time.time()
        for st in stages:
            ok = run_stage(st, ctx, logf, dry=args.dry_run)
            if not ok:
                print(f"\n[error] 阶段 {st.get('id')} 失败，流水线中止。日志：{logp}")
                sys.exit(1)
        print(f"\n[done] 全部阶段完成，总用时 {time.time() - t0:.1f}s")

    if args.serve or cfg.get("serve"):
        url = f"http://localhost:{ctx['port']}/"
        print(f"\n[serve] 启动前端：{url}  (Ctrl+C 退出)")
        subprocess.run([sys.executable, "-m", "http.server", ctx["port"]], cwd=WEB)


if __name__ == "__main__":
    main()
