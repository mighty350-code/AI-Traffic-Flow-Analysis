#!/usr/bin/env bash
# FLUID / TI 路口：全链路复现
#
# 构建统一训练数据 → 数据质量体检 → 建模与分类评价 → 数据契约校验
#
# 用法:
#   bash scripts/run_all.sh                 # 用默认 python
#   PY=/path/to/python bash scripts/run_all.sh
#
# 退出码: 任一步失败即中止并返回非 0。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
RAW="${RAW:-raw}"
DATA="${DATA:-data}"
REPORTS="${REPORTS:-reports}"

echo "=========================================="
echo " FLUID / TI 路口轨迹数据管线"
echo " 根目录 : $ROOT"
echo " 解释器 : $($PY --version 2>&1)"
echo "=========================================="

# 前置检查：原始数据不在仓库内，缺失时给出可操作的指引而不是 Python traceback。
if [ ! -d "$RAW/traj" ]; then
  echo ""
  echo "[错误] 缺少原始数据目录: $RAW/traj"
  echo ""
  echo "原始数据来自 Figshare FLUID v2 的 derived_data（场景 TI），未随仓库上传。"
  echo "请按 ti_trajectory/README.md 第 3 节准备，解压后应得："
  echo ""
  echo "  $RAW/traj/      20250529_video-{1,2,3}_Traj.csv"
  echo "  $RAW/route/     20250529_video-{1,2,3}_Route.csv"
  echo "  $RAW/signal/    20250529_video-{1,2,3}_signal.csv"
  echo "  $RAW/conflict/  TI_conflict.csv"
  echo "  $RAW/map/       TI.osm"
  echo ""
  echo "若只想查看已有评价结果（仓库已提交 data/ 与 reports/），无需原始数据，可单独运行："
  echo "  $PY scripts/evaluate.py --data $DATA --out $REPORTS"
  echo "  $PY scripts/verify_dataset.py --data $DATA"
  echo ""
  exit 2
fi

echo ""
echo "[1/4] 构建统一训练数据 ..."
"$PY" scripts/build_dataset.py --raw "$RAW" --out "$DATA"

echo ""
echo "[2/4] 数据质量体检 ..."
"$PY" scripts/qc_check.py --raw "$RAW" --out "$REPORTS"

echo ""
echo "[3/4] 建模与分类评价 ..."
"$PY" scripts/evaluate.py --data "$DATA" --out "$REPORTS"

echo ""
echo "[4/4] 数据契约校验（划分无泄漏等断言） ..."
"$PY" scripts/verify_dataset.py --data "$DATA"

echo ""
echo "=========================================="
echo " 全链路完成 ✓"
echo " 训练数据 : $DATA/"
echo " 报告     : $REPORTS/"
echo " 字段说明 : DATA_DICTIONARY.md"
echo "=========================================="
