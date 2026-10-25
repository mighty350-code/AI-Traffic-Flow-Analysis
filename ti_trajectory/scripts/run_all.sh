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
