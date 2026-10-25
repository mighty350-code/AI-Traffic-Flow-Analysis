# FLUID / TI 路口轨迹：统一训练数据与行为分类

基于 FLUID 开源数据集 **TI 场景**的三段完整航拍轨迹视频，整理出统一的训练输入，接入对应的路线与信号信息，并完成停车 / 直行 / 转弯三分类的建模与评价。

> 数据来源：Figshare **FLUID v2** `derived_data`（场景 TI）。
> FLUID 是第三方开源算法（[sysu19351014/FLUID](https://github.com/sysu19351014/FLUID)），本项目做的是数据工程与行为分类建模，不是原创 FLUID。

本目录是仓库的独立子项目，与主链 `src/`（检测 → 跟踪 → FLUID 后处理）互补：**主链产出轨迹，本目录把官方发布的轨迹整理成可复现的训练输入并做行为分类**。

---

## 1. 成果概览

| 交付物 | 位置 | 说明 |
|---|---|---|
| 统一训练数据 | `data/` | 轨迹级 / 窗口级 / 标签表 / 信号表 + 清单 |
| **字段说明** | [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | 每张表每个字段：含义、单位、取值域、来源、是否可作特征 |
| 质量体检报告 | `reports/qc_report.md`（+`.json`） | 时间轴 / 坐标 / 断轨 / 缺失值 / 路线 / 信号 |
| 分类评价报告 | `reports/eval_report.md`（+`.json`） | 五个层次的分类评价 + 12 张图 |
| 可复现脚本 | `scripts/` | 构建 → 体检 → 评价 → 校验，四步全链 |

**核心规模**：3 段视频 / 1515 条轨迹 / 39703 个滑动窗口 / 1457 条信号相位。

> `data/` 与 `reports/` 已随仓库提交，克隆后**无需原始数据即可直接查看评价结果**；要逐位重新生成 `data/`，则需按第 3 节准备 `raw/`。

---

## 2. 快速复现

```bash
# 0) 准备原始数据（见第 3 节）后，确保 raw/ 目录结构完整
# 1) 构建统一训练数据
python scripts/build_dataset.py --raw raw --out data
# 2) 数据质量体检
python scripts/qc_check.py --raw raw --out reports
# 3) 建模与分类评价
python scripts/evaluate.py --data data --out reports
# 4) 数据契约校验（划分无泄漏等断言）
python scripts/verify_dataset.py --data data
```

或一键跑完：

```bash
bash scripts/run_all.sh
```

也可以只校验已提交的数据集契约（无需 `raw/`）：

```bash
python scripts/verify_dataset.py --data data
```

**环境**：Python 3.12+，依赖 `numpy pandas scikit-learn matplotlib`（见仓库根 `requirements.txt`）。

**可复现性**：所有随机种子固定（`SEED = 42`）；模型选择按 val 宏 F1，test 集全程不参与选择；同一份 `raw/` 输入可重复生成逐位一致的 `data/`（已验证：全链路重跑后 6 个产物文件 MD5 全部不变）。

---

## 3. 数据来源与准备

原始数据来自 Figshare FLUID v2 的 `derived_data` 压缩包（TI 场景），解压后应得：

```
raw/
├── traj/      20250529_video-{1,2,3}_Traj.csv     逐帧轨迹点（含世界坐标、航向、速度）
├── route/     20250529_video-{1,2,3}_Route.csv   轨迹级路线（进/出方向、转向、信号交互）
├── signal/    20250529_video-{1,2,3}_signal.csv  相位级信号状态
├── conflict/  TI_conflict.csv                    交通冲突标注
├── map/       TI.osm                            路口 OSM 底图
└── video/     20250529_video-{1,2,3}.mp4         稳定化后的 10 FPS 航拍视频
```

> **原始数据不入库**：轨迹/路线/信号/冲突 CSV 与 OSM 底图体积约 85 MB，视频约 6.4 GB，且均非本项目原创，故按仓库惯例不上传。仓库仅提交本项目**加工产出**（`data/`、`reports/`、脚本、字段说明）。
>
> 获取方式：Figshare FLUID v2 `derived_data` 压缩包 → 解压出 `TI/` 目录 → 按上述结构放到 `raw/` 下。
> 视频为可选项，只有 `scripts/overlay_frames.py` 的轨迹叠加可视化需要；`scripts/fetch_videos.py` 可从 Figshare 按条目流式提取（ZIP 内 mp4 的 moov atom 位于文件末尾，必须顺序解压整个条目）。

---

## 4. 数据处理要点

### 4.1 划分粒度 = 整段视频（防相邻帧泄漏）

```
video-1 → train   519 条轨迹 / 14504 窗口
video-2 → val     524 条轨迹 / 12538 窗口
video-3 → test    472 条轨迹 / 12661 窗口
```

同一视频的全部帧、全部轨迹只出现在一侧。`verify_dataset.py` 断言：1515 条轨迹**跨边违规数 = 0**。

> **主键须用复合键 `(video_id, track_id)`**——`track_id` 在视频间重复（1515 条轨迹只有 984 个不同 `track_id`）。

### 4.2 质量检查结论

| 项 | 结论 |
|---|---|
| 时间轴 | `time == frame/10`；帧无缺失；id 内单调；无重复行 |
| 坐标 | speed 与坐标反算速度中位误差 0.010–0.015 m/s，P95 ≤ 0.077 m/s |
| 断轨 | 相邻观测帧间隔 >1 的次数 **0**；插值行占比 1.1%–3.6% |
| 缺失值 | traj NaN 占比 0.15%–0.43%，与 `isReal=0` 行**完全对应**；`course` 的 NaN 全在 id 末行（差分定义所致） |
| 路线 | `route.id ⊆ traj.id`；给出时刻的行**真正越界数 0**；geometry 与 traj 坐标中位偏差 0.0 m |
| 信号 | `in_state` 与信号表**一致率 1.0**；信号时间覆盖到轨迹末端 |

### 4.3 信号接入的语义处理

源信号表只对 8 个 (进口方向, 转向) 组合给出相位；route 用到的 **右转 `r` 与掉头 `t` 在本路口未单独信号化**——这是源数据语义，不是缺陷。处理方式：

- `signal_source` 列区分 `direct`（有专属相位）/ `straight`（回退同进口道直行相位）/ `unknown`；
- `signal_state_ref` 单独保留参考相位，评价时用 `kinematics_plus_signal` 与 `kinematics_plus_signal_ref` 两组特征对照，检验把参考相位当作实际所受相位是否误导模型——**结论是不会**（test 宏 F1 0.9392 vs 0.9397）。

---

## 5. 分类结果

### 5.1 窗口级三分类（自监督标签）

标签由运动学判据导出，故**判据量本身从特征集中剔除**，否则分类器只是复述标签规则、准确率虚高到近 1.0。

| 模型 | val 宏F1 | test 准确率 | test 宏F1 |
|---|---|---|---|
| GradientBoosting（选定） | 0.9714 | 0.9775 | 0.9666 |
| RandomForest | 0.9575 | 0.9659 | 0.9482 |
| LogisticRegression | 0.9049 | 0.9244 | 0.8888 |

### 5.2 轨迹级三分类：停车 / 直行 / 转弯

窗口特征按轨迹聚合（均值），每条轨迹一个样本，划分仍按整段视频。

| 子集 | 轨迹数 | test 准确率 | test 宏F1 |
|---|---|---|---|
| 全部 | 1515 | 0.9237 | 0.8746 |
| 机动车 | 1468 | 0.9187 | 0.8689 |

### 5.3 监督式转向分类（`route.turn` 独立真值）

这份真值独立于运动学判据，反映**真实泛化性能**。

| 特征设置 | 选定模型 | 特征数 | val 宏F1 | test 准确率 | test 宏F1 |
|---|---|---|---|---|---|
| kinematics_only | RandomForest | 35 | 0.9506 | 0.9397 | 0.9397 |
| kinematics_plus_signal | RandomForest | 39 | 0.9496 | 0.9393 | 0.9392 |
| kinematics_plus_signal_ref | RandomForest | 39 | 0.9499 | 0.9394 | 0.9393 |

### 5.4 轨迹级交叉校验

窗口标签按轨迹多数票回落为「转弯/直行」，与人工 `route.turn` 比对：**整体一致率 0.7900**（train 0.7749 / val 0.8027 / test 0.7904）。差异是两类标签的真实差异，作为一致性指标呈现。

### 5.5 停车 × 信号相位交叉验证

检验「停车」标签是否真的对应红灯停车。**口径很关键**：应取路口内窗口按其所在时刻的相位统计，而非「进路口时相位 → 该轨迹是否含任意停车段」（后者把进路口前排队也算进去，红绿灯差别会被抹平）。

| 相位 | 路口内窗口数 | stop 窗口占比 |
|---|---|---|
| G（绿） | 8122 | 0.0175 |
| r（红） | 1061 | **0.2837** |
| y（黄） | 312 | 0.0096 |

**红灯路口内停车占比是绿灯的 16.2 倍**，说明停车标签确实对应信号停车场景。

---

## 6. 目录结构

```
ti_trajectory/
├── README.md               本文件
├── DATA_DICTIONARY.md      字段说明
├── raw/                    原始数据（只读，不入库，按第 3 节自备）
├── data/                   统一训练数据（脚本产物，已入库）
│   ├── tracks.csv          轨迹级统一表      1515 × 59
│   ├── segments.csv        窗口级统一表      39703 × 54
│   ├── track_labels.csv    轨迹级分类清单     1515 × 36
│   ├── signal_states.csv   时序信号状态表     1457 × 9
│   ├── signal_intervals.csv 路口整体放行表    525 × 6
│   ├── conflict_events.csv 冲突事件表         69 事件
│   └── manifest.json       数据清单与统计
├── reports/                （已入库）
│   ├── qc_report.md/.json  数据质量体检
│   ├── eval_report.md/.json 分类评价
│   └── figures/            12 张图件
└── scripts/
    ├── build_dataset.py    构建统一训练数据
    ├── qc_check.py         数据质量体检
    ├── evaluate.py         建模与分类评价
    ├── verify_dataset.py   数据契约校验（断言防泄漏）
    ├── run_all.sh          一键复现
    ├── fetch_videos.py     视频下载
    └── overlay_frames.py   轨迹叠加可视化
```

---

## 7. 已知限制

- 行为标签为**自监督**（运动学判据导出），与人工标注一致率 0.79；真实泛化性能应以 §5.3 的监督式结果为准（test 宏 F1 0.9397）。
- `route_turn` 缺失 258 条（源数据未标注），这些轨迹的转向标签来自运动学回退，`track_class_source = default`。
- 停车标签的判定阈值为通用值（速度中位 < 0.5 m/s 且低速占比 ≥ 0.8），未按路口做逐相位标定。
- 三段视频均来自**同一路口 TI**，跨路口泛化能力未验证。
- 原始数据（`raw/`）未随仓库发布，克隆后若要重新生成 `data/` 需自行按第 3 节准备；已提交的 `data/` 可直接用于训练与评价。
