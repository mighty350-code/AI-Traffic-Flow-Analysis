# AI Traffic Flow Analysis

基于航拍视频的车辆检测、跨帧跟踪、轨迹处理与交通冲突指标计算。这里是从本地实验工作区筛出的代码版，原始视频、模型权重和第三方原始数据集均未上传。

仓库另含一个已随库提交的加工产出子项目 `ti_trajectory/`：基于 FLUID 官方发布的 TI 路口轨迹整理出的统一训练数据、字段说明与行为分类评价结果（见下节）。

**来源说明：FLUID 是从 GitHub 引入的第三方开源现成算法，由 [sysu19351014/FLUID](https://github.com/sysu19351014/FLUID) 提供，并非本项目原创。本项目的工作是将它接入自采航拍视频流程，编写检测与跟踪衔接代码，对部分脚本和参数做适配与实验。YOLO/BoT-SORT 同样是外部模型与算法，本项目没有原创这些模型架构。**

## 当前实现

| 环节 | 主要代码 | 当前做法 |
|---|---|---|
| 静态图像检测 | `src/detect_vehicles.py` | Ultralytics YOLO26m 通用权重，按车辆类筛选 |
| 视频感知与跟踪 | `src/video_to_txt.py`、`src/track_vehicles.py` | HBB 检测 + BoT-SORT 产出 ID；另用 YOLO26m-OBB 匹配旋转框，写出 FLUID 可读的 HBB/OBB 文件 |
| 航拍域训练 | `src/train_aerial.py`、`src/train_detrac.py`、`training/obb_train*.py` | 数据转换和微调脚本；本地另有 `ta_obb_best.pt`，未随仓库发布 |
| 轨迹处理 | `src/2_holography_*.py` 至 `src/5_Motion_*.py` | 基于 FLUID 上游脚本的项目适配版本，处理平滑、坐标变换、类别映射、清洗和行为标注 |
| 安全指标 | `src/2_drone*.py`、`src/4_Conflict_*.py`、`src/5_Batch_*.py` | 对应 FLUID 上游交通冲突分析脚本；包含本地改动 |
| 编排 | `src/run_pipeline.py`、`src/pipeline_config.yaml` | 按阶段调用检测、FLUID 后处理和数据转换 |
| **轨迹数据集与行为分类** | [`ti_trajectory/`](ti_trajectory/) | 把 FLUID 官方发布的 TI 路口三段完整轨迹整理成统一训练输入，接入路线与信号信息，产出停车 / 直行 / 转弯三分类的评价结果（详见下节） |

## TI 路口轨迹数据集（`ti_trajectory/`）

主链产出的是「从自采视频得到轨迹」；`ti_trajectory/` 是互补的另一环：**把 FLUID 官方发布的 TI 路口三段完整轨迹，整理成可重复生成的训练输入并做行为分类**。原始数据来自 Figshare FLUID v2 `derived_data`（场景 TI），非本项目原创。

| 交付物 | 位置 |
|---|---|
| 统一训练数据（轨迹级 / 窗口级 / 标签表 / 信号表） | `ti_trajectory/data/` |
| **字段说明**（每表每列：含义、单位、取值域、来源、是否可作特征） | `ti_trajectory/DATA_DICTIONARY.md` |
| 数据质量体检（时间轴 / 坐标 / 断轨 / 缺失值） | `ti_trajectory/reports/qc_report.md` |
| 分类评价报告 + 12 张图 | `ti_trajectory/reports/eval_report.md` |
| 可复现脚本（构建 → 体检 → 评价 → 契约校验） | `ti_trajectory/scripts/` |

**规模**：3 段视频 / 1515 条轨迹 / 39703 个滑动窗口 / 1457 条信号相位。

**划分粒度 = 整段视频**（video-1 train、video-2 val、video-3 test），同一视频的全部帧与轨迹只出现在一侧，相邻帧不会跨边混入；`scripts/verify_dataset.py` 以断言方式校验（跨边违规数 = 0）。主键须用复合键 `(video_id, track_id)`——`track_id` 在视频间重复。

**主要结果**：

| 任务 | 指标 |
|---|---|
| 监督式转向分类（`route.turn` 独立真值，唯一可信泛化指标） | test 宏 F1 **0.9397** |
| 轨迹级三分类 停车/直行/转弯（机动车子集） | test 宏 F1 0.8689 |
| 窗口级三分类（自监督标签，判据量已剔除防泄漏） | test 宏 F1 0.9666 |
| 运动学标签 vs 人工路线标注一致率 | 0.7900 |
| 红灯路口内停车占比 vs 绿灯 | 0.2837 vs 0.0175（**16.2×**） |

`data/` 与 `reports/` 已随仓库提交，克隆后无需原始数据即可查看全部结果；逐位重新生成 `data/` 需按 `ti_trajectory/README.md` 第 3 节准备 `raw/`。

## 依赖与使用

1. 安装 `requirements.txt` 中的 Python 依赖。
2. 从 [FLUID 原作者的 GitHub 仓库](https://github.com/sysu19351014/FLUID)单独取得源码，放在 `third_party/FLUID/`。本仓库不包含其完整源码、数据集和前端。
3. 把自己的视频放在 `data/`，模型权重放在 `models/`，并修改 `src/pipeline_config.yaml` 中的视频、模型、标定和阶段路径。FLUID 的地理变换阶段还需要对应的标定文件。
4. 先查看阶段，再按需要运行。完整流水线必须先备齐 FLUID、输入数据和权重。

```bash
python src/run_pipeline.py --list
python src/run_pipeline.py --dry-run
python src/track_vehicles.py --source data/input.mp4 --model models/yolo26m.pt
python src/video_to_txt.py --source data/input.mp4 --model models/yolo26m.pt --obb-model models/yolo26m-obb.pt
```

## 已知限制

- 通用 YOLO 权重对本项目航拍画面的细小车辆仍有漏检；`video_to_txt.py` 中 OBB 推理分辨率仍固定为 1024。仓库不宣称已经做到全车识别。
- `ta_obb_best.pt` 是本地航拍域训练权重，但现有主流水线配置仍默认通用 OBB 权重；切换权重前需核对类别词表和检测效果。
- `run_pipeline.py` 默认调用 FLUID 上游后处理脚本。仓库中的同名项目改编脚本是另一套实现，清洗阈值不同，不能直接视为主链实际运行代码。
- `video_to_txt.py` 先输出无表头 TXT，配置中的 `structure` 阶段调用 FLUID 第 1 步补成 CSV；不要把原始 TXT 直接交给第 2 步。另一条 `make_multitrack_video.py` 实验链产生的 CSV 缺少真实 `confidence`、`type`，本仓库未把它当成主链输入。
- FLUID 原版第 4 步对 DJI 文件名没有适配，会跳过数据。完整串联前需选择并核对项目适配版；`run_pipeline.py` 会在缺失必需脚本时中止，但不能仅凭退出状态判定结果质量。
- `src/2_holography_smooth_kinematic_4K_batch.py` 的发布副本按上游 `video_to_txt.py` 的弧度 `r` 读取。官方数据约 10 FPS，本项目原视频帧率不同，运行前须设置对应 FPS 和标定；仓库尚无可直接复现全流程的统一配置。
- `training/obb_train_prep.py` 使用已有检测结果制作训练标注，属于伪标签流程；使用前须人工核查标注。脚本仍带有历史视频命名约定，需按新数据修改。
- 代码保留了实验阶段的实现和参数。本仓库是可审阅的代码快照，不附带一键复现所需的私有素材。

## 文件取舍

本仓库收录一份主代码、少量训练脚本，以及 `ti_trajectory/` 子项目（加工产出的训练数据、字段说明、评价报告与复现脚本）。工作区中的 `01_data/`、`03_outputs/`、`04_models/`、`05_ops/`、`07_third_party/`、缓存、日志、PDF、临时调试脚本均未复制。运维脚本中还存在明文服务器凭据，绝不可直接上传。

`ti_trajectory/` 只收录本项目**加工产出**的部分：FLUID 官方发布的原始轨迹/路线/信号/冲突 CSV、OSM 底图与视频均未入库（体积约 85 MB + 6.4 GB，且非本项目原创），获取方式见 `ti_trajectory/README.md` 第 3 节。

FLUID 派生脚本清单、上游来源及许可证见 `third_party/NOTICE.md`。仓库中的衍生文件不应被表述为本项目独立原创算法。
