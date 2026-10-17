# AI Traffic Flow Analysis

基于航拍视频的车辆检测、跨帧跟踪、轨迹处理与交通冲突指标计算。这里是从本地实验工作区筛出的代码版，原始视频、数据集、模型权重和生成结果均未上传。

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

本仓库只收录一份主代码和少量训练脚本。工作区中的 `01_data/`、`03_outputs/`、`04_models/`、`05_ops/`、`07_third_party/`、缓存、日志、PDF、临时调试脚本均未复制。运维脚本中还存在明文服务器凭据，绝不可直接上传。

FLUID 派生脚本清单、上游来源及许可证见 `third_party/NOTICE.md`。仓库中的衍生文件不应被表述为本项目独立原创算法。
