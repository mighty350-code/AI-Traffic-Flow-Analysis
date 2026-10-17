# FLUID 来源说明

**FLUID 是第三方现成算法，并非本项目原创。**上游来源是 [sysu19351014/FLUID](https://github.com/sysu19351014/FLUID)。本项目在它的基础上做了输入格式衔接、脚本适配、参数调整和实验。整理时对照的上游提交为 `4d93d0663bb2bd2784f575a55a04906306c95d3e`。FLUID 使用 Apache License 2.0；许可证副本见 `FLUID_LICENSE`。

本仓库以下文件对应 FLUID 上游的轨迹处理和交通冲突分析代码，包含原样保留或项目改动的内容：

- `src/2_holography_smooth_kinematic_4K_batch.py`、`src/3_Object_Type_Mapping.py`、`src/4_Multi-source_Data_Integration_and_Cleaning.py`、`src/5_Motion_Correction_and_Behavior_Labeling.py`，对应上游 `demo/data_process/`。
- `src/2_droneTwoDimSSM.py`、`src/2_droneDGTBoundary.py`、`src/4_Conflict_Full_Statistics.py`、`src/5_Batch_Analysis_of_Objects_Near_Conflicts.py`，对应上游 `demo/data_verification/`。其中上游二维 SSM 文件名为 `1_droneTwoDimSSM.py`。

`src/detect_vehicles.py`、`src/track_vehicles.py`、`src/video_to_txt.py` 和 `src/run_pipeline.py` 是本项目用于检测、跟踪和串联 FLUID 的代码；它们调用的 Ultralytics YOLO 与 BoT-SORT 也属于外部技术。

本仓库不打包 FLUID 完整源码、数据集、前端和模型文件。使用完整后处理链时需另行取得上游仓库，并核对输入格式、标定文件、版本和各阶段参数。
