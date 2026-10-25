# 数据字段说明

FLUID / TI 路口统一训练数据集。本文说明 `data/` 下每张表的每个字段：**含义、单位、取值域、数据来源，以及是否可作为建模特征**。

> 生成方式：字段的取值域与缺失统计均由 `scripts/build_dataset.py` 从原始 CSV 实际计算后人工核对，未凭空虚构。示例数值取自本数据集实测。

---

## 0. 总览

| 文件 | 行数 | 一行代表 | 主键 |
|---|---|---|---|
| `tracks.csv` | 1515 | 一条完整轨迹（车辆） | `(video_id, track_id)` |
| `segments.csv` | 39703 | 一个滑动时间窗口 | `seg_idx` |
| `track_labels.csv` | 1515 | 一条轨迹的分类结果与判定依据 | `(video_id, track_id)` |
| `signal_states.csv` | 1457 | 一个 (进口方向, 转向) 的一个相位 | — |
| `signal_intervals.csv` | 525 | 路口整体放行状态的一个时间区间 | — |
| `conflict_events.csv` | 138 | 一辆车在一次冲突事件中的一条记录 | — |
| `manifest.json` | — | 数据清单、阈值、标签分布 | — |

**关键约定**

- **主键必须用复合键 `(video_id, track_id)`**。`track_id` 在不同视频间会重复（1515 条轨迹只有 984 个不同 `track_id`），单用 `track_id` 会把不同视频的轨迹误判为同一条。
- **划分粒度 = 整段视频**：`video-1`→train、`video-2`→val、`video-3`→test。同一视频的全部帧、全部轨迹只出现在一侧。
- 时间单位统一为**秒**，帧率 **10 FPS**（`time == frame / 10`）。
- 坐标单位统一为**米**（世界坐标 `cx_m` / `cy_m`），原点为路口几何中心附近。

---

## 1. `tracks.csv` — 轨迹级统一表（1515 行 × 59 列）

一条轨迹一行，已接入对应的路线（route）与信号（signal）信息。

### 1.1 标识与划分

| 字段 | 类型 | 含义 | 取值 |
|---|---|---|---|
| `video_id` | 字符串 | 来源视频 | `20250529_video-1/2/3` |
| `scene` | 字符串 | 路口场景标识 | 恒为 `TI` |
| `track_id` | 整数 | 轨迹编号（**视频内唯一**，跨视频可重复） | 1–1314 |
| `split` | 字符串 | 数据划分 | `train` / `val` / `test` |

### 1.2 目标与类别

| 字段 | 类型 | 含义 | 取值 |
|---|---|---|---|
| `type` | 字符串 | 车辆类型（源自 FLUID 检测） | `car`(1106) / `moped`(192) / `truck`(134) / `pedestrian`(41) / `trailer`(17) / `bus`(13) / `van`(6) / `tricycle`(6) |
| `is_motor` | 布尔 | 是否机动车 | True(1468) / False(47) |
| `track_class` | 字符串 | **轨迹级三分类标签** | `straight`(1048) / `turn`(321) / `stop`(146) |
| `track_class_source` | 字符串 | 标签判定依据 | `route_turn`(1132) / `default`(237) / `stop_ratio`(146) |
| `track_turn_class` | 字符串 | 轨迹级转向判定（仅看路口内） | `straight`(927) / `turn`(269) / `unknown`(319) |
| `route_turn_binary` | 字符串 | `route_turn` 的二值化 | `straight`(884) / `turn`(373) / 空(258) |
| `route_turn` | 字符串 | **人工路线标注**（原始转向） | `l` / `r` / `s` / `t` / 空(258) |
| `route_turn_label` | 字符串 | `route_turn` 的可读化 | `left` / `right` / `straight` / `u_turn` |
| `turn_label_agree` | 布尔 | 运动学转向判定与人工标注是否一致 | True(1125) / False(132) |
| `class_agree_route` | 布尔 | `track_class` 与 `route_turn` 是否一致 | True(1132) / 空(383) |

**`is_motor` 口径说明**：FLUID 官方 README 只给了 `e.g., car, moped, pedestrain` 的示例，未定义法规口径。本项目按中国大陆交规归类——`moped`（轻便摩托车）属机动车，`tricycle` 可能是人力三轮车故归非机动车，`pedestrian` 为行人。口径集中在 `build_dataset.py` 的 `MOTOR_TYPES` 一处定义，上游若细化词表只改这一行即可全量重算。

**标签来源优先级**（`track_class_source`）：

1. `stop_ratio` — 点级静止占比达标且存在连续静止段 ≥ 1 s，判为 **stop**（最优先，因为停车是点级属性）；
2. `route_turn` — 人工标注 `l/r/t` 判为 **turn**，`s` 判为 **straight**；
3. `default` — 无人工标注时，按运动学转向判定回落到 `straight` / `turn` / `unknown`。

### 1.3 时间轴

| 字段 | 单位 | 含义 | 取值 |
|---|---|---|---|
| `frame_start` / `frame_end` | 帧 | 轨迹首末帧号 | 1–16551 |
| `time_start` / `time_end` | s | 轨迹首末时刻 | 0.1–1655 |
| `duration` | s | 轨迹时长 | 1.1–1655 |
| `n_points` | 个 | 轨迹点总数（含插值行） | 12–16551 |
| `n_real` | 个 | 真实检测点数 | 4–16340 |

### 1.4 几何与运动学

| 字段 | 单位 | 含义 | 取值 |
|---|---|---|---|
| `length_med` | m | 车身长度中位数 | 1.02–22.29 |
| `width_med` | m | 车身宽度中位数 | 0.78–4.66 |
| `entry_direction` | 字符串 | 驶入路口的方向 | `N` / `E` / `W` / `Unknown` |
| `exit_direction` | 字符串 | 驶出路口的方向 | `E` / `W` / `N` / `S` / `Unknown` |
| `overall_direction` | 字符串 | `驶入-驶出` 组合 | 如 `N-E`、`W-E`、`N-Unknown` |
| `v_median` | m/s | 速度中位数 | 0–27.66 |
| `v_max` | m/s | 速度最大值 | 0.63–118.0 |
| `v_mean` / `v_std` | m/s | 速度均值 / 标准差 | — |
| `path_len` | m | 轨迹累计路程 | 1.24–326.2 |

### 1.5 数据质量标记

| 字段 | 类型 | 含义 | 取值 |
|---|---|---|---|
| `conf_mean` | — | 平均检测置信度 | 0.224–0.896 |
| `interp_ratio` | — | 插值行占比（`isReal=0`） | 0–0.8475 |
| `max_interp_run` | 帧 | 最长连续插值段长度 | 0–31 |
| `has_track_break` | 布尔 | 是否存在帧断口 | True / False |
| `n_jump` | 个 | 坐标跳变次数（速度 > 30 m/s） | 0–5 |

### 1.6 路线接入

来自 `raw/route/*_Route.csv`，一条轨迹对应一行。

| 字段 | 单位 | 含义 |
|---|---|---|
| `route_in_time` / `route_out_time` | s | 进入 / 驶出路口的时刻 |
| `route_in_state` / `route_out_state` | — | 进入 / 驶出时的信号相位（`G` 绿 / `r` 红 / `y` 黄） |
| `route_in_cycle` / `route_out_cycle` | — | 对应的信号周期序号 |
| `route_in_delta` / `route_out_delta` | s | 相对该相位起始的偏移 |

**缺失说明**：`route_in_time` 空 285 条、`route_out_time` 空 318 条、`route_turn` 空 258 条——这是**源数据本身未标注**，不是处理丢失。质检确认：凡给出了时刻的行，**真正越界（落在轨迹时间跨度外）的行数为 0**。

### 1.7 信号接入

| 字段 | 含义 |
|---|---|
| `sig_in_state` / `sig_out_state` | 由 `signal_states.csv` 按 (进口方向, 转向) 查得的**专属相位** |
| `sig_in_cycle` / `sig_out_cycle` | 对应周期序号 |
| `sig_in_state_ref` / `sig_out_state_ref` | 回退参考值：**同进口道直行相位** |
| `sig_state_match_in` | 查表结果与 `route_in_state` 是否一致（实测**一致率 1.0**） |

**为什么需要 `_ref` 参考列**：源信号表只对 8 个 (进口方向, 转向) 组合给出相位（`E:p, E:s, N:l, N:p, N:s, W:l, W:p, W:s`），而 route 标注用到了 `E:r, E:t, N:r, N:t, W:t`——**右转与掉头在本路口未单独信号化**，属源数据语义而非缺陷。因此：

- `signal_source`（见 segments）区分 `direct`（有专属相位，23982 窗口）/ `straight`（回退到直行相位，15521）/ `unknown`（200）；
- 评价报告用 `kinematics_plus_signal` 与 `kinematics_plus_signal_ref` 两组特征做对照，检验把参考相位当作车辆实际所受相位是否会误导模型——结论是不会（宏F1 0.9392 vs 0.9397）。

### 1.8 路口内统计

仅统计 `route.in_time` ~ `route.out_time` 之间的窗口。

| 字段 | 含义 |
|---|---|
| `n_seg_in_intersection` | 路口内窗口总数（空 319 条，对应无进出时刻） |
| `n_seg_in_turn` / `n_seg_in_straight` / `n_seg_in_stop` | 其中转向 / 直行 / 停车窗口数 |
| `turn_net_deg_in_intersection` | 路口内行驶段净转角（deg） |
| `turn_window_ratio_in_intersection` | 转弯窗口占行驶窗口比例 |

### 1.9 停车统计

| 字段 | 单位 | 含义 |
|---|---|---|
| `stop_ratio` | — | 点级静止占比（用点级而非窗口时长，避免滑动窗口重叠重复计时） |
| `stop_run_max_s` | s | 最长连续静止时长 |

---

## 2. `segments.csv` — 窗口级统一表（39703 行 × 54 列）

**滑动窗口**：长 30 帧（3.0 s），步长 10 帧（1.0 s），10 FPS 下共 39703 个窗口。

### 2.1 标识与标签

| 字段 | 类型 | 含义 |
|---|---|---|
| `seg_idx` | 整数 | 窗口全局序号（0 起，按视频内顺序） |
| `video_id` / `scene` / `track_id` | — | 同 tracks |
| `split` | 字符串 | 同 tracks |
| `label` | 字符串 | **窗口级三分类标签**：`stop`(18535) / `straight`(14467) / `turn`(6701) |
| `seg_start_frame` / `seg_end_frame` | 帧 | 窗口首末帧 |
| `seg_start_time` / `seg_end_time` / `seg_mid_time` | s | 窗口首末/中点时刻 |
| `seg_n_points` | 个 | 窗口内点数（恒为 30） |
| `seg_duration` | s | 窗口时长（恒为 2.9–3.0 s） |

**窗口标签判据**（`label` 由以下运动学量导出，因此这些量**必须**从建模特征集中剔除，否则分类器只是复述标签规则）：

| 标签 | 判据 |
|---|---|
| `stop` | `seg_speed_smooth_med` < 0.5 m/s **且** `seg_low_speed_ratio` ≥ 0.8 |
| `turn` | `seg_net_turn_deg` ≥ 30° **或** `seg_cum_turn_deg` ≥ 60° |
| `straight` | 以上都不满足 |

### 2.2 速度特征

| 字段 | 单位 | 含义 |
|---|---|---|
| `seg_v_mean` / `seg_v_max` / `seg_v_std` | m/s | 窗口内速度均值 / 最大 / 标准差 |
| `seg_v_first` / `seg_v_last` | m/s | 窗口首末点速度 |
| `seg_speed_smooth_mean` | m/s | 平滑后速度均值 |
| `seg_speed_smooth_med` | m/s | 平滑后速度中位数 ★**判据量，剔除** |
| `seg_speed_smooth_std` | m/s | 平滑后速度标准差 |
| `seg_speed_smooth_min` / `seg_speed_smooth_max` | m/s | 平滑后速度极值 |
| `seg_speed_smooth_p25` / `seg_speed_smooth_p75` | m/s | 平滑后速度四分位 |
| `seg_low_speed_ratio` | — | 低速帧占比 ★**判据量，剔除** |
| `seg_accel_abs_mean` / `seg_accel_abs_max` | m/s² | 绝对加速度均值 / 最大值 |

### 2.3 几何特征

| 字段 | 单位 | 含义 |
|---|---|---|
| `seg_path_len` | m | 窗口内累计路程 |
| `seg_displacement` | m | 首末点直线距离 |
| `seg_straightness` | — | 位移/路程，1 为直线 ★**判据量，剔除** |
| `seg_net_turn_deg` | deg | 净转角（首末航向差） ★**判据量，剔除** |
| `seg_cum_turn_deg` | deg | 累计转角（绝对值累加） ★**判据量，剔除** |
| `seg_turn_rate_deg_s` | deg/s | 转角速率 ★**判据量，剔除** |
| `seg_curvature` | 1/m | 曲率积分（= 累计转角/路程） |
| `seg_cx_m_mean` / `seg_cy_m_mean` | m | 窗口内世界坐标均值 |
| `seg_dist_to_center` | m | 到路口中心的距离（5.4–57.6） |
| `seg_interp_ratio` | — | 窗口内插值行占比（0–0.133） |
| `in_intersection` | 布尔 | 是否位于路口内（`route.in_time`~`out_time` 之间） |

### 2.4 轨迹上下文

| 字段 | 含义 |
|---|---|
| `track_type` | 车辆类型 |
| `is_motor` | 是否机动车 |
| `entry_direction` / `exit_direction` / `overall_direction` | 轨迹级方向，窗口内恒定 |
| `route_turn` / `route_turn_label` | 人工转向标注（空 1103） |
| `length_med` / `width_med` | 车身尺寸 |

### 2.5 信号特征

| 字段 | 含义 | 取值 |
|---|---|---|
| `signal_state` | 该窗口时刻的**专属相位** | `G` / `r` / `y` / 空 |
| `signal_cycle` | 相位周期序号 | 1–24 |
| `signal_source` | 相位来源标记 | `direct`(23982) / `straight`(15521) / `unknown`(200) |
| `signal_state_known` | 是否有已知相位 | True(23982) / False(15721) |
| `signal_state_ref` | 回退参考相位（同进口道直行） | `G` / `r` / `y` / 空 |

---

## 3. `track_labels.csv` — 轨迹级分类清单（1515 行 × 36 列）

从 `tracks.csv` 抽取的**精简分类表**，只保留分类结果与判定依据，便于直接用于训练或人工核查。

| 字段 | 含义 |
|---|---|
| `video_id` / `scene` / `track_id` / `split` / `type` / `is_motor` | 同 tracks |
| `track_class` / `track_class_source` | 分类结果与依据 |
| `stop_ratio` / `stop_run_max_s` | 停车判定依据 |
| `route_turn` / `route_turn_label` | 人工转向标注 |
| `class_agree_route` / `turn_label_agree` | 与人工标注的一致性 |
| `track_turn_class` | 运动学转向判定 |
| `entry_direction` / `exit_direction` / `overall_direction` | 方向 |
| `route_in_time` / `route_out_time` | 进出路口时刻 |
| `sig_in_state` / `sig_in_state_ref` / `sig_out_state` / `sig_out_state_ref` | 信号相位与参考值 |
| `duration` / `n_points` / `n_real` / `interp_ratio` | 时间轴与插值 |
| `v_median` / `v_max` / `path_len` | 运动学摘要 |
| `has_track_break` / `max_interp_run` / `n_jump` | 断轨与跳变 |
| `length_med` / `width_med` | 车身尺寸 |

---

## 4. `signal_states.csv` — 时序信号状态表（1457 行 × 9 列）

按 (进口方向, 转向) 展开的相位时间轴，可按时刻精确查询车辆所受相位。

| 字段 | 单位 | 含义 | 取值 |
|---|---|---|---|
| `video_id` / `scene` | — | 同 tracks | — |
| `direction` | — | 进口方向 | `E` / `N` / `W` |
| `turn` | — | 转向 | `p`(直行待行区) / `s`(直行) / `l`(左转) |
| `state` | — | 相位状态 | `G` 绿 / `r` 红 / `y` 黄 |
| `begin_time` / `end_time` | s | 相位起止时刻 | 0–1675 |
| `duration` | s | 相位时长 | 3–118 |
| `cycle` | — | 周期序号 | 1–24 |

---

## 5. `signal_intervals.csv` — 路口整体放行表（525 行 × 6 列）

把各进口方向的相位投影到同一时间轴，得到「路口整体是否放行」。

| 字段 | 单位 | 含义 | 取值 |
|---|---|---|---|
| `video_id` | — | 所属视频 | — |
| `begin_time` / `end_time` | s | 区间起止 | 0–1675 |
| `states` | — | 该区间内各方向出现的相位集合 | `G` / `r` / `y` / `r\|y` |
| `green_present` | 布尔 | 该区间是否有任一方向放行 | True / False |
| `n_groups` | 个 | 区间内相位组数 | 1–5 |

---

## 6. `conflict_events.csv` — 交通冲突事件表（138 行 × 15 列）

来自 FLUID 上游的交通冲突检测结果（`raw/conflict/TI_conflict.csv`）。**一行代表一辆车在一次冲突事件中的一条记录**，因此同一事件会出现两行（`role` = `car1` / `car2`）。

| 字段 | 单位 | 含义 | 取值 |
|---|---|---|---|
| `video_id` | — | 所属视频 | `20250529_video-1/2/3` |
| `frame` | 帧 | 冲突发生帧 | 1–16551 |
| `time` | s | 冲突发生时刻 | `time == frame/10` |
| `track_id` | 整数 | 涉事车辆轨迹编号（**视频内唯一**） | 1–1314 |
| `role` | — | 该车在冲突中的角色 | `car1` / `car2` |
| `peer_track_id` | 整数 | 冲突另一方的轨迹编号 | 1–1314 |
| `x` / `y` | m | 涉事车辆世界坐标 | — |
| `diff` | m | 两车间最小距离 | 中位 2.11，最小 0.01 |
| `TTC` | s | 碰撞时间 | 中位 0.095，最小 0.00 |
| `MTTC` | s | 修改的 TTC | 中位 2.40 |
| `DGT_PET` | m/s² | 避免碰撞的减速度需求 | 最大 4.00 |
| `conflict_type` | — | 冲突类型 | `rear end`(53) / `head on`(9) / `angle`(4) / `sideswipe`(3) |
| `matched` | 布尔 | 该车轨迹是否在 `tracks.csv` 中找到 | True(134) / False(4) |
| `split` | — | 所属划分（随 `video_id`） | `train` / `val` / `test` |

> 138 行对应 69 次冲突事件，涉及 134 条车-事件记录（其中 4 条涉及已被上游删除的轨迹，`matched = False`）。
> `TTC < 1.0 s` 的车-事件记录 96 条，可作为严重冲突子集。

---

## 7. `manifest.json` — 数据清单

| 字段 | 含义 |
|---|---|
| `scene` / `source` / `fps` | 场景、来源说明、帧率 |
| `videos` | 每个视频的 split、轨迹数、窗口数 |
| `thresholds` | 全部行为判据与清洗阈值（改阈值后须重跑 build） |
| `window` | 窗口长度、步长、秒数 |
| `n_tracks_total` / `n_segments_total` | 总量 |
| `label_distribution` / `label_distribution_by_split` | 窗口标签分布（全量及分侧） |
| `track_label_distribution` | 人工 `route_turn_label` 分布 |
| `track_class_distribution` / `..._by_split` / `..._motor_only` | 轨迹三分类分布（全量、分侧、机动车子集） |
| `track_class_source_distribution` | 标签来源分布 |
| `signal_source_distribution` | 信号相位来源分布 |
| `signal_state_known_ratio` / `signal_state_ref_known_ratio` | 信号覆盖率 |
| `turn_map` | `l/r/t/s` → `left/right/u_turn/straight` 映射 |

---

## 8. 建模使用约定

### 7.1 可直接用作特征的列

`segments.csv` 中**数值型且不属于判据列**的列，共 **27 个**（已剔除 7 个判据量 + 若干标识/字符串列）。`tracks.csv` 同理可得 36 个。

### 7.2 必须剔除的判据列（防标签泄漏）

```
seg_speed_smooth_med, seg_low_speed_ratio, seg_net_turn_deg,
seg_cum_turn_deg, seg_turn_rate_deg_s, seg_straightness, label
```

> 若不剔除，分类器只是复述标签规则，窗口级准确率会虚高到近 1.0，失去泛化意义。`scripts/evaluate.py` 的 `LEAKY` 常量与 `scripts/verify_dataset.py` 保持一致，并有断言守护。

### 7.3 字符串列的处理

`signal_state` / `signal_state_ref` 需先独热编码（`pd.get_dummies`）。**只对 `direct` 来源的专属相位做独热**更语义准确：无专属相位的右转/掉头轨迹会落到 `sig_nan` 哑元上，这本身携带信息（说明该转向未单独信号化）。

### 7.4 划分与防泄漏

- 划分粒度为整段视频，**相邻帧与同一辆车的轨迹绝不跨边**（已由 `verify_dataset.py` 断言：1515 条轨迹跨边违规数 = 0）。
- 轨迹级建模时，把窗口特征按 `(video_id, track_id)` 聚合后再划分，样本数等于轨迹数（1515），而不是窗口数。
- **注意复合键**：`track_id` 跨视频重复，单用 `track_id` 分组会错误地把不同视频的轨迹合并。

### 7.5 两套标签的用途

| 标签 | 来源 | 用途 |
|---|---|---|
| `segments.label` / `tracks.track_class` | 运动学判据导出（自监督） | 主线任务；须剔除判据量 |
| `tracks.route_turn` | **人工路线标注**（独立真值） | 交叉校验与监督式评价，反映真实泛化性能 |

两者一致率 0.7900——差异是真实存在的，报告中作为「运动学判据 vs 人工标注」的一致性指标呈现，而非误差。

### 7.6 检验「红灯停车」时的口径陷阱

**不要**用「轨迹进入路口时的相位 → 该轨迹是否含任意停车段」来检验红灯停车。该口径会把「进路口前排队」也算进去，与进路口时的相位无关，因此绿灯轨迹的停车率同样高（实测 0.373 vs 红灯 0.329），看不出差别。

**正确口径**：取**路口内**的窗口，按该窗口**所在时刻**的相位统计 stop 窗口占比。实测红灯 0.2837、绿灯 0.0175，**相差 16.2 倍**，这才是有意义的信号。

两种口径在 `reports/eval_report.md` 第 5 节并列给出，读者可自行判断。

---

## 9. 质量体检结论摘要

完整报告见 `reports/qc_report.md`，机器可读版 `reports/qc_report.json`。

| 检查项 | 结论 |
|---|---|
| 时间轴 | `time == frame/10` 成立；帧无缺失；id 内单调；无重复行 |
| 坐标 | 像素与世界坐标量纲一致；speed 与坐标反算速度中位误差 0.010–0.015 m/s，P95 ≤ 0.077 m/s |
| 断轨 | 相邻观测帧间隔 >1 的次数为 **0**；插值行占比 1.1%–3.6% |
| 缺失值 | traj NaN 占比 0.15%–0.43%，且与 `isReal=0` 行**完全对应**；`course` 的 NaN 全在各 id 末行（差分定义所致，非缺陷） |
| 路线 | `route.id ⊆ traj.id`；给出时刻的行**真正越界数为 0**；geometry 与 traj 坐标中位偏差 0.0 m |
| 信号 | `in_state` 与信号表**一致率 1.0**；信号时间覆盖到轨迹末端 |
