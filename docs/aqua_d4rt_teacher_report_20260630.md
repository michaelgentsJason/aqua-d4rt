# Aqua-D4RT 项目研究进展汇报

日期：2026-06-30

## 1. 一句话总结

Aqua-D4RT 是一个面向水下动态场景的 D4RT-native transient-aware static reliability 系统。它在 D4RT/OpenD4RT 的 query 层显式估计动态物体和悬浮颗粒对静态几何的污染，并输出可用于静态地图、点云、SfM/SLAM 前端的 `static_confidence`。

当前项目已经达到可以开始写 ICRA 主稿的状态，但主线必须收窄。最稳的论文主线不是“恢复一张干净 RGB 图像”，也不是“比 SAM/DINO 更会分割鱼”，而是：

> Aqua-D4RT estimates D4RT-native query-level static reliability for underwater dynamic scenes, reducing transient contamination and exposing a controllable contamination/retention/front-end Pareto under water-specific degradation.

中文口径：

> Aqua-D4RT 在 D4RT query 层估计水下动态场景的静态可靠性，显著降低 transient 污染，并在水下特有退化下提供 contamination / retention / front-end 的可控 Pareto。

这条 claim 现在是能立住的，但前提是严格控制边界：不写 broad underwater SLAM/SfM SOTA，不写 2D fish segmentation SOTA，不写 one-pass online deployment。

## 2. 研究问题与动机

水下机器人在真实环境中经常遇到鱼类、漂浮颗粒、marine snow、反光悬浮物、低照度、非均匀照明、turbidity/backscatter、模糊等退化。这些因素会污染静态地图、特征匹配和相机注册。

传统 2D prefilter 或 segmenter 可以在图像层抠出鱼，但它们不一定适合 D4RT 的 query-level static geometry：

- 2D mask 关注像素轮廓，不直接判断 query 是否对静态几何有用。
- Detector/SAM 对大目标有效，但容易过度遮挡静态背景纹理。
- 悬浮颗粒、marine snow 等小型 transient 不一定能被语义 detector 稳定覆盖。
- 下游 SfM/SLAM 不只需要“干净”，还需要足够的静态纹理和几何连通性。

因此本项目的核心目标是：在 D4RT query 层估计静态可靠性，而不是只做 dense pixel segmentation。

## 3. 为什么选择 D4RT 作为 backbone

选择 D4RT/OpenD4RT 的原因主要有三点。

第一，D4RT 本身已经具备视频级 query tracking 和 3D reconstruction 能力。我们不是从零训练一个水下重建模型，而是在已有 D4RT query 表达上增加 transient awareness，这样可以保留原始模型的几何能力。

第二，D4RT 的 query 正好对应机器人 mapping 里的“候选静态支持”。每个 query 都有时空位置、置信度和 3D 输出。Aqua 在这个层面判断 query 是否被鱼、颗粒或雪状噪声污染，比单纯像素 mask 更贴近静态地图和 SfM 前端。

第三，query-level 输出天然适合做 retention policy。我们可以根据任务选择不同 static-confidence 阈值：clean-map 模式更干净，front-end 模式保留更多静态结构以维持注册成功率。

## 4. 方法概述

在 OpenD4RT 的基础上，Aqua-D4RT 增加了两个 transient heads 和一个静态可靠性分数：

- `dynamic_object_logit`：预测鱼类、大型水下动物、潜水员等动态物体污染。
- `particle_logit`：预测悬浮颗粒、marine snow、小型瞬态噪声。
- `static_confidence = sigmoid(confidence) * (1 - sigmoid(dynamic)) * (1 - sigmoid(particle))`

实现上，模型头定义在 `src/model/heads.py`，静态分数融合在 `src/model/static_confidence.py`。训练损失里新增了 transient BCE 监督，见 `src/losses/d4rt_loss.py`。`geometry_masking` 是可选保护项，但当前主 checkpoint 并不依赖几何损失来撑 claim。

训练策略以 head-only / light adaptation 为主，不大改 D4RT backbone。这样能降低破坏原始几何能力的风险。当前主 checkpoint 的训练只更新 heads，冻结了 encoder、memory projection、query embedder 和 decoder。

下游使用上分为两类模式：

- Clean-map mode：高阈值，例如 `static_conf >= 0.55`，用于输出更干净的静态 query-map。
- Front-end / registration-first mode：低阈值，例如 WebUOT dynamic100 上 `0.15-0.25`，用于保留更多静态纹理和几何连通性。

此外，Tank GT-pose stress benchmark 上还实现了 v3 pose-aware retention scorer 和 R099 self-diagnostic multi-pass selector，用于验证污染、注册和位姿之间的 Pareto。它们是 downstream support，不是主方法的核心卖点。

## 5. 主要创新点

### 5.1 Query-level transient-aware D4RT mapping

贡献不是做一个新的 2D fish segmenter，而是把 transient awareness 做进 D4RT query 层。Aqua 直接判断每个 D4RT query 是否适合作为静态地图支持，从而生成 static query-map。

### 5.2 同时建模动态物体和颗粒类 transient

水下 transient 不只有鱼。Aqua 显式区分 dynamic object 和 particle 两类干扰。消融实验显示两个 head 都必要：去掉 dynamic head 会导致鱼类污染压不住，去掉 particle head 会导致颗粒检测崩。

### 5.3 Static query-map contamination 作为核心指标

项目定义并系统评估了 query-map / voxel / ORB feature / match contamination，而不是只报 mask F1。这让实验更贴近静态重建和机器人前端任务。

### 5.4 Retention-aware downstream Pareto

Aqua 不是简单把可疑区域全部删掉，而是提供可控 retention 策略。最新 R110 / R115 / R120 结果说明同一个 checkpoint 可以根据任务切换 operating point：高阈值用于 clean-map，低阈值用于更稳的前端注册。

### 5.5 水下退化鲁棒性和 matched baseline 体系

R117-R120 把 Aqua 放到非均匀照明、turbidity/backscatter、low light、blur 和 AQUALOC boundary sanity 下重新审视；R115 则把 GroundingDINO-box / GroundingDINO+SAM 变成了真正可讨论的强 baseline。这一层不是“附加实验”，而是把 claim 收紧并加固。

## 6. 模型改了什么，怎么训练的

### 6.1 模型改动

当前 Aqua 不是大改 backbone，而是在 D4RT query 头上做最小必要扩展：

- 保留原有 `xyz_3d`、`uv_2d`、`visibility`、`displacement`、`normal`、`confidence` 等 D4RT 输出接口。
- 新增 `dynamic_object_head` 和 `particle_head`。
- 用 `static_confidence` 将原始 confidence 与 transient 概率融合成静态可靠性分数。
- 训练侧保留 `geometry_masking` 作为可选项，但当前主 checkpoint 没有靠它来写主 claim。

### 6.2 训练路线

| Stage | Init | Frozen modules | Supervision | Steps | Decision |
| --- | --- | --- | --- | ---: | --- |
| Phase A synthetic pretrain | OpenD4RT init | encoder / memory_proj / query_embedder / decoder frozen | synthetic dynamic + particle transient BCE | 1000-pilot retained | retained as base checkpoint |
| Phase C small WebUOT adaptation | Phase A ckpt | same frozen modules | WebUOT dynamic BCE only | 300 | useful but not final |
| Final main mix checkpoint | Phase A ckpt | same frozen modules | WebUOT dynamic BCE + synthetic replay; particle loss off on real labels | 1000 | current best checkpoint |
| Query-balanced pilots | current best ckpt | same frozen modules | boundary / hard-negative variants | 300-400 | rejected as main checkpoint |

Phase A 的 synthetic 100-clip 版本是在 `configs/train_aqua_synth_phase_a_multiclip_100.yaml` 上跑出的 1k pilot，保留下来的基座是 `output/exp_aqua_d4rt/aqua_synth_phase_a_multiclip_100_1k/checkpoints/best.ckpt`。主 checkpoint `output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt` 则是在这之上做的 real+synth mix head-only 继续训练。

### 6.3 主 checkpoint 的具体训练方式

`configs/train_aqua_real_synth_mix_headonly.yaml` 的关键点是：

- 输入是 32-frame、256x256 clip。
- 训练集是 `WebUOT fish30 train24 x3` + `watermask_caves_100 train`。
- 验证集是 `WebUOT val6` + `watermask_caves_100 val`。
- `freeze_encoder=true`、`freeze_memory_proj=true`、`freeze_query_embedder=true`、`freeze_decoder=true`。
- 只训练 head，约 200 万参数量级，约占总参数 0.17%。
- `dynamic_object` 有监督，`particle` 在 real WebUOT mix 中不硬加假标签，以免把真实鱼监督变成错误颗粒监督。
- 最终主 checkpoint 的粒子行为主要继承自 synthetic pretrain，并通过 synthetic replay 保持。

这也是为什么我们不再盲目继续 retrain backbone：当前收益已经主要来自 operating point 和 retention policy，而不是更重的参数更新。

## 7. 当前数据集与实验规模

| 数据集 / benchmark | 规模 | 标注 / GT | 主要用途 |
| --- | ---: | --- | --- |
| Synthetic watermask_caves_100 | 100 clips，70/15/15 split；test 15 clips，491,520 query points | 完整 fish + particle mask | 最稳主 claim：static query-map 去污染 |
| WebUOT fish30 | 30 clips，train24 / val6 | tracked-target bbox mask | 真实水下动态验证，DINO/SAM 强 baseline |
| WebUOT dynamic100 | 100 clips，其中 70 条不在 fish30 | tracked-target bbox mask | 更大规模真实动态 stress test |
| WebUOT all238 | 238 clips，7,616 frames | tracked-target bbox mask | 当前最大真实 WebUOT scale-up |
| AQUALOC harbor07 sanity | 9 个 source clips；108 degraded clips | 真实背景 + synthetic transient 注入 | 外部背景 sanity，不是自然动态 AQUALOC SOTA |
| Tank GT-pose stress v2 | 48 full clips，192 windows；8 stress variants | GT pose + injected fish/snow masks | 下游 pyCOLMAP / pose / registration Pareto |
| Tank stress4 subset | 24 clips x 3 pyCOLMAP seeds = 72 records | high-stress GT pose | R099 主 downstream 表 |

WebUOT 的 caveat 必须保留：其 mask 是 tracked-target bounding box，不是完整 fish instance mask，也不标所有动态物体。

VAROS 仍然是候选外部 benchmark，但完整 SEQ1 zip 约 17.96GB，本地磁盘压力下尚未进入主 claim。

## 8. 核心实验结果

### 8.1 Synthetic degradation robustness：R117

R117 是当前最强的 synthetic 支撑。它不是“看起来能用”，而是把 Aqua 放进水下特有退化里做了完整 sweep。

| Method | Query contamination | Static retention | Voxel contamination |
| --- | ---: | ---: | ---: |
| Raw D4RT | 10.82% | 100.00% | 11.03% |
| Aqua pred transient filter | 2.58% | 96.90% | 3.30% |
| Aqua @0.25 | 2.11% | 95.95% | 2.71% |
| Aqua @0.55 | 0.97% | 89.70% | 1.25% |
| Temporal RGB | 9.23% | 97.17% | 10.23% |
| Oracle GT | 0.00% | 100.00% | 0.00% |

Transient-query detection：

- AUROC: 0.9678
- AP: 0.7959

解读：Aqua 的 query-level static reliability 在 non-uniform illumination、turbidity/backscatter、low light/noise、blur/flicker 以及组合 stress 下仍然稳定。R117 支持的是“可靠性信号”和“去污染能力”，不是 downstream SLAM 终局胜利。

### 8.2 WebUOT scale-up / matched baseline：R118 + R115

R118 扩展到 WebUOT dynamic100 all100 / new70，R115 进一步把 GroundingDINO-box / GroundingDINO+SAM 放进 matched-baseline 框架里。这里的重点是：Aqua 不该写成鱼轮廓分割 SOTA，而要写成更有用的 query-level retention policy。

| Dataset | Method | Query contam. | Static ret. | Feature contam. | Match contam. | E success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic100 all100 | Raw D4RT | 16.69% | 100.00% | 37.74% | 39.39% | 96.00% |
| dynamic100 all100 | Aqua @0.25 | 10.28% | 88.78% | 14.05% | 18.23% | 80.06% |
| dynamic100 all100 | GroundingDINO-box | 2.65% | 35.30% | 8.32% | 1.16% | 17.31% |
| dynamic100 new70 | Raw D4RT | 18.60% | 100.00% | 44.85% | 44.88% | 94.29% |
| dynamic100 new70 | Aqua @0.20 | 11.65% | 90.22% | 17.88% | 20.30% | 80.09% |
| dynamic100 new70 | GroundingDINO-box | 2.99% | 34.79% | 9.63% | 1.43% | 17.95% |
| all238 | Raw D4RT | 15.24% | 100.00% | 29.89% | 33.19% | 97.74% |
| all238 | Aqua @0.15 | 10.13% | 88.79% | 13.45% | 19.73% | 88.87% |
| all238 | Aqua @0.45 | 8.92% | 74.67% | 11.48% | 14.81% | 81.33% |

WebUOT fish30 仍然是诚实 caveat：

| Method | Query contam. | Static ret. |
| --- | ---: | ---: |
| Aqua @0.55 | 4.47% | 75.53% |
| GroundingDINO-box | 2.19% | 83.93% |

解读：

- fish30 上 GroundingDINO-box 是强 baseline，Aqua 不能声称在鱼 mask 上全面赢它。
- dynamic100 / new70 才是 reviewer defense 的重点：DINO-box 很干净，但 static retention 只有约 35%，E success 也掉到 17% 左右。
- Aqua @0.20-0.25 保留约 89%-90% 静态 queries，同时把 feature/match contamination 压下去，并保住约 80% 的 E success。
- all238 证明这不是 fish30 小样本偶然性。

### 8.3 AQUALOC external-background sanity：R119

AQUALOC 不是自然动态 benchmark，而是外部真实背景 sanity。它的意义是：Aqua 不只是对 WebUOT 有效，在真实海底/港口背景上也有方向一致的信号，但域差异很真实。

静态地图结果：

| Method | Query contamination | Static retention | Voxel contamination |
| --- | ---: | ---: | ---: |
| Raw D4RT | 14.43% | 100.00% | 20.27% |
| Aqua pred transient filter | 9.28% | 79.70% | 13.47% |
| Aqua @0.11 | 8.83% | 88.86% | 12.98% |
| Aqua @0.55 | 3.38% | 36.54% | 4.65% |
| Temporal RGB | 11.92% | 96.11% | 17.10% |
| Oracle GT | 0.00% | 100.00% | 0.00% |

ORB 前端结果：

| Method | Feature contamination | Match contamination | E success |
| --- | ---: | ---: | ---: |
| Raw D4RT | 59.84% | 42.48% | 61.23% |
| Aqua pred transient filter | 47.71% | 36.09% | 50.58% |
| Aqua @0.11 | 45.91% | 35.53% | 50.23% |
| Aqua @0.55 | 26.67% | 19.72% | 22.97% |
| Temporal RGB | 57.01% | 43.81% | 52.31% |
| Oracle GT | 0.00% | 0.00% | 62.79% |

Calibration result:

- R117-trained ridge scorer: query 0.87%, retention 86.07%, feature 66.41%, match 45.74%, E success 1.46% on the synthetic training sweep.
- On AQUALOC, the scorer lands at 6.53% query contamination, 68.58% retention, and 47.57% E success.
- The tuned ORB rule mostly falls back to fixed low thresholds, so the robust interpretation is that calibration helps operating-point selection but does not magically solve the AQUALOC boundary.

解读：AQUALOC 说明 Aqua 的信号在外部真实背景上仍然成立，但没有把 AQUALOC 变成自然动态 SOTA 的资格。它更像 boundary/sanity，帮助我们把 claim 写诚实。

### 8.4 Tank GT-pose downstream：R099 / R101

R099 是目前最强 downstream 结果，但它是 self-diagnostic multi-pass selector，不是 one-pass online method。R101 再次说明它有 Pareto 价值，但不是 uniform pose win。

| Method | Pose-eval success | Input registration | ATE RMSE | RPE trans RMSE | Feature contamination | Match contamination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 81.94% | 38.00% | 0.0435 | 0.0283 | 22.86% | 17.63% |
| Aqua hard mask | 84.72% | 35.29% | 0.0581 | 0.0410 | 9.94% | 9.66% |
| v3 pose-soft | 83.33% | 23.05% | 0.0306 | 0.0253 | 13.83% | 15.40% |
| R099 pose-soft + raw fallback | 90.28% | 40.13% | 0.0389 | 0.0281 | 15.17% | 15.73% |

all-stress seed42 的边界验证：

| Method | Pose-eval success | Input registration | ATE RMSE | RPE trans RMSE | Feature contamination |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 91.67% | 33.56% | 0.0379 | 0.0263 | 13.63% |
| v3 pose-soft | 83.33% | 25.65% | 0.0301 | 0.0220 | 8.44% |
| R099-style fallback | 89.58% | 42.68% | 0.0397 | 0.0250 | 9.29% |

解读：R099 和 R101 支持“污染更低、注册更完整”的 Pareto，但不支持“所有场景位姿都优于 raw”。论文里最多写到 downstream Pareto，不能写 broad SLAM SOTA。

## 9. 当前论文 claim 边界

可以写：

- Aqua-D4RT introduces query-level dynamic-object and particle heads into D4RT.
- Aqua produces `static_confidence` for transient-aware static query-map construction.
- Aqua significantly reduces static query-map contamination on synthetic full-mask data, real WebUOT bbox-labeled data, and AQUALOC external-background stress sanity.
- Dynamic and particle heads are both necessary.
- Aqua exposes a controllable clean-map / registration-first retention Pareto.
- On Tank stress4, R099 improves aggregate downstream metrics with a multi-pass caveat.
- DINO/SAM are strong 2D prefilter baselines, but can over-mask static structure on broader dynamic scenes.

不能写：

- 不能说 Aqua 恢复干净 RGB 图像。
- 不能说 Aqua 比 SAM/DINO 更会抠鱼轮廓。
- 不能声称 broad underwater SLAM/SfM SOTA。
- 不能声称所有下游 pose / registration 指标都全面优于 raw。
- 不能把 GT-box SAM / GrabCut 当作 fair non-oracle baseline。
- 不能忽略 WebUOT bbox-mask caveat。
- 不能把 R099 写成 one-pass online method。

## 10. 当前是否可以写 ICRA

我的判断：可以开始写 ICRA 主稿。

原因：

1. 主 claim 有强数字支撑：Synthetic raw 10.82% contamination 到 Aqua @0.25 2.11% / @0.55 0.97%，且保持 95.95% / 89.70% retention。
2. 有真实数据支撑：WebUOT fish30、dynamic100、new70、all238 都已覆盖。
3. 有强 baseline：Temporal RGB、GroundingDINO-box、GroundingDINO+SAM、GT-box SAM/GrabCut。
4. 有下游验证：ORB/SfM proxy 和 Tank GT-pose pyCOLMAP。
5. 有消融：dynamic/particle heads 和 static-score terms 都有结果。
6. 有边界验证：R101 all-stress seed42、AQUALOC sanity、DINO-box over-mask、WebUOT caveat 都能提前回答 reviewer。

论文写作时应该采用稳健叙事：Aqua 不是万能 SLAM 系统，而是让 D4RT 的静态 query-map 和下游前端在水下 transient 场景中更干净、更可控。

## 11. 下一步建议

### P0：开始 paper package

- 把 R117 / R118 / R115 / R099 的关键表格并入 `figures/aqua_paper_tables/`。
- 整理主图：Input / Raw D4RT query map / Aqua static query map / degradation Pareto / matched baseline / failure cases。
- 画方法图：video -> D4RT queries -> dynamic/particle/static heads -> static_confidence -> retention-aware mapping / front-end。

### P1：写论文骨架

建议章节：

1. Introduction：水下 transient 污染静态地图和 SfM 前端。
2. Related Work：D4RT / underwater mapping / dynamic SLAM / SAM-DINO prefilter。
3. Method：query-level heads、static confidence、retention policy。
4. Experiments：synthetic degradation、WebUOT all238、AQUALOC sanity、Tank GT-pose。
5. Limitations：WebUOT bbox、R099 multi-pass、非 broad SLAM SOTA。

### P2：低风险优化只做 calibration / selector

Query-balanced retraining 已做两版 head-only pilot，结果都不应替换主 checkpoint。R120 的结论也很清楚：当前 checkpoint 已经有可用的 operating-point story，后续优先做 calibration / retention scorer / selector，而不是继续盲目训练 heads。

当前主模型保持冻结：

- `output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt`

### P3：若还有时间，再补边界验证

如果时间允许，可以再做两件事：

1. 扩展 AQUALOC 多 sequence / 多起点 sanity。
2. 如果能补足磁盘或云端空间，再把 VAROS 至少跑一个 sequence 做补充 sanity。

但这两件都不应该挤占主稿写作的优先级。当前最有价值的工作是把 claim、图表和边界写清楚。

## 12. 关键 artifact 索引

主要 checkpoint：

- `output/exp_aqua_d4rt/aqua_real_synth_mix_headonly/checkpoints/best.ckpt`

Synthetic degradation / robustness：

- `tmp/aqua_degradation_r117_batch_20260630/`
- `tmp/aqua_degradation_r117_eval_20260630/static_map/`
- `tmp/aqua_degradation_r117_eval_20260630/orb_full/`

WebUOT scale-up / matched baseline：

- `tmp/aqua_degradation_eval_20260626/r118_all100_summary/`
- `figures/aqua_matched_baselines_20260626/`
- `figures/aqua_main_claim_hero_20260626/`

AQUALOC boundary sanity：

- `tmp/aqua_degradation_r119_batch_20260630/`
- `tmp/aqua_degradation_r119_eval_20260630/static_map/`
- `tmp/aqua_degradation_r119_eval_20260630/orb_full/`

Calibration / selector：

- `tmp/aqua_degradation_threshold_calibration_20260630_r117train/`
- `tmp/aqua_degradation_threshold_selector_20260630_r117tune/`

Tank downstream：

- `tmp/aqua_adaptive_v3_t073_full_stress4_selector_v3_rawfallback/`
- `tmp/aqua_adaptive_v3_t073_allstress_seed42_selector_v3_rawfallback/`

Paper tables / docs：

- `docs/aqua_matched_baseline_pareto_20260626.md`
- `docs/aqua_result_to_claim_review_20260625.md`
- `docs/aqua_underwater_degradation_optimization_plan_20260626.md`
- `docs/aqua_webuot_fish30_results.md`
- `docs/aqua_tank_pose_stress_gt_validation.md`
