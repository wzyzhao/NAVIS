# 当前实验结果与下一步计划

核查日期：2026-09-29。以下结论来自现存 JSON、manifest、逐帧指标和实网报告；没有新增编码或无线运行。

## 1. 目前完成了什么

已有七个 MOT17 训练序列（02/04/05/09/10/11/13）的编码候选、七序列离线消融；实网归档为 NAVIS/ELIC 各两个单节点 run01，加一组双节点 health_weighted run01。另有同接收载荷的 no-proxy 重建和检测/状态评估。尚没有归档的同轨迹同预算编码器对照、equal_share 实网对照、四节点、掉线恢复或独立重复运行。

## 2. decoded 应该存什么；YUV 是否删除

| 路径 | 阶段 | 应保存的内容 |
|---|---|---|
| `encoding/dcvc/decoded/<sequence>/` | 发送前离线参考解码 | background / required_background / roi 解码 YUV、逐帧质量 |
| `encoding/elic/decoded/<sequence>/` | 发送前离线参考解码 | 本地完整合成帧/视频（已存在时）、逐帧质量 |
| `transmission/<method>/<sequence>/runNN/receiver/` | 真实接收 | 完整重组的 bin、CRC/ACK、接收 manifest |
| `transmission/.../rebuild/decoded_layers/` | 接收端解码 | 实际收到载荷对应的背景/ROI 层；ELIC 的现有入口直接输出 rebuild/frames |
| `transmission/.../rebuild/frames/` 或 `reconstructed_videos/` | 接收端重建 | 当帧收到内容与配置允许的历史缓存合成图像/视频、状态和任务指标 |

**保持这两类 decoded 分开。** 接收端内容必须按 run、节点、链路/缓存配置归档；把它们混入编码参考目录，会把“本地可解码”误当作“实网已交付”。

DCVC YUV 是 DCVC 工具解码码流后保存的未压缩 YUV420 像素帧，不是待发码流，不是接收端视频。第三方 `src/utils/video_writer.py:YUV420Writer` 依次写入 8-bit Y、U、V 平面，没有视频容器或自带时间戳。偶数宽高下每帧大小是 W×H×1.5 字节；1920×1080 一帧约 3.11 MB，多帧、多 QP 很快占用大量空间。ROI 文件对应 crop 的尺寸，不能都按全帧尺寸解释。

| DCVC decoded YUV | 文件数 | 字节数 |
|---|---:|---:|
| background | 18 | 41,698,022,400 |
| required_background | 69 | 188,121,600 |
| roi | 156,269 | 9,550,968,762 |
| **合计** | **156,356** | **51,437,112,762（47.90 GiB）** |

**本次保留 YUV，当前不建议直接删除。** `offline_schedule_simulator.py` 的候选再导出读取解码背景；完整合成和 PSNR 复算还读取 ROI 解码 YUV。`run_level=export_candidates` 明确复用这些缓存，缺失后会报错。直接读取已存指标、离线调度重放、发送既有 payload、从实网 receiver 重建不依赖这些编码端 YUV。

可在结果冻结后清理：保留所有 encoded/bin、payload/bin、manifest、模型版本、逐帧指标和尺寸/QP 信息；先提供从既有码流重新解码的独立入口，再把 YUV 标为可删除缓存。删除后复算会增加解码耗时。当前主入口 refresh_full 会重新编码，不能把它等同于已有的便捷缓存恢复命令。不要删除接收端最终重建帧/视频，也不要连带删掉 decoded 中的 frame_metrics.json。

## 3. 编码层结果与覆盖缺口

| 序列 | 候选帧数 | DCVC 有 PSNR 帧数 | DCVC PSNR（dB） | DCVC 历史估计 BPP | ELIC 有质量帧数 | ELIC 已测帧 PSNR（dB） |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02-FRCNN | 600 | 600 | 31.736 | 0.022431 | 0 | 未测 |
| MOT17-04-FRCNN | 1050 | 1050 | 22.950 | 0.014390 | 11 | 34.778 |
| MOT17-05-FRCNN | 837 | 837 | 29.620 | 0.188975 | 0 | 未测 |
| MOT17-09-FRCNN | 525 | 525 | 32.738 | 0.070810 | 525 | 33.138 |
| MOT17-10-FRCNN | 654 | 654 | 32.254 | 0.035829 | 0 | 未测 |
| MOT17-11-FRCNN | 900 | 900 | 32.155 | 0.060065 | 0 | 未测 |
| MOT17-13-FRCNN | 750 | 750 | 31.751 | 0.022502 | 0 | 未测 |

DCVC 七序列 5,316 帧均有 PSNR，根据逐帧记录复核的七序列等权 macro 为 **30.4574 dB**，已修正旧 overall 与逐序列结果不一致的问题。跨序列统一只使用 macro，并保留逐序列数据。DCVC 当前配置 calc_ssim=false，未得到全序列 MS-SSIM；底层值 0 不能视为实际测得的 MS-SSIM。

ELIC 有七序列候选，却仅 **536/5,316 帧**在当前 frame_metrics 中有质量值：MOT17-04 11 帧、MOT17-09 525 帧。summary 已将七序列 PSNR、MS-SSIM 与未审计 BPP 标为 NA，保存逐序列有效帧数；frame_count=5316 不能解释成 5316 帧已评估。已有 MOT17-04 图像并不能补回被覆盖/缺失的逐帧质量记录。当前不能据这两份摘要宣称七序列完整的 DCVC/ELIC 公平率失真比较。

证据：[DCVC 汇总](encoding/dcvc/quantitative_metrics.json)、[DCVC 帧指标索引](encoding/dcvc/frame_metrics.json)、[ELIC 帧指标索引](encoding/elic/frame_metrics.json)、[ELIC 摘要](encoding/elic/summary.json)。encoding/metrics/candidate_minimal_metrics 仅为旧四序列/3162 帧快照，主统计应使用当前七序列入口；不能把旧快照称作完整结果。

## 4. 离线调度与基线

七序列等权 macro；每个序列先按物理 ROI 计数计算比例，多质量候选不重复计目标。

| 七序列消融 | 当前覆盖 macro | TTL 可用率 macro | 平均录取 MB/序列 |
|---|---:|---:|---:|
| navis_ack | 69.86% | 88.47% | 2.2335 |
| no_proxy | 69.86% | 69.86% | 2.2335 |
| single_q_roi | 37.14% | 56.09% | 2.1320 |

`navis_ack` 是历史变体名：本实验按候选大小、元数据压力（6–40 kB）和 10 帧 TTL 离线选择，没有真实 ACK。no_proxy 同选择、仅关闭 TTL 模型；single_q_roi 仅使用最高质量候选。budget_fill 有可选实现，尚无七序列结果。三者 metadata 窗口录取完整率均为 100%，不能当作无线交付完整率。

证据：[macro 汇总](ablations/scheduler/report.md)、[逐序列结果与机器可读 macro](ablations/scheduler/summary.json)。固定数量和固定预算相关消融、扫描、重放与绘图已清理。ELIC FIFO 实网 baseline 保留，现有离线单序列入口未归档独立结果，不能视为已完成受控比较。

## 5. 单节点与双节点真实链路

| 单节点 run01 | 帧数 | 当前收到/应用 ROI | 视频合成 proxy | 最终应用覆盖 | 确认应用载荷（MB） | goodput（kbps） | RTT p95（s） |
|---|---:|---:|---:|---:|---:|---:|---:|
| NAVIS mot17_04 | 1050 | 3790/4343（87.27%） | 275 | 93.60% | 1.997 | 6.66 | 11.90 |
| NAVIS mot17_09 | 525 | 1987/4439（44.76%） | 766 | 62.02% | 4.409 | 10.13 | 6.57 |
| ELIC mot17_04 | 1050 | 820/4343（18.88%） | 0 | 18.88% | 4.480 | 10.69 | 5.70 |
| ELIC mot17_09 | 525 | 274/4439（6.17%） | 0 | 6.17% | 2.414 | 12.37 | 4.88 |

四次单节点重建均有全部目标帧、无背景缺失；NAVIS 的解码报告错误数为 0。不同方法的实网运行并未配对到同链路/同确认字节，单次结果只能支持端到端原型表现，不能分离编码器、预算、FIFO、proxy 和链路的贡献。ACK coverage=100% 的分母是被选中的消息，不是全部源 ROI；尤其 ELIC MOT17-09 的低覆盖主要受选择限制。

no-proxy 以同一 NAVIS 接收数据重新重建，当前 ROI 不变，视频 proxy 从 275/766 变为 0；最终覆盖变为 87.27%/44.76%。它无需再占用无线链路，属于接收端重建消融。

双节点目前仅 health_weighted 的 MOT17-04/09 run01：372 轮、确认应用载荷 6.990 MB、goodput 5.23 kbps、RTT p95 13.15 s。node_1 当前 ROI=4227/4343（97.33%）、加视频 proxy 69 后覆盖98.92%；node_2 当前2246/4439（50.60%）、加视频 proxy824后覆盖69.16%。两节点均完成目标帧重建、解码错误0。

该组已验证双节点接收会话隔离与端到端工作流。没有 equal_share 配对结果、节点交换、四节点或重复运行，不能据不同序列的 ROI 百分比断定策略公平/不公平。node_2 在第197轮结束后 node_1 接续使用全局资源；后期预算闲置同时受可录取需求不足影响。发送日志中的模块确认、应用ACK完整消息和接收端重组文件计数有差异，应按 session/message ID 对账，而不是相互替代。

证据：[MOT17-04 实网对比](baselines/navis_vs_elic/mot17-04/run01/report.md)、[no-proxy](transmission/navis/no_proxy_real_payload_summary.md)、[双节点分析](transmission/multinode/health_weighted/mot17_04_09/run01/multinode_run01_analysis.md)。

## 6. 接收后的视觉与状态任务

| 单节点 NAVIS 检测一致性 | 完整收到的参考 ROI | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| mot17-04 | 3790 | 83.21% | 75.30% | 79.06% |
| mot17-09 | 1987 | 72.29% | 82.84% | 77.20% |

双节点检测一致性 F1 为 node_1 77.58%、node_2 78.43%。这些任务的参考是已交付 ROI 在源语义元数据中的位置，衡量检测器一致性；未收到的源目标不进入该参考，因此不能解释为全场景人工真值准确率。no-proxy 下还保留了全源语义参考与已收到ROI参考两种历史云端结果，引用时必须标明分母。

| track/TTL 状态模型 | 当前覆盖 | TTL=10 的 task-ready | task-ready 连续性 |
|---|---:|---:|---:|
| NAVIS 04（TTL=10） | 87.27% | 94.13% | 95.40% |
| NAVIS 09（TTL=10） | 44.76% | 65.35% | 67.42% |
| ELIC 04（TTL=0） | 18.88% | 18.88% | 3.48% |
| ELIC 09（TTL=0） | 6.17% | 6.17% | 0.47% |
| 双节点 node_1（TTL=10） | 97.33% | 99.01% | 99.21% |
| 双节点 node_2（TTL=10） | 50.60% | 72.67% | 74.34% |

此模型直接使用源 track ID，并非独立图像 tracker 的 IDF1/MOTA/HOTA。任务 TTL 与视频合成缓存不同：例如单节点04任务 proxy298、视频proxy275，09分别914/766；双节点分别73/69和980/824。这些数字不冲突，表示不同状态/视觉应用条件，论文必须分表。

## 7. 下一步实验：按优先级执行

| 优先级 | 实验 | 控制与对照 | 必须报告 | 归档位置 |
|---|---|---|---|---|
| P0 | 补齐编码质量与指标审计 | 七序列同源完整帧，固定JPEG质量/候选版本；跨序列表只采用七序列等权macro；按完整源像素和实际载荷字节统一BPP及开销口径（见metric_protocol.md）；补全ELIC缺失质量和DCVC MS-SSIM | 有效评估帧数、PSNR/MS-SSIM、ROI质量、视觉/metadata/封装字节分解、权重哈希 | `encoding/*/decoded/<sequence>/`，跨方法表放 `encoding/metrics/` |
| P0 | 编码器×调度器的受控比较 | DCVC/ELIC×FIFO/语义选择，固定同源物理ROI、窗口、总确认预算/服务轨迹；proxy先关闭，后单独打开 | 当前ROI、task-ready、实际确认字节、质量、预算利用率；收益分别归因 | `baselines/controlled_codec_scheduler/<sequence>/runNN/` |
| P1 | 重复与场景覆盖 | 先04/09做配对独立重复，建议至少3次，资源允许5次以上；方法顺序随机/交叉，并覆盖好/差链路 | 每次配置与RSSI/链路日志、均值/标准差/区间、失败运行；不要只汇报最好一次 | 新 `transmission/.../run02` 起 |
| P1 | 人工真值视觉任务 | 同检测器/跟踪器分别运行原图、离线完整重建、实网重建、no-proxy；使用本地MOT17 train/gt/gt.txt及有效目标/忽略区域规则 | 原图上限、全源GT检测Precision/Recall/AP；独立跟踪IDF1/HOTA/MOTA，漏交付目标仍计漏检 | `evaluation/ground_truth/<sequence>/runNN/` |
| P1 | 多节点公平性与扩展 | 同两节点对比equal_share、health_weighted、需求/公平约束；交换节点与序列；再扩展4节点，改变全局预算和异质链路 | 每节点配额/确认字节/需求/ROI可用率，Jain值及使用变量定义，最差节点、饥饿时间、资源回收 | `transmission/multinode/<policy>/<sequence_set>/runNN/` |
| P1 | 掉线、ACK丢失、缺片与重启 | 受控丢片/延迟、单节点断连、receiver/sender重启、背景关键帧丢失 | 恢复耗时、重复/误应用、CRC失败、metadata完整率、背景年龄、依赖恢复、会话隔离 | `ablations/recovery/<fault>/runNN/`，真实载荷另保存 |
| P1 | 端到端性能与能耗 | 同硬件、并发、源帧；分开测场景分解、编码、排队、无线/ACK、接收、解码、云端推理 | p50/p95/p99年龄/时延、CPU/GPU峰值内存、处理帧率、排队长度、modem能量/每可用ROI能量 | `performance/<environment>/runNN/` |
| P2 | proxy/质量/背景参数敏感性 | 同接收载荷扫TTL=0/3/6/10/20；单/多质量、背景周期、预算填充、mask刷新；同时覆盖高速运动/遮挡 | 状态可用与视觉误差的取舍、过期appearance/错位、ROI局部质量、背景新鲜度 | `ablations/cache_and_quality/<sequence>/runNN/` |

建议先完成“质量/口径修正 → 同轨迹因素隔离 → 人工GT与多节点配对 → 实网重复与故障/性能”，再扩大规模。规则型调度已能进行上述验证，不必先引入AI预算预测；如果未来加入学习模型，还要单独定义按序列划分的训练/验证/测试、影子决策和同预算对照。

实网不可能把两次独立蜂窝运行视为严格同一轨迹：先用受控服务重放隔离机制，再随机交叉实网运行并完整记录链路。每次新运行至少保存源/候选/权重哈希、配置与软件版本、窗口决策、session/message ID、接收manifest、重建参数、逐帧GT评估输入及摘要。

## 8. 现在能够写出的结论

可以写：系统已贯通候选编码—语义录取—NB-IoT ACK接收—视频重建；七序列离线结果显示多质量候选和历史状态复用提高模型可用性；04/09实网与一组双节点原型实现了接收后任务评估。

当前证据不足以写：DCVC单独造成全部实网收益、动态ACK预算已在同轨迹下证明独立收益、七序列ELIC完整率失真曲线已评估、人工GT检测/跟踪精度已提升、health_weighted优于equal_share、可推广到四节点且公平/低能耗。

本次根据已存逐序列数据重新聚合，未新增编码或无线运行。保留三种消融的逐序列原始数据，只提供七序列等权 macro。目录与导出说明见 [README.md](README.md)，指标定义见 [统一指标协议](metric_protocol.md)。

## 9. 分别补什么数据和实验

| 类别 | 已有证据 | 需要补的数据 | 下一项实验及控制条件 |
|---|---|---|---|
| 消融：proxy | 七序列 TTL 模型；04/09 同接收载荷 no-proxy 重建 | 七序列真实应用 current/proxy/missing、ROI age、错位/过期误差、逐帧质量及全源 GT | 同一接收载荷 TTL=0/3/6/10/20，接收端实际重建和任务评估；比较状态可用与视觉收益 |
| 消融：多质量 | 七序列最高质量单候选对照 | 每 ROI 候选大小、选中档位、局部质量、缺失原因、实际录取成本 | 同候选/窗口/服务轨迹比较多质量与单质量，并匹配总传输成本；补率失真曲线 |
| 消融：预算填充 | 仅有 budget_fill 可选实现 | 七序列完整归档、预算闲置、质量降档、最差 track 覆盖、背景年龄 | 开/关填充，其他参数不变；验证增加 ROI 是否挤压质量或背景刷新 |
| 消融：其他机制 | 在线 ACK、bundle、背景刷新与连续性加分实现 | 因果逐轮 ACK、quota、队列、重传、完整消息、实际背景更新时间 | 先做 ACK 控制响应/异常恢复；再单独关闭 bundle、连续性加分和可选背景，保留必要依赖；使用共同服务轨迹，不复用已删除方案 |
| baseline：编码质量 | 七序列候选；DCVC 完整 PSNR | ELIC 尚缺 4,780 帧质量；DCVC 完整 MS-SSIM；实际应用 BPP、分层字节、版本哈希 | 七序列 5,316 帧完整同源评估；多个编码工作点、同 application BPP 下比较全帧/ROI 质量 |
| baseline：因素隔离 | NAVIS/ELIC 的04/09单次实网 | 同窗口、服务/损失轨迹、完整接收与发送成本、关闭proxy时的质量与GT | DCVC/ELIC × FIFO/语义调度，先关闭proxy，之后独立打开；分开报告编码器与调度器贡献 |
| baseline：传统与最小表示 | 尚无完整受控 anchor | H.264/H.265 码流、依赖丢失/恢复；metadata-only 和 metadata+background 接收与任务指标 | 同时间或同实际传输字节约束，匹配源帧与服务轨迹，报告失效帧与依赖恢复成本 |
| baseline：实网推广 | 04/09单节点及一组双节点 | 七序列或明确子集、至少3次配对重复、链路日志、GT、能耗和端到端时延 | 随机交叉方法顺序；equal_share/health_weighted 双节点配对、节点交换，之后四节点扩展 |

优先顺序：完整质量与字节审计 → 同服务轨迹的编码器×调度器比较及接收端消融 → 人工 GT 与实网重复 → 多节点公平性、恢复和能耗。未覆盖七序列的实验保留逐序列结果，macro 标 NA。

核心调度算法、合理性与优化步骤见 [调度算法说明](../../docs/scheduler_algorithm_and_plan.md)。
