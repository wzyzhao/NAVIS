# 实验结果目录与口径

当前实验结论及后续计划见 [实验总览](experiment_summary.md)。消融取舍、七序列等权 macro、同源质量与载荷 BPP 标准见 [统一指标协议](metric_protocol.md)。

本目录按“编码候选 → 调度实验 / 真实传输 → 接收后任务评估”分层。`encoding/elic` 的文件是发送前的候选；`transmission/elic` 是真实 NB-IoT 接收与重建结果，两者不能混用。

```text
experiments/
├─ ablations/、baselines/、evaluation/     实验和评估代码
└─ results/
   ├─ encoding/
   │  ├─ elic/
   │  │  ├─ encoded/MOT17-XX-FRCNN/<帧>/   ELIC/JPEG 本地 pickle
   │  │  ├─ decoded/MOT17-XX-FRCNN/         离线参考指标、frames/、视频（若有）
   │  │  └─ payloads/MOT17-XX-FRCNN/run01/  candidate_manifest.json 与 NEJ1 bin
   │  ├─ dcvc/
   │  │  ├─ encoded/MOT17-XX-FRCNN/         background/、required_background/、roi/ 码流
   │  │  │                                  source/ 为原始输入 YUV；统计 JSON 按序列存放
   │  │  ├─ decoded/MOT17-XX-FRCNN/         离线参考解码 YUV、逐帧质量指标
   │  │  └─ payloads/MOT17-XX-FRCNN/        candidate_manifest.json、metadata/、background/、roi/
   │  └─ metrics/                          跨方法指标汇总
   ├─ ablations/scheduler/mot17-XX/        离线调度消融
   ├─ baselines/                          ELIC baseline 与历史实网比较
   ├─ transmission/{navis,elic,multinode}/ 真实链路 sender/receiver/rebuild
   ├─ evaluation/{detection,stateful_tracking}/ 任务评估
   └─ experiment_summary.md             当前实验总览、证据边界与下一步实验
```

实网的单次运行继续按 `sender/`（决策、发送与 ACK 日志）、`receiver/`（接收 manifest、ACK、reassembled_payloads）、`rebuild/`（解码、图像、视频、重建统计）分层。NAVIS 另有 `rebuild_no_proxy/`；多节点的 receiver/rebuild 内按 node 分层。`runNN` 表示独立运行，编码 run01 与传输 run01 不表示同一阶段或同一会话。

| 结果目录 | 做什么实验 | 代码 / 证据 | 解读边界 |
|---|---|---|---|
| `ablations/scheduler/` | MOT17-02、04、05、09、10、11、13，比较完整离线策略、禁用 proxy、单质量 ROI；跨序列只取七序列等权 macro | `experiments/ablations/scheduler.py`；各序列 `summary.json`、`report.md` | 按候选字节大小离线录取，6–40 kB 元数据压力预算，默认 10 帧 TTL；没有 modem、真实 ACK、重传。`navis_ack` 是历史变体名，不代表真实 ACK 轨迹。 |
| `baselines/navis_vs_elic/` | MOT17-04 两次真实链路 run01 的接收覆盖、确认字节、goodput、RTT 对比 | `report.md` 和汇总 JSON，来源是 `transmission/navis`、`transmission/elic` | 端到端联合设计对比；链路和确认字节不同，不能单独归因于编码器或调度器。这个目录是汇总，不是另一次发送实验。 |
| `evaluation/detection/` | 在 MOT17-04/09 重建帧上运行 YOLOv8s，以完整接收 ROI 对应的源语义位置作参考，按类别和 IoU 匹配，报告 precision/recall/F1 | `experiments/evaluation/detection.py`；各序列 `summary.json` | 衡量已交付内容的检测一致性，不是 MOT17 人工真值准确率，也不能表示所有源目标的检测召回率。 |
| `evaluation/stateful_tracking/` | 用源 track ID 和物理 ROI，比当前收到的 appearance 与 10 帧 TTL 历史 appearance 的状态恢复、任务覆盖、连续性 | `experiments/evaluation/stateful_tracking.py`；各序列 `summary.json` | 确定性 track/TTL 模型，不是独立 tracker 的 MOTA/IDF1；此任务 TTL 与视频视觉 cache 的口径分开解释。 |
| `transmission/elic/`（原 `results/elic/`） | MOT17-04/09 静态 ELIC 背景 + JPEG ROI 的真实 NB-IoT 单节点发送、ACK 接收、重建 | `src/elic_jpg_compression/baseline.py`、`src/nbiot_transport/elic.json`；sender/receiver/rebuild | static FIFO，20,000 B 初始/固定窗口预算，关闭预算自适应和 ROI 历史代理；接收重建只用完整 ACK、CRC 通过的载荷。历史配置以各运行快照为准。 |

MOT17-04 消融中的动态预算变体当前 ROI 为 3553/4343，可用 ROI 为 4173/4343；实网 NAVIS 当前/最终应用 ROI 为 3790/4343、4065/4343，ELIC 当前 ROI 为 820/4343。它们来自不同实验，不能替换引用。stateful_tracking 的 TTL 任务模型算得 proxy 298，也不同于实网视频重建的 proxy 275。

## ELIC payloads 怎样导出

1. 场景分解输出在 `src/scene_factorization/scene_factorization_output/<sequence>/<frame>/`，含背景、ROI、掩码和 `metadata.json`。
2. `src/elic_jpg_compression/main.py` 调用 `elic_jpg_codec/runner.py`：ELIC 压缩背景，`serialize_rois()` 用 JPEG（当前静态配置质量 85）压缩 ROI 图像、PNG 无损保存 ROI mask，并写入 `encoded/<sequence>/<frame>/background.pkl`、`foreground.pkl`。完整参考模式会解码、合成帧、记录 PSNR/MS-SSIM；`candidate_export_only` 模式只生成候选需要的背景和 ROI 中间文件，质量指标为空。
3. `baseline.export(settings)` 读取配置 `candidate_source_manifest`、`candidate_source_sequence`、`elic_encoded_root`、`static_roi_jpeg_quality`。当前源是 DCVC 的 `payloads/<sequence>/candidate_manifest.json`（全序列入口是 `payloads/candidate_manifest.json` 索引），源文件 SHA256 写入 ELIC manifest。按源帧逐项对齐，而不是重新任意选 ROI。
4. metadata 沿用源 manifest 的字节文件；背景只在源 manifest 有 `background_keyframe_summary` 的帧导出 ELIC 背景。ROI 按 `candidate_group_id` 去重，用 `(track_id, roi_bbox)` 找对应 JPEG/PNG，每个物理 ROI 只保留一个质量候选。缺少的 ROI 会从场景分解结果重编码并记录 `reencoded_missing_rois`；已有 JPEG 直接复用，因此改变质量配置后应重生成中间编码。
5. `src/nbiot_transport/core/elic_jpeg_payload.py:encode()` 把二进制 blob 转成索引，生成 JSON 元信息，封装为 `NEJ1 + 4字节大端JSON长度 + JSON头 + 顺序二进制blob`。背景 codec 为 `elic_background`，ROI codec 为 `jpeg_roi`；线上不发送 pickle。
6. 写入 `payloads/<sequence>/run01/000001_background.bin`、`000002_roi000.bin` 等。导出 `candidate_manifest.json`，记录相对 payload_path、实际封装字节数、codec、rate_idx=0、quality_rank=0、源哈希和 ROI 数。metadata 不重复写到 ELIC payloads，而是引用 `encoding/dcvc/payloads/<sequence>/metadata/`。
7. 发送阶段校验来源哈希和配置，再由传输模块分片、调度。真正接收成功的内容位于 `transmission/elic/.../receiver/reassembled_payloads/`；它们才用于接收重建。

编码入口是 `python -m elic_jpg_compression.main`。候选导出入口是 `python -m elic_jpg_compression.baseline`，执行前在 `src/nbiot_transport/elic.json` 设置 `baseline_action` 为 `export`，以及序列、编码根目录和候选 manifest 路径。当前文件的 action 为 `rebuild`，不能将直接运行它当作导出命令。每个序列输出到 `encoding/elic/payloads/MOT17-XX-FRCNN/runNN/`，保留源 manifest 与编码中间文件版本一致。

## 精简后的保存规则

- 每种编码器只有 `encoded/`、`decoded/`、`payloads/` 三个产物目录，第二级用完整 MOT 序列名。原始输入 YUV 保留在 `encoded/<sequence>/source/`，不会混称为解码结果。
- 跨序列的 `summary.json`、DCVC `quantitative_metrics.json` 保留，便于最终报告直接读取。DCVC `payloads/candidate_manifest.json`、根目录 `frame_metrics.json` 和底层编码统计入口是小型 `artifact_index_v1` 索引；实际候选、逐帧指标和编码统计各存一份，按序列分布。
- 用 `common.encoding_layout.load_artifact_json(path)` 读取索引或普通 JSON；索引载荷路径会转成绝对路径。现有调度、传输、ELIC 导出和重建读取入口已支持这种格式，不要直接对索引使用 `json.load()` 后寻找 frames。
- 删除底层逐码流临时 JSON、完全重复的全序列候选 manifest、安装/进度/fallback 日志、已完成运行的空重传队列、上一轮整理的临时备份。最终汇总、逐帧质量数据、候选、完整 ACK 接收记录、发送/调度决策、重建状态、任务指标及配置仍保留。
- DCVC manifest 内容因路径调整、按序列拆分而产生新哈希。ELIC 当前 source_manifest 与源哈希已同步；`source_manifest_provenance_sha256` 保存原全序列来源哈希。历史发送日志/快照中的原路径和原哈希保留作溯源。
- 已有 ELIC 完整解码图像/视频主要来自 MOT17-04/09；其他序列有候选编码和指标文件，不虚构不存在的解码图像。质量记录中的 null 表示未计算，不能当作零分。
- 新实网运行继续使用 `transmission/<method>/<sequence>/runNN/`；新评估保存输入运行路径、模型/TTL 和指标口径；不要覆盖历史运行。

清理脚本是 `scripts/prune_and_reorganize_results.py`，仅用于本次迁移，重复执行会拒绝覆盖。后续编码和候选导出代码已按新结构生成结果。

## 编码端 decoded 与接收端 rebuild

`encoding/<codec>/decoded/` 保存编码端本地解码参考，用于压缩质量和候选生成，不受链路丢失影响。`transmission/<method>/<sequence>/runNN/rebuild/decoded_layers/` 保存实网接收载荷的解码层，frames/reconstructed_videos 保存重建内容；ELIC 接收重建使用 rebuild/frames 与该目录下的视频。接收结果要按运行、节点、配置组织，不能混到 encoding/decoded，否则会混淆离线编码质量与实际交付质量。

DCVC 解码 YUV 是无容器的 YUV420 像素帧，宽高、帧数、QP 要结合文件名和统计 JSON 解释。当前 decoded 内 156,356 个 YUV 共 51,437,112,762 字节，约 47.90 GiB。它们是可再生成的缓存，但当前 export_candidates 和质量复算仍有依赖，暂时保留。只读取现有 manifest/指标、离线调度重放或从实网载荷重建都不需要这些编码端 YUV。若冻结编码结果并接受以后重新解码，再按缓存策略删除；保留所有 bin、manifest、指标和模型版本。详见实验总览中的缓存建议。

