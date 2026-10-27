# ELIC 编码结果

- `encoded/<MOT17序列>/<帧>/`：背景和 ROI 的本地 pickle。
- `decoded/<MOT17序列>/`：逐帧质量 JSON、`frames/` 图像与 `reconstruction.mp4`（实际存在时）。
- `payloads/<MOT17序列>/run01/`：候选 manifest 和 NEJ1 背景 / JPEG ROI + PNG mask bin。metadata 引用 DCVC 同序列的 metadata 文件。

`summary.json` 是跨序列摘要；`frame_metrics.json` 是小型索引，实际逐帧指标在各序列 decoded 目录。用 `common.encoding_layout.load_artifact_json()` 展开索引。

导出步骤和统计边界见 [结果总索引](../../README.md)。真实接收结果在 [transmission/elic](../../transmission/elic/)。

这里的 decoded 是编码端的离线参考解码，不是无线接收端结果。真实接收/重建内容仍按运行保存到 transmission/.../rebuild。完整结果与实验计划见 [实验总览](../../experiment_summary.md)。
