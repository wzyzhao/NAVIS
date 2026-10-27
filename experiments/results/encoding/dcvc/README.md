# DCVC 编码结果

- `encoded/<MOT17序列>/`：background / required_background / roi 的原始 DCVC bin，以及序列级底层统计 JSON。`source/` 为原始输入 YUV。
- `decoded/<MOT17序列>/`：各层解码 YUV 和该序列的 frame_metrics.json。
- `payloads/<MOT17序列>/`：序列候选 manifest，以及 metadata / background / roi 的待发送 bin。

`payloads/candidate_manifest.json` 为全序列候选索引；根目录 frame_metrics.json、dcvc_layered_metrics.json、required_background_dcvc_output.json 为指标索引。通过 `common.encoding_layout.load_artifact_json()` 可读取索引或普通 JSON。summary.json 与 quantitative_metrics.json 保留跨序列最终统计。

底层工具的逐码流临时 JSON 已删除；最终统计值、原始输入、码流、载荷和解码产物保留。该目录不表示真实无线交付，参见 [结果总索引](../../README.md)。

这里的 decoded 是编码端的离线参考解码，不是无线接收端结果。真实接收/重建内容仍按运行保存到 transmission/.../rebuild。完整结果与实验计划见 [实验总览](../../experiment_summary.md)。
