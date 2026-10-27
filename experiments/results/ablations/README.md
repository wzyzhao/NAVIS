# 离线消融结果

`scheduler/mot17-XX/` 保存七个 MOT17 训练序列的候选大小重放，每序列有 summary.json 和 report.md。主表变体为 navis_ack、no_proxy、single_q_roi；budget_fill 是尚无完整归档的可选实现。录取完整消息后才计入当前 ROI，10 帧 TTL 模型用于可用 ROI。没有真实 modem、ACK 或交付仿真。

运行单序列：`python -m experiments.ablations.scheduler`。七序列汇总：`python -m experiments.ablations.aggregate_macro`。跨序列只使用七序列等权 macro，缺序列拒绝汇总。

[macro 表格](scheduler/report.md) · [机器可读汇总](scheduler/summary.json) · [统一口径](../metric_protocol.md) · [待补实验](../experiment_summary.md)
