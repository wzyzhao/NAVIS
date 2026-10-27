# 基线与对比结果

`navis_vs_elic/mot17-04/run01/` 是两次历史真实链路运行的汇总，原始数据在 [transmission](../transmission/)。NAVIS/ELIC 的 04/09 单节点数据均保留。链路、预算、编码表示与 proxy 同时不同，不能归因于单个组件。

`experiments.baselines.static_elic_replay` 保留 ELIC FIFO 的单序列离线比较入口，没有独立归档结果，且双方窗口和总预算可能不同；不能称为已完成受控编码器对照。

baseline 优先补完整七序列质量、实际应用字节、同服务轨迹编码器×调度器比较，以及传统编码 anchor 和实网重复。跨序列只使用完整七序列等权 macro，现有实网结果逐序列列出。

具体数据缺口与控制条件见 [实验总览](../experiment_summary.md)。
