# 真实 NB-IoT 运行

- `navis/mot17_04|mot17_09/run01/`：NAVIS 单节点，sender/receiver/rebuild，以及同接收载荷禁用 proxy 的 rebuild_no_proxy。
- `elic/mot17_04|mot17_09/run01/`：静态 ELIC 背景与 JPEG ROI 单节点；rebuild 仅应用完整接收 ROI，无历史 ROI proxy。
- `multinode/health_weighted/mot17_04_09/run01/`：双节点实网；receiver/sessions 与 rebuild 按 node 分隔。

sender 历史日志与 config_snapshot 保留原始路径以记录运行环境。当前配置和派生图像/接收索引已更新；不要直接把历史快照中的旧工作区路径作为当前输入。对比汇总在 [baselines](../baselines/)，检测和状态评估在 [evaluation](../evaluation/)。详情见 [总索引](../README.md)。
