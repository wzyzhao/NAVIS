# 接收后任务评估

`detection/mot17-04/`、`mot17-09/` 用完整收到的 ROI 对应语义位置作为参考，评估重建帧上 YOLO 的检测一致性。`stateful_tracking/` 用源 track ID 与 10 帧 TTL 比较当前 appearance 和历史 proxy 的任务状态覆盖与连续性。

两者均由实网接收/重建产物派生，但计算过程离线。检测指标不等于 MOT17 人工真值准确率；状态连续性不等于独立跟踪器 MOTA/IDF1。

代码在 `experiments/evaluation/`。详细输入与口径见 [结果总索引](../README.md)。
