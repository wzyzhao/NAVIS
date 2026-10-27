# NAVIS 与 Static ELIC+JPEG：MOT17-04 run01 结果对比

## 重建结论

两种方法都从各自接收端的 ACK 确认、CRC 校验通过的载荷重建了 1,050 帧，且均无背景缺失。ELIC 重建只使用实际收到的 ELIC 背景和 JPEG ROI，不使用发送端文件或 ROI 代理。

| 指标 | NAVIS | Static ELIC+JPEG |
| --- | ---: | ---: |
| 重建帧数 | 1,050 | 1,050 |
| 预期 ROI | 4,343 | 4,343 |
| 当前收到并实际应用的 ROI | 3,790 (87.27%) | 820 (18.88%) |
| 元数据 ROI 代理 | 275 | 0 |
| 最终应用 ROI | 4,065 (93.60%) | 820 (18.88%) |
| 未应用 ROI | 278 | 3,523 |
| 背景缺失帧 | 0 | 0 |

NAVIS 的实际收到 ROI 数量是 ELIC 的 4.62 倍；纳入接收端元数据代理后的可应用 ROI 数量是 4.96 倍。ELIC 的静态 FIFO 调度未提供缺失 ROI 的接收端代理，因此其 ROI 覆盖仅来自实际到达的 820 个 JPEG ROI。

## 已确认传输统计

| 指标 | NAVIS | Static ELIC+JPEG |
| --- | ---: | ---: |
| ACK 确认载荷 | 4,005 | 1,034 |
| ACK 确认应用载荷字节 | 1,997,208 | 4,480,338 |
| ACK 确认线上字节 | 2,011,568 | 4,606,146 |
| 发送耗时（秒） | 2,404.86 | 3,282.15 |
| ACK goodput（bps） | 6,655.51 | 10,689.31 |
| p95 RTT（ms） | 11,901.04 | 5,702.38 |
| 调度轮数 | 210 | 210 |

ELIC 在这次历史运行中确认传输了 NAVIS 的 2.24 倍应用载荷，耗时也多 36.5%，但实际到达的 ROI 更少。这说明在这两份归档的实际链路记录中，NAVIS 更有效地把有限载荷用于 ROI 覆盖；不能把 goodput 或 RTT 的差异归因于编码方法本身。

## Baseline 定位与解释边界

本组结果可作为论文的端到端 baseline：NAVIS 的 ACK 动态预算与 DCVC 表示联合设计，对照固定 20,000-byte 每轮预算、静态 ELIC+JPEG ROI 的基线。两侧均处理同一 MOT17-04 序列、1,050 帧和 210 个窗口，并以接收端 ACK 确认载荷重建。

结果支持“联合设计相对于静态 ELIC+JPEG baseline 的端到端收益”。它不能把差异进一步拆分为仅由动态预算或仅由 DCVC 压缩造成；该拆分需要额外的消融实验。NAVIS 启用了 ACK 驱动的预算调整（首轮后下一轮预算为 19,245 bytes，末轮实际预算为 17,773 bytes），ELIC 则固定为每轮 20,000 bytes；ACK goodput 与 RTT 作为实际链路条件一并报告。

## 证据文件

- NAVIS 传输：`experiments/results/transmission/navis/mot17_04/run01/sender/delivery_log.json`
- NAVIS 重建：`experiments/results/transmission/navis/mot17_04/run01/rebuild/reconstruction_quality_report.json`
- ELIC 传输：`experiments/results/transmission/elic/mot17_04/run01/sender/delivery_log.json`
- ELIC 重建：`experiments/results/transmission/elic/mot17_04/run01/rebuild/summary.json`
