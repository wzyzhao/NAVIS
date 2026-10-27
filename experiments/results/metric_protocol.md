# 七序列消融与统一指标协议

## 保留的消融与聚合口径

主表保留 navis_ack、no_proxy、single_q_roi。budget_fill 仅保留可选实现，目前没有七序列归档，不能填入已完成结果。

| 变体 | 当前 ROI 覆盖 macro | TTL 可用率 macro | 每序列平均录取 MB | 当前限制 |
|---|---:|---:|---:|---|
| navis_ack | 69.86% | 88.47% | 2.2335 | 历史名称；离线元数据压力预算，没有真实 ACK |
| no_proxy | 69.86% | 69.86% | 2.2335 | 同选择结果、关闭 TTL 模型；不能直接证明视觉质量或人工 GT 收益 |
| single_q_roi | 37.14% | 56.09% | 2.1320 | 仅保留最高质量候选，同预算目标但实际录取字节不同 |

机器可读汇总见 [scheduler/summary.json](ablations/scheduler/summary.json)，表格见 [scheduler/report.md](ablations/scheduler/report.md)。复算入口：`python -m experiments.ablations.aggregate_macro`。

## 跨序列只使用 macro

固定序列集合为 MOT17-02/04/05/09/10/11/13-FRCNN。每个序列权重为 1/7，先算序列指标，再取七个值的算术平均：`macro(m) = sum(m_s) / 7`。

- PSNR_s：该序列全部源帧的 PSNR dB 算术平均；macro PSNR 是七个序列均值的平均。
- MS-SSIM_s：先按完整源帧平均，再按序列等权平均；SSIM 与 MS-SSIM 分开命名。
- BPP_s：`8 × 实际字节 B_s / 完整源像素 P_s`；macro BPP 为七个 BPP_s 的平均。
- 当前 ROI 覆盖：序列内完整录取的物理 ROI 数 / 预期物理 ROI 数，然后取七序列平均。
- TTL 可用率：序列内（当前 ROI + 满足 TTL 的历史 proxy）/ 预期物理 ROI 数，然后取七序列平均。
- 资源效率：先计算各序列可用 ROI/MB，再平均；录取 MB 报每序列平均值。逐序列原始计数保留用于核查，不用合并计数构造跨序列比例。

缺失任一序列或重复序列应报错；缺失任一序列的指标则该 macro 标 NA，不能填零或以六序列均值代替。实网目前只覆盖 04/09，保留逐序列单次结果，不称为七序列 macro。独立重复先在每序列内汇总，再对七序列等权，并保存重复方差与配对差值。

DCVC 七序列全部 5,316 帧有 PSNR，现有 macro 约 30.4574 dB。ELIC 只有 536 帧质量记录，七序列完整 PSNR/MS-SSIM macro 为 NA。DCVC 关闭 calc_ssim，底层零值不能当作实测 MS-SSIM。

## 同源、候选与质量冻结

两方法使用七序列同一批完整 5,316 帧、相同帧号/宽高/颜色空间、场景分解版本、物理 ROI 与 mask。记录源帧清单、模型权重、配置和候选 manifest 的 SHA256，以及编码实现版本。

ELIC 当前 JPEG 质量为 85；固定编码库和参数。改变配置不会改变已存在的 foreground.pkl JPEG，必须重新编码中间文件并导出新版本候选。DCVC QP 与 JPEG quality 数字不能视为同一质量；编码器率失真比较需多个工作点，或同 application BPP 下比较质量。

分别评估：完整编码参考；按实际候选的背景刷新/ROI 工作点无损交付后重建；实网接收后重建。每一种质量必须与同一种表示的成本配对。不能将逐帧完整背景参考的 PSNR 配上稀疏背景候选的载荷 BPP。所有源帧均进入质量/任务评估，缺片时使用明确的重建策略；不能只保留成功解码帧。

## 像素、BPP 与字节边界

所有端到端 BPP 的像素分母固定为 P_s：完整源帧的 W×H 总和，不乘颜色通道数，不使用收到的像素、裁剪/补边面积或 YUV 文件字节。局部 ROI BPP 若另报，必须明确 bbox/mask/padding 分母。1 byte = 8 bit；MB 为 10^6 字节，MiB 为 2^20 字节。

| 指标 | 分子字节范围 | 用途 |
|---|---|---|
| codec_bpp | 实际背景/ROI 压缩码流，含 codec 原生头；不含 pickle、mask 与应用封装 | 纯视觉编码成本 |
| application_bpp | 单一工作点所需唯一逻辑消息的实际序列化字节，含视觉、mask、metadata、NEJ1/其他应用封装 | 系统率失真主口径 |
| delivered_application_bpp | 接收端去重、CRC 通过且完整的逻辑消息字节 | 实际交付成本与质量配对 |
| transmitted_protocol_bpp | 所有实际前向发送尝试的协议字节，含分片/聚合头和重传 | 无线应用协议发送成本，需要完整发送尝试日志 |
| confirmed_protocol_bpp | 历史 confirmed_wire_bytes 对应的确认成功协议字节 | 保留历史口径，不能当作全部发送字节 |

反向 ACK/控制消息单独报告。应用协议字节不等于全部 IP/LTE 空口字节；未测量的底层开销标 NA。

候选库存里的多质量版本不是一次运行全部发送的成本；application_bpp 每个物理 ROI 只选该工作点的一种质量。metadata、共享依赖与背景按实际发送次数计入，不能漏算或重复计入。

采用互斥分解：B_application = B_codec + B_mask + B_semantic_sideinfo + B_serialization。DCVC metadata 可能含 mask，ELIC NEJ1 内含 JPEG、PNG mask 与 JSON 头；解析实际序列化消息后归类，不能在完整 bin 字节上再加 mask。旧数据无法细分时保留总 application 字节，细分项标 NA。协议另分首次发送和重传。

当前 ELIC summary 的本地 pickle 字节与可用重建尺寸口径、DCVC estimated_total_bpp 的逐帧背景/侧信息估算，均不能直接作为上述 application_bpp。后续从实际 payload/metadata 长度重算，并对完全一致的表示补齐逐帧质量。

## 新结果的最小记录

每序列保存 source_frame_count、evaluated_frame_count、source_pixels、physical_roi_count、编码/候选版本与哈希、JPEG 参数或 QP、背景刷新规则、各层实际字节及成本边界、逐帧质量、覆盖与 TTL 定义。跨序列表只保存 per_sequence 与七序列等权 macro。预算/服务轨迹、候选版本或质量评估帧集发生变化时，新建实验版本，不能覆盖旧结果。
