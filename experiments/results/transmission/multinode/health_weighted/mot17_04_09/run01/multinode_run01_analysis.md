# Multi-node real-network run: MOT17-04 and MOT17-09

## Scope

This report analyzes the completed TCP/NB-IoT run in which `node_1` sent
MOT17-04-FRCNN and `node_2` sent MOT17-09-FRCNN.  Results are based on the
sender delivery log, per-node receiver manifests, reconstructed videos, and
cloud tasks below.

## Transport and scheduling

| Metric | node_1 (MOT17-04) | node_2 (MOT17-09) | Run total |
|---|---:|---:|---:|
| Sequence frames | 1,050 | 525 | 1,575 |
| Active scheduling rounds | 372 | 197 | 372 |
| Selected payload bytes | 1,978,990 | 3,278,234 | 5,257,224 |
| Mean selected bytes per active round | 5,320 | 16,641 | 14,133 |
| Receiver reassembled payloads | 4,603 | 2,445 | 7,048 |
| Receiver current ROI payloads | 4,227 | 2,246 | 6,473 |
| Application-ACK-complete messages | 4,604 / 4,604 | 2,435 / 2,446 | 7,039 / 7,050 |

The global online controller used 372 windows, with a 6,000--35,000 B quota
(mean 20,608 B), application-ACK goodput of 5.23 kbps, and a 13.15 s p95 ACK
RTT.

`node_2` completed after round 197.  In rounds 198--372, `node_1` alone
received all 680,921 B selected for that phase; `node_2` received 0 B.  The
mean nominal global quota in this phase was 17,232 B, while `node_1` selected
only 3,891 B/window on average.  Therefore the scheduler did reclaim
`node_2`'s share, but the unused part of each later window was caused by
limited admissible demand, not by a quota reservation for the completed node.

The receiver manifest contains all reassembled data artifacts used below.
The sender feedback nevertheless records 11 non-complete `node_2` messages
(at frames 124--126) and 978 module-unconfirmed messages.  Since the receiver
has complete reassembled payload artifacts while these sender-side events are
present, module confirmation must be reported separately from application ACK
and receiver artifacts.  It is not valid to infer task loss from module
confirmation alone.

## Reconstruction

Each node was rebuilt from its isolated receiver session with the NAVIS
composition settings and a six-frame visual ROI cache.  Neither rebuild had a
decode error, black output frame, or missing metadata frame.

| Metric | node_1 (MOT17-04) | node_2 (MOT17-09) |
|---|---:|---:|
| Reconstructed frames | 1,050 / 1,050 | 525 / 525 |
| Background keyframes decoded | 4 | 2 |
| Current ROI applications | 4,227 / 4,343 | 2,246 / 4,439 |
| Visual cache proxy applications | 69 | 824 |
| Source ROIs without current/proxy appearance | 47 | 1,369 |
| Decode errors | 0 | 0 |

The two videos are `rebuild/node_1/reconstructed_videos/node_1.mp4` and
`rebuild/node_2/reconstructed_videos/node_2.mp4`.

## Cloud tasks

### YOLOv8s detection on reconstruction

YOLOv8s ran on all reconstructed frames at confidence 0.25 and IoU 0.5.  The
reference comprises only complete ROI payloads in each receiver manifest,
localized with the source semantic metadata.  Precision, recall, and F1 are
therefore detector-consistency metrics for delivered semantic content, not
MOTChallenge detection accuracy.

| Metric | node_1 (MOT17-04) | node_2 (MOT17-09) |
|---|---:|---:|
| Reference received ROI instances | 4,227 | 2,246 |
| True positive / false positive / false negative | 3,066 / 611 / 1,161 | 1,867 / 648 / 379 |
| Precision | 83.38% | 74.23% |
| Recall | 72.53% | 83.13% |
| F1 | 77.58% | 78.43% |

### Stateful cloud tracking

This task uses source track IDs and receiver-visible decoded ROI layers.  A
track is task-ready when its current ROI arrives or a prior current ROI for the
same track is within a 10-frame TTL.  It measures the availability of task
state; it is not an image detector benchmark.

| Metric | node_1 (MOT17-04) | node_2 (MOT17-09) |
|---|---:|---:|
| Current-only recall | 97.33% | 50.60% |
| TTL task-ready recall | 99.01% | 72.67% |
| TTL recovery gain | 1.68 pp | 22.08 pp |
| Task-ready continuity | 99.21% | 74.34% |

`node_1` has near-complete current ROI coverage.  `node_2` has substantially
lower current coverage because its sequence admitted fewer ROI groups under
the observed service, but the 10-frame state cache recovers 22.08 percentage
points of task-ready recall.  This is evidence for receiver-state continuity,
not a claim that cached pixels replace fresh visual observations.

## Artifacts

- `sender/delivery_log.json` and `sender/rounds.jsonl`: transport and quota decisions.
- `receiver/sessions/node_1/` and `receiver/sessions/node_2/`: ACKs, manifests, and received payloads.
- `rebuild/node_1/reconstruction_quality_report.json` and `rebuild/node_2/reconstruction_quality_report.json`: rebuild checks.
- `rebuild/node_*/cloud_yolo_task.json`: YOLO results.
- `rebuild/node_*/stateful_tracking_task.json`: TTL tracking results.
