# Static ELIC+JPEG MOT17-09 run01

This reconstruction uses only the 394 ACK-confirmed, CRC-verified receiver payloads. It reconstructs all 525 metadata frames with no missing background frame, but applies 274 current JPEG ROIs out of 4439 expected physical ROIs. ELIC has no receiver-side ROI proxy, so its ROI availability and current-ROI coverage are both 6.17%.

The stateful tracking task uses the source metadata tracks as its fixed reference. ELIC reaches task-ready recall 6.17%, mean per-track ready coverage 14.67%, and adjacent-frame task-ready continuity 0.47%. Proxy TTL is zero by design.

The sender selected 274 ROI groups over 118 rounds (2.32 per round) and discarded 52647 of 54884 candidate packets under the static FIFO and 20 kB window policy. ACK coverage is 100% for the selected logical messages. Thus the low coverage is chiefly selection-limited: it does not show that selected logical messages were absent from the receiver. This run acknowledges 2.414 MB application payload in 1598.83 s; it is not yet a paired comparison with a NAVIS MOT17-09 run under the same cellular trace.
