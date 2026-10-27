"""Check indexed artifacts and sequence-specific codec lookups."""

import tempfile
import unittest
from pathlib import Path

from common.encoding_layout import (
    dump_json,
    load_artifact_json,
    metric_stream_directory,
    organize_dcvc_outputs,
    write_candidate_manifest,
    write_frame_index,
    write_metric_index,
)


class EncodingLayoutTests(unittest.TestCase):
    def test_candidate_index_round_trip_and_payload_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for seq in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
                blob = root / "payloads" / seq / "roi" / "frame.bin"
                blob.parent.mkdir(parents=True)
                blob.write_bytes(b"test")
                frames.append({"sequence": seq, "frame_idx": 1, "items": [{"bytes": 4, "payload_path": str(blob)}]})
            index = write_candidate_manifest(root, {"format": "original", "frames": frames})
            self.assertLess(index.stat().st_size, 1000)
            loaded = load_artifact_json(index)
            self.assertEqual(loaded["format"], "original")
            self.assertEqual(len(loaded["frames"]), 2)
            for frame in loaded["frames"]:
                self.assertEqual(Path(frame["items"][0]["payload_path"]).read_bytes(), b"test")

    def test_metrics_and_quality_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "format": "dcvc_layered_metrics_v1",
                "foreground": {
                    "roi": {
                        "MOT17-02-FRCNN_fgroi_test": {"000": {"psnr": 33.125}},
                        "MOT17-04-FRCNN_fgroi_test": {"000": {"psnr": 31.25}},
                    }
                },
            }
            path = root / "metrics.json"
            dump_json(path, values)
            write_metric_index(path, root / "encoded")
            self.assertEqual(load_artifact_json(path), values)
            frames = {
                "frames": [{"sequence": "MOT17-02-FRCNN", "psnr": None}, {"sequence": "MOT17-04-FRCNN", "psnr": 35.01}]
            }
            write_frame_index(root / "frames.json", frames, root / "decoded")
            self.assertEqual(load_artifact_json(root / "frames.json"), frames)

    def test_codec_outputs_preserve_binaries_and_separate_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = root / "foreground_streams/layered_foreground_roi"
            stream.mkdir(parents=True)
            name = "MOT17-04-FRCNN_fgroi_test"
            (stream / (name + ".bin")).write_bytes(b"codec")
            (stream / (name + ".yuv")).write_bytes(b"pixels")
            organize_dcvc_outputs(root)
            self.assertEqual((root / "encoded/MOT17-04-FRCNN/roi" / (name + ".bin")).read_bytes(), b"codec")
            self.assertEqual((root / "decoded/MOT17-04-FRCNN/roi" / (name + ".yuv")).read_bytes(), b"pixels")
            self.assertEqual(metric_stream_directory(root / "encoded", name), root / "encoded/MOT17-04-FRCNN/roi")

    def test_refresh_replaces_previous_codec_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "MOT17-04-FRCNN_fgroi_test.bin"
            stream = root / "foreground_streams/layered_foreground_roi"
            stream.mkdir(parents=True)
            (stream / name).write_bytes(b"new")
            target = root / "encoded/MOT17-04-FRCNN/roi" / name
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                organize_dcvc_outputs(root)
            self.assertEqual(target.read_bytes(), b"old")
            organize_dcvc_outputs(root, replace_existing=True)
            self.assertEqual(target.read_bytes(), b"new")

    def test_index_rejects_sequence_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump_json(root / "wrong.json", {"frames": [{"sequence": "MOT17-04-FRCNN"}]})
            dump_json(
                root / "index.json",
                {
                    "format": "artifact_index_v1",
                    "kind": "frames",
                    "metadata": {},
                    "entries": [{"sequence": "MOT17-02-FRCNN", "path": "wrong.json"}],
                },
            )
            with self.assertRaises(ValueError):
                load_artifact_json(root / "index.json")


if __name__ == "__main__":
    unittest.main()
