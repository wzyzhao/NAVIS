from collections.abc import Iterator, Sequence
from configparser import ConfigParser
from pathlib import Path

import cv2

from .types import Record


class MOT17Loader:
    """Discover MOT17 sequences and yield decoded frames in frame order."""

    def __init__(
        self, root: str | Path, split: str | None = None, sequences: Sequence[str] | None = None
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sequence_filter = set(sequences) if sequences else None
        self.sequences = self._discover_sequences()

        if not self.sequences:
            raise FileNotFoundError(f"No MOT17 sequences found under {self.root}")

    def _discover_sequences(self) -> list[Record]:
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        if self.root.name.lower() == "img1":
            return [self._build_sequence(self.root.parent)]

        if (self.root / "seqinfo.ini").exists() and (self.root / "img1").exists():
            return [self._build_sequence(self.root)]

        search_root = self.root
        if self.split:
            candidate = self.root / self.split
            if candidate.exists():
                search_root = candidate

        sequence_dirs = []
        for path in sorted(search_root.iterdir()):
            if path.is_dir() and (path / "seqinfo.ini").exists() and (path / "img1").exists():
                sequence_dirs.append(path)

        if sequence_dirs:
            return self._filter_sequences(sequence_dirs)

        if (self.root / "train").exists() or (self.root / "test").exists():
            split_dirs = []
            splits = [self.split] if self.split else ["train", "test"]
            for split_name in splits:
                split_root = self.root / split_name
                if not split_root.exists():
                    continue
                for path in sorted(split_root.iterdir()):
                    if (
                        path.is_dir()
                        and (path / "seqinfo.ini").exists()
                        and (path / "img1").exists()
                    ):
                        split_dirs.append(path)
            return self._filter_sequences(split_dirs)

        raise FileNotFoundError(f"Unable to resolve MOT17 sequences from {self.root}")

    def _filter_sequences(self, sequence_dirs: list[Path]) -> list[Record]:
        if self.sequence_filter is None:
            return [self._build_sequence(path) for path in sequence_dirs]
        filtered = [path for path in sequence_dirs if path.name in self.sequence_filter]
        return [self._build_sequence(path) for path in filtered]

    def _build_sequence(self, sequence_dir: Path) -> Record:
        seqinfo = self._read_seqinfo(sequence_dir / "seqinfo.ini")
        image_dir = sequence_dir / "img1"
        image_ext = seqinfo.get("imExt", ".jpg")
        frame_paths = sorted(image_dir.glob(f"*{image_ext}"))

        return {
            "name": sequence_dir.name,
            "dir": sequence_dir,
            "image_dir": image_dir,
            "frame_paths": frame_paths,
            "seqinfo": seqinfo,
        }

    @staticmethod
    def _read_seqinfo(seqinfo_path: Path) -> Record:
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(seqinfo_path)
        raw = dict(parser["Sequence"]) if parser.has_section("Sequence") else {}

        int_keys = {"frameRate", "seqLength", "imWidth", "imHeight"}
        for key in int_keys:
            if key in raw:
                raw[key] = int(raw[key])
        return raw

    def __iter__(self) -> Iterator[Record]:
        """Yield readable frames with sequence metadata."""
        for sequence_index, sequence in enumerate(self.sequences):
            for frame_offset, frame_path in enumerate(sequence["frame_paths"], start=1):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue

                yield {
                    "frame": frame,
                    "frame_id": frame_offset,
                    "frame_path": str(frame_path),
                    "sequence": sequence["name"],
                    "sequence_frame_paths": sequence["frame_paths"],
                    "sequence_index": sequence_index,
                    "is_new_sequence": frame_offset == 1,
                    "seqinfo": sequence["seqinfo"],
                }
