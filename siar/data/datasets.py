from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unitorchplate.data.datasets import Dataset, Sequence, Item, DatasetWithSelfTarget, DatasetWithGroundTruthTarget, \
    DatasetConfig
from unitorchplate.utils.constants import DATAFILES_DIR


@dataclass
class SiarDatasetConfig(DatasetConfig):
    data_dir: str = DATAFILES_DIR
    transform: Any = None
    target_transform: Any = None
    apply_same_transform: bool = True
    limit: int | None = None
    return_sequence: bool = False
    channel_first: bool = True
    item_data_is_path: bool = True
    gt_target: bool = False
    self_target: bool = True
    items_per_sequence: int | None = 10

    def __post_init__(self):
        assert self.gt_target or self.self_target, "Either gt_target or self_target must be True"
        assert not (self.gt_target and self.self_target), "Only one of gt_target or self_target can be True"

    def instance(self):
        if self.gt_target:
            return SiarWithGroundTruthTarget(self)
        elif self.self_target:
            return SiarWithSelfTarget(self)
        else:
            return Siar(self)


class Siar(Dataset):
    def __init__(
            self,
            config: SiarDatasetConfig,
    ):
        super().__init__(config)

    def _init_data(self) -> tuple[list[Sequence], int]:
        idx = -1
        sequences = []
        items_per_sequence = self.config.items_per_sequence
        for directory in Path(self.config.data_dir).iterdir():
            if self.config.limit is not None and idx + 1 >= self.config.limit:
                break
            if directory.is_dir() and directory.name[0] != ".":
                seq_id = int(directory.name)
                seq = []
                gt = None
                for file in directory.iterdir():
                    if file.is_file():
                        filepath = str(file)
                        if file.name != "gt.png" and file.name != "front.png":
                            file_number_in_seq = int(file.name.split(".")[0])
                            seq.append(Item(file_number_in_seq, seq_id, filepath))
                        if file.name == "gt.png":
                            gt = filepath
                if items_per_sequence is None:
                    items_per_sequence = len(seq)
                else:
                    if items_per_sequence != len(seq):
                        print(f"Skipping sequence {directory.name} has {len(seq)} items, but expected {items_per_sequence}")
                        continue
                idx += 1
                sequences.append(Sequence(idx, seq_id, seq, gt))
            else:
                print(f"Skipping {directory.name}")
        return sequences, items_per_sequence


class SiarWithSelfTarget(Siar, DatasetWithSelfTarget):
    """Returns the image itself as target."""


class SiarWithGroundTruthTarget(Siar, DatasetWithGroundTruthTarget):
    """
    Returns the target image of the corresponding sequence.
    """
