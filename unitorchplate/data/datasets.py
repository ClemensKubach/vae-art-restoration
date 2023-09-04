from abc import ABC, abstractmethod
from typing import Any, Tuple
from dataclasses import dataclass

import torch
import torch.utils.data as data
import torchvision.io as io
import torchvision.transforms.functional as functional

from unitorchplate.utils.constants import DATAFILES_DIR


@dataclass
class DatasetConfig(ABC):
    """Configuration for the Dataset class."""
    data_dir: str = DATAFILES_DIR
    transform: Any = None
    target_transform: Any = None
    apply_same_transform: bool = True
    limit: int | None = None
    return_sequence: bool = False
    channel_first: bool = True
    item_data_is_path: bool = True
    items_per_sequence: int | None = None

    @abstractmethod
    def instance(self):
        """Create an instance of the Dataset from the current config."""
        return Dataset(self)


@dataclass
class Item:
    """An item as part of the internally used Sequence class.
    The data can be i.e. a path to an image or the image itself.
    """
    id: int
    sequence_id: int
    data: Any


@dataclass
class Sequence:
    """A sequence of items."""
    idx: int
    id: int
    seq: list[Item]
    gt: Any


class Dataset(data.Dataset, ABC):
    """The Dataset class can be used to load data from a directory and return it as a torch Dataset.
    It is optimized for sequential data, but can also be used for single images.
    A lot of work is already handled here. Thus, only the _init_data method has to be implemented in subclasses.
    """

    def __init__(
            self,
            config: DatasetConfig,
    ):
        self.config = config
        self.sequences, self.items_per_sequence = self._init_data()
        self._idx_sequence_map = {seq.idx: seq for seq in self.sequences}
        self._id_sequence_map = {seq.id: seq for seq in self.sequences}

    def get_sequence_object(self, i: int, by_idx: bool = True) -> Sequence:
        """Returns the sequence object by idx or id.
        If by_idx is True, i is the index of the sequence in the dataset.
        If by_idx is False, i is the id of the sequence.
        """
        if by_idx:
            return self._idx_sequence_map[i]
        else:
            return self._id_sequence_map[i]

    def get_item_object(self, idx: int | None = None, sequence_id: int | None = None, item_id: int | None = None) -> Item | None:
        """Experimental feature.

        There are two modes how to get an item object:
        1. By idx: The idx is the index of the item in the dataset, same is used in __getitem__ call. Only allowed if return_sequence is False.
        2. By sequence_id and item_id: The sequence_id is the id of the sequence the item belongs to.
        """
        if idx is not None:
            assert not self.config.return_sequence, "Getting an item object by idx is only allowed if return_sequence is False."
            seq_idx = idx // self.items_per_sequence
            item_idx = idx % self.items_per_sequence
            seq = self._idx_sequence_map[seq_idx]
            return seq.seq[item_idx]
        else:
            assert sequence_id is not None and item_id is not None, "Either idx or sequence_id and item_id must be given."
            seq = self._id_sequence_map[sequence_id]
            for item in seq.seq:
                if item.id == item_id:
                    return item
            return None

    @property
    def sequence_indices(self) -> list[int]:
        return list(self._idx_sequence_map.keys())

    @abstractmethod
    def _init_data(self) -> tuple[list[Sequence], int]:
        """Here, exploring the use case specific dataset files is required.
        It has to return a list of sequences and the number of items per sequence.
        This is independent of the return_sequence config parameter.
        If there is no sequential data, it is recommended to assign each item to an own sequence.
        """
        pass

    def prepare(self):
        """Can be used to prepare the data before training, like loading into memory."""
        pass

    def __len__(self) -> int:
        if self.config.return_sequence:
            return len(self.sequences)
        else:
            return len(self.sequences) * self.items_per_sequence

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a tensor either of shape (C, ...), (..., C), (T, C, ...), or (T, ..., C).
        """
        if self.config.return_sequence:
            sequence = self._idx_sequence_map[idx]
            if self.config.item_data_is_path:
                data_instances = []
                for item in sequence.seq:
                    data_instances.append(self.load_image(item.data))
                    if self.config.transform:
                        data_instances = [
                            self.config.transform(instance) for instance in data_instances
                        ]
            else:
                data_instances = [item.data for item in sequence.seq]
            return torch.stack(data_instances)
        else:
            seq_idx = idx // self.items_per_sequence
            item_idx = idx % self.items_per_sequence
            sequence = self._idx_sequence_map[seq_idx]
            data = self.load_image(sequence.seq[item_idx].data)
            if self.config.transform:
                data = self.config.transform(data)
            return data

    def load_image(self, filepath: str) -> torch.Tensor:
        """Loads an image from a filepath and returns it as a torch tensor."""
        img = functional.convert_image_dtype(io.read_image(filepath), dtype=torch.float)  # [0:3]
        if not self.config.channel_first:
            img = img.permute(2, 1, 0)
        return img


class DatasetWithSelfTarget(Dataset, ABC):
    """Returns the data itself as target, like (x, x).
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = super().__getitem__(idx)
        return data, data


class DatasetWithGroundTruthTarget(Dataset, ABC):
    """
    Returns the ground truth data of the corresponding sequence, like (x, y).
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = super().__getitem__(idx)
        if self.config.return_sequence:
            target = self.load_image(self._idx_sequence_map[idx].gt)
            if self.config.target_transform:
                target = self.config.target_transform(target)
            return data, target
        else:
            seq_idx = idx // self.items_per_sequence
            target = self.load_image(self._idx_sequence_map[seq_idx].gt)
            if self.config.target_transform:
                target = self.config.target_transform(target)
            return data, target
