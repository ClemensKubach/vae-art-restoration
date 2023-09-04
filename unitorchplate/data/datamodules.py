import logging
import random
from dataclasses import dataclass

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from unitorchplate.data.datasets import DatasetConfig, Dataset


logger = logging.getLogger(__name__)


def random_split(input_list, lengths: list[int], shuffle=True):
    full_list = input_list.copy()
    if shuffle:
        random.shuffle(full_list)
    exclusive_parts = []
    start_idx = 0
    for length in lengths:
        exclusive_parts.append(full_list[start_idx:start_idx+length])
        start_idx += length
    return exclusive_parts


@dataclass
class DataModuleConfig:
    """Configuration for the DataModule, especially with Lightning parameters."""
    dataset_config: DatasetConfig
    batch_size: int
    shuffle_train: bool
    train_size: float
    val_size: float
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int

    def __post_init__(self):
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 2 if self.num_workers > 0 else None

    def instance(self):
        """Create an instance of the DataModule from the current config."""
        return DataModule(self)


class DataModule(LightningDataModule):
    """LightningDataModule standardizes the training, val, test splits, data preparation and transforms. The main advantage is
    consistent data splits, data preparation and transforms across models. (see lightning docs).
    This implementation is optimized for using the Dataset class and its subclasses for sequential datasets as well as single input images.
    """

    def __init__(
            self,
            config: DataModuleConfig
    ):
        super().__init__()
        assert config.train_size + config.val_size < 1, "train_size + val_size must be smaller than 1"

        self.config = config

        self.full_dataset = self.config.dataset_config.instance()
        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None
        self.test_dataset: Subset | None = None

    def prepare_data(self):
        self.full_dataset.prepare()

    def setup(self, stage):
        items_per_sequence = self.config.dataset_config.items_per_sequence
        sequence_indices = self.full_dataset.sequence_indices
        num_sequences = len(sequence_indices)

        num_train = int(num_sequences * self.config.train_size)
        num_val = int(num_sequences * self.config.val_size)
        num_test = num_sequences - num_train - num_val

        seq_indices_train, seq_indices_val, seq_indices_test = random_split(sequence_indices, [
            num_train,
            num_val,
            num_test
        ], shuffle=True)
        if not self.config.dataset_config.return_sequence:
            sequence_exclusive_items = lambda sequence_indices, items_per_sequence: [sidx * items_per_sequence + iidx for iidx in range(items_per_sequence) for sidx in sequence_indices]
            seq_indices_train = sequence_exclusive_items(seq_indices_train, items_per_sequence)
            seq_indices_val = sequence_exclusive_items(seq_indices_val, items_per_sequence)
            seq_indices_test = sequence_exclusive_items(seq_indices_test, items_per_sequence)

        self.train_dataset = Subset(self.full_dataset, seq_indices_train)
        self.val_dataset = Subset(self.full_dataset, seq_indices_val)
        self.test_dataset = Subset(self.full_dataset, seq_indices_test)

        if num_train == 0:
            logger.warning("No training set available.")
        if num_val == 0:
            logger.warning("No validation set available.")
        if num_test == 0:
            logger.warning("No test set available.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.shuffle_train,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )