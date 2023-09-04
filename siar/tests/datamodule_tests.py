import os
import unittest
from functools import partial
from multiprocessing import Pool

import torch
from lightning import seed_everything
from p_tqdm import p_map
from tqdm import tqdm
import numpy as np

from siar.data.datamodules import SiarDataModule, SiarDataModuleConfig
from siar.data.datasets import SiarDatasetConfig


def func(train_sample, siar_val, siar_test):
    x_train, y_train = train_sample
    siar_val_iter = iter(siar_val)
    siar_test_iter = iter(siar_test)
    val_dups_x = 0
    val_dups_y = 0
    for j in range(len(siar_val)):
        x_val, y_val = next(siar_val_iter)
        if torch.equal(x_train, x_val):
            val_dups_x += 1
        if torch.equal(y_train, y_val):
            val_dups_y += 1
    test_dups_x = 0
    test_dups_y = 0
    for k in range(len(siar_test)):
        x_test, y_test = next(siar_test_iter)
        if torch.equal(x_train, x_test):
            test_dups_x += 1
        if torch.equal(y_train, y_test):
            test_dups_y += 1
    assert val_dups_x + val_dups_y + test_dups_x + test_dups_y == 0
    return (val_dups_x, val_dups_y), (test_dups_x, test_dups_y)


class DataModuleTestCase(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

        self.siar_dm = SiarDataModule(
            SiarDataModuleConfig(
                dataset_config=SiarDatasetConfig(return_sequence=False),
                shuffle_train=True,
                batch_size=1,
                num_workers=0
            )
        )
        self.siar_dm.setup("placeholder")

    def test_determinism(self):
        x = []
        y = []
        seed_everything(42)
        siar_train = iter(self.siar_dm.train_dataloader())
        for i in range(10):
            x1, y1 = next(siar_train)
            x.append(x1)
            y.append(y1)
        # imitate restarting training
        seed_everything(42)
        siar_train2 = iter(self.siar_dm.train_dataloader())
        for x1, y1 in zip(x, y):
            x2, y2 = next(siar_train2)
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)

    def test_disjoint_split(self):
        siar = SiarDataModule(
            SiarDataModuleConfig(
                dataset_config=SiarDatasetConfig(return_sequence=False, gt_target=True, self_target=False, limit=500),
                shuffle_train=True,
                batch_size=1,
                num_workers=0,
            )
        )
        siar.setup("placeholder")

        siar_train = siar.train_dataloader()
        siar_train_iter = iter(siar_train)
        siar_val = siar.val_dataloader()
        siar_test = siar.test_dataloader()

        MULTIPROCESSING = True
        if not MULTIPROCESSING:
            for i in tqdm(range(len(siar_train))):
                func(next(siar_train_iter), siar_val, siar_test)
        else:
            with Pool(os.cpu_count()) as pool:
                try:
                    results = p_map(partial(func, siar_val=siar_val, siar_test=siar_test), siar_train_iter)
                    assert np.sum(results) == 0
                finally:
                    pool.terminate()
                    pool.join()

    def test_equal_val_order(self):
        siar_val = self.siar_dm.val_dataloader()
        siar_val_iter = iter(siar_val)
        siar_val_iter2 = iter(siar_val)
        for i in range(len(siar_val)):
            x_val, y_val = next(siar_val_iter)
            x_val2, y_val2 = next(siar_val_iter2)
            assert torch.equal(x_val, x_val2)
            assert torch.equal(y_val, y_val2)

    def test_seq_datamodule(self):
        siar = SiarDataModule(
            SiarDataModuleConfig(
                dataset_config=SiarDatasetConfig(return_sequence=True, gt_target=True, self_target=False),
                shuffle_train=False,
                batch_size=1,
                num_workers=0,
            )
        )
        siar.prepare_data()
        siar.setup("placeholder")
        assert len(siar.train_dataset) + len(siar.val_dataset) + len(siar.test_dataset) == len(siar.full_dataset)


if __name__ == '__main__':
    unittest.main()
