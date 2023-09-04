import unittest

import torch
from tqdm import tqdm

from siar.data.datamodules import SiarDataModule, SiarDataModuleConfig
from siar.data.datasets import Siar, SiarDatasetConfig, SiarWithGroundTruthTarget, SiarWithSelfTarget


class DatasetTestCase(unittest.TestCase):

    def setUp(self, data_dir=None):
        self.data_dir = data_dir
        self.inputs_per_sequence = 10
        self.max_test_samples = 20
        self.num_tests = lambda ds, max_tests: min(len(ds), max_tests) if max_tests else len(ds)

    def test_dataset_size(self):
        siar = Siar(SiarDatasetConfig(return_sequence=False))
        assert len(siar) == len(siar.sequences) * self.inputs_per_sequence

    def test_single_sequence(self):
        siar = Siar(SiarDatasetConfig(return_sequence=True))
        item = siar[0]
        assert item.shape == (self.inputs_per_sequence, 3, 256, 256)

    def test_single_file(self):
        siar = Siar(SiarDatasetConfig(return_sequence=False))
        item = siar[0]
        assert item.shape == (3, 256, 256)

    def test_single_with_gt(self):
        siar = SiarWithGroundTruthTarget(SiarDatasetConfig(return_sequence=False))
        for i in range(self.num_tests(siar, self.max_test_samples)):
            x, y = siar[i]
            assert x.shape == (3, 256, 256)
            assert y.shape == (3, 256, 256)

    def test_sequence_with_gt(self):
        siar = SiarWithGroundTruthTarget(SiarDatasetConfig(return_sequence=True))
        for i in range(self.num_tests(siar, self.max_test_samples)):
            x, y = siar[i]
            assert x.shape == (self.inputs_per_sequence, 3, 256, 256)
            assert y.shape == (3, 256, 256)

    def test_all_sequences_with_self_target(self):
        siar = SiarWithSelfTarget(SiarDatasetConfig(return_sequence=False))
        for i in tqdm(range(self.num_tests(siar, 10))):
            x, y = siar[i]
            assert torch.equal(x, y)


if __name__ == '__main__':
    unittest.main()
