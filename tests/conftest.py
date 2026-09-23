import torch
import pytest
from cesc.config import ModelConfig
@pytest.fixture(autouse=True)
def deterministic_fixture():
    torch.set_num_threads(2);torch.manual_seed(42)
@pytest.fixture
def cfg():return ModelConfig(stats_width=4,energy_width=4,plain_width=6,feature_channels=2)
