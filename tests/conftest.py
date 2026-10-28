"""Small deterministic CPU fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seeded_rng():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
