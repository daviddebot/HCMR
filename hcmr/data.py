"""MNIST-Addition data and concept labels."""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTAddition(Dataset):
    """Pairs MNIST digits and predicts their sum through digit concepts.

    Each example contains two images, 20 one-hot digit concepts, and 19 one-hot
    sum nodes.  The complete 39-node vector is used as the hierarchical target.
    """

    def __init__(self, root: str | Path, train: bool, limit: int | None = None):
        dataset = MNIST(root=str(root), train=train, download=True, transform=ToTensor())
        images, digits = dataset.data.float().div(255), dataset.targets
        if limit is not None:
            images, digits = images[:limit], digits[:limit]
        if len(images) < 2:
            raise ValueError("MNISTAddition requires at least two MNIST images.")

        # Pair two disjoint halves, matching the protocol used by the original
        # research code while keeping the dataset generation transparent.
        half = len(images) // 2
        self.images = torch.stack((images[:half], images[half : 2 * half]), dim=1).unsqueeze(2)
        left, right = digits[:half], digits[half : 2 * half]
        digit_concepts = torch.cat((F.one_hot(left, 10), F.one_hot(right, 10)), dim=1).float()
        sums = F.one_hot(left + right, 19).float()
        self.nodes = torch.cat((digit_concepts, sums), dim=1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.nodes[index]


def mnist_addition_names() -> list[str]:
    return [f"left_{digit}" for digit in range(10)] + [f"right_{digit}" for digit in range(10)] + [f"sum_{total}" for total in range(19)]

