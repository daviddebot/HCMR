#!/usr/bin/env python3
"""Train H-CMR on MNIST-Addition."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hcmr.data import MNISTAddition, mnist_addition_names
from hcmr.model import HCMR


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, nodes in loader:
            output = model(images.to(device))
            correct += ((output["node_prob"] > .5) == nodes.to(device).bool()).sum().item()
            total += nodes.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/mnist-addition")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-rules", type=int, default=10)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    train = MNISTAddition(args.data_dir, train=True, limit=args.train_limit)
    test = MNISTAddition(args.data_dir, train=False, limit=args.test_limit)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = HCMR(n_nodes=39, n_rules=args.n_rules).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for images, nodes in train_loader:
            images, nodes = images.to(device), nodes.to(device)
            output = model(images)
            # Direct concept supervision and hierarchical reconstruction.
            loss = F.binary_cross_entropy(output["base_prob"], nodes) + F.binary_cross_entropy(output["node_prob"], nodes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(images)
        score = evaluate(model, test_loader, device)
        print(f"epoch={epoch:03d} train_loss={loss_total / len(train):.4f} test_node_accuracy={score:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "names": mnist_addition_names(), "args": vars(args)}, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w") as file:
        json.dump({"test_node_accuracy": evaluate(model, test_loader, device)}, file, indent=2)


if __name__ == "__main__":
    main()

