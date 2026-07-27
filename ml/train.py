import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from ml.dataset import TransitDataset
from ml.model import TransitFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the transit-fusion classifier.")
    parser.add_argument("--data", type=Path, default=None, help="Optional directory of NPZ samples")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TransitDataset(args.data, samples=args.samples, seed=args.seed)
    validation_size = max(1, round(len(dataset) * 0.15))
    training, validation = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        "train": DataLoader(
            training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            validation,
            batch_size=args.batch_size,
            num_workers=args.workers,
        ),
    }

    model = TransitFusionNet().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )
    criterion = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_metrics = run_epoch(model, loaders["validation"], criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_loss": validation_metrics["loss"],
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")
            export_torchscript(model, args.output / "transit-fusion.ts", device)


def run_epoch(
    model: TransitFusionNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_positive = false_positive = false_negative = correct = count = 0

    for batch in loader:
        phase_flux = batch["phase_flux"].to(device)
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(phase_flux, features)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

        predictions = torch.sigmoid(logits) >= 0.5
        truth = labels >= 0.5
        total_loss += float(loss.detach()) * labels.size(0)
        true_positive += int((predictions & truth).sum())
        false_positive += int((predictions & ~truth).sum())
        false_negative += int((~predictions & truth).sum())
        correct += int((predictions == truth).sum())
        count += labels.size(0)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / max(count, 1),
        "accuracy": correct / max(count, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def export_torchscript(model: TransitFusionNet, path: Path, device: torch.device) -> None:
    model.eval()
    example_flux = torch.zeros(1, 1, 256, device=device)
    example_features = torch.zeros(1, 8, device=device)
    traced = torch.jit.trace(model, (example_flux, example_features))
    torch.jit.save(traced, path)


if __name__ == "__main__":
    train(parse_args())
