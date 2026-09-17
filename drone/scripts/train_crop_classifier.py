#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "dataset"
)

OUTPUT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "classifier"
)

MODEL_PATH = OUTPUT_DIR / "mobilenetv3_small_best.pt"
META_PATH = OUTPUT_DIR / "metadata.json"

MODEL_NAME = "mobilenetv3_small_100"

EPOCHS = 40
BATCH_SIZE = 32
LR = 3e-4

class FixedClassFolder(Dataset):
    """
    Validation dataset that uses the training class mapping even when
    some classes are absent from the temporal holdout.
    """

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform=None,
    ):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        for class_name, class_index in class_to_idx.items():
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in valid_extensions:
                    continue

                self.samples.append(
                    (
                        path,
                        class_index,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        DATA_ROOT / "train",
        transform=train_transform,
    )

    classes = train_dataset.classes

    val_dataset = FixedClassFolder(
        DATA_ROOT / "val",
        class_to_idx=train_dataset.class_to_idx,
        transform=val_transform,
    )

    print(
        f"Classes: {classes}"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples:   {len(val_dataset)}"
    )

    print(
        f"Device: {device}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=len(classes),
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_accuracy = -1.0
    best_epoch = -1

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                float(loss.item())
                * len(labels)
            )

            train_correct += int(
                (
                    logits.argmax(dim=1)
                    == labels
                )
                .sum()
                .item()
            )

            train_total += len(labels)

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                val_correct += int(
                    (
                        logits.argmax(dim=1)
                        == labels
                    )
                    .sum()
                    .item()
                )

                val_total += len(labels)

        train_accuracy = (
            train_correct
            / train_total
        )

        val_accuracy = (
            val_correct
            / val_total
        )

        train_loss = (
            train_loss_sum
            / train_total
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

    META_PATH.write_text(
        json.dumps(
            {
                "model_name":
                    MODEL_NAME,
                "classes":
                    classes,
                "class_to_idx":
                    train_dataset.class_to_idx,
                "best_epoch":
                    best_epoch,
                "best_val_accuracy":
                    best_accuracy,
            },
            indent=2,
        )
    )

    print()
    print("=" * 72)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best val accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Saved: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
