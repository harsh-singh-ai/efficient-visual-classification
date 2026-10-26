from pathlib import Path
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]

NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15

LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# Directories
# ============================================================

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data transformations
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Dataset
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = models.mobilenet_v3_small(
    weights=weights
)

# Freeze pretrained feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace classifier.
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[3].parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training / validation functions
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Training
# ============================================================

print("=" * 70)
print("MOBILENETV3-SMALL TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {CLASSES}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Random seed: {RANDOM_SEED}")

print()
print("Training started...")
print()


history = []

best_val_accuracy = 0.0
best_epoch = 0

best_model_path = MODELS_DIR / "mobilenet_v3_small_best.pth"


for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best validation model.
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "epoch": best_epoch,
            },
            best_model_path
        )


# ============================================================
# Save training metrics
# ============================================================

metrics_file = METRICS_DIR / "training_metrics.csv"

with open(
    metrics_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# Load best model
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Final test evaluation
# ============================================================

test_loss, test_accuracy = evaluate(test_loader)


# ============================================================
# Training curves
# ============================================================

epochs = [
    row["epoch"]
    for row in history
]

train_losses = [
    row["train_loss"]
    for row in history
]

val_losses = [
    row["val_loss"]
    for row in history
]

train_accuracies = [
    row["train_accuracy"]
    for row in history
]

val_accuracies = [
    row["val_accuracy"]
    for row in history
]


plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "training_validation_loss.png",
    dpi=200
)

plt.close()


plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "training_validation_accuracy.png",
    dpi=200
)

plt.close()


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Best epoch:               {best_epoch}")
print(f"Final test loss:          {test_loss:.4f}")
print(f"Final test accuracy:      {test_accuracy:.4f}")

print()
print(f"Best model: {best_model_path}")
print(f"Metrics:    {metrics_file}")

print()
print("Training curves:")
print("  results/figures/training_validation_loss.png")
print("  results/figures/training_validation_accuracy.png")

print("=" * 70)