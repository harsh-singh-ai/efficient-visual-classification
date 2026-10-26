from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "data" / "processed" / "test"

MODEL_PATH = (
    BASE_DIR
    / "results"
    / "models"
    / "mobilenet_v3_small_finetuned_best.pth"
)

METRICS_DIR = BASE_DIR / "results" / "metrics"
FIGURES_DIR = BASE_DIR / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINE-TUNED MODEL TEST EVALUATION")
print("=" * 70)


# ============================================================
# TEST DATA
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = test_dataset.classes


print(f"Device:       {device}")
print(f"Test images:  {len(test_dataset)}")
print(f"Classes:      {class_names}")


# ============================================================
# LOAD MODEL
# ============================================================

model = models.mobilenet_v3_small(
    weights=None
)


num_features = model.classifier[-1].in_features


model.classifier[-1] = nn.Linear(
    num_features,
    len(class_names)
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)


# The training script saved a checkpoint dictionary.
# We only need the actual model parameters.

model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


print()
print("Model loaded successfully.")

if "epoch" in checkpoint:
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")

if "best_val_accuracy" in checkpoint:
    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(report).transpose()


report_file = (
    METRICS_DIR
    / "finetuned_classification_report.csv"
)


report_df.to_csv(
    report_file
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)


cm_file = (
    METRICS_DIR
    / "finetuned_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)


plt.title(
    "Fine-Tuned MobileNetV3-Small Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


plt.tight_layout()


figure_file = (
    FIGURES_DIR
    / "finetuned_confusion_matrix.png"
)


plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL TEST ACCURACY
# ============================================================

correct_predictions = sum(
    true == pred
    for true, pred in zip(
        all_labels,
        all_predictions
    )
)


accuracy = (
    correct_predictions
    / len(all_labels)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Test accuracy: {accuracy:.4f}"
)

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    f"Classification report:"
)

print(
    report_file
)

print()

print(
    f"Confusion matrix:"
)

print(
    cm_file
)

print()

print(
    f"Confusion matrix figure:"
)

print(
    figure_file
)

print()
print("=" * 70)