from pathlib import Path
import time
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "data" / "processed" / "test"
MODEL_PATH = (
    BASE_DIR
    / "results"
    / "models"
    / "mobilenet_v3_small_finetuned_best.pth"
)

device = torch.device("cpu")

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_RUNS = 50


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = models.mobilenet_v3_small(weights=None)

num_features = model.classifier[-1].in_features

model.classifier[-1] = torch.nn.Linear(
    num_features,
    len(dataset.classes)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ------------------------------------------------------------
# Model statistics
# ------------------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

model_size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)


# ------------------------------------------------------------
# Inference benchmark
# ------------------------------------------------------------

sample_images, _ = next(iter(loader))
sample_images = sample_images.to(device)


with torch.no_grad():

    for _ in range(NUM_WARMUP):
        model(sample_images)

    start_time = time.perf_counter()

    for _ in range(NUM_RUNS):
        model(sample_images)

    end_time = time.perf_counter()


total_time = end_time - start_time

average_time_ms = (
    total_time / NUM_RUNS
) * 1000

images_per_second = (
    1 / (total_time / NUM_RUNS)
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("=" * 70)
print("MODEL EFFICIENCY BENCHMARK")
print("=" * 70)

print(f"Device:                 {device}")
print(f"Model:                  MobileNetV3-Small")
print(f"Test images:            {len(dataset)}")

print()
print("MODEL SIZE")
print("-" * 70)
print(f"Total parameters:       {total_parameters:,}")
print(f"Trainable parameters:   {trainable_parameters:,}")
print(f"Checkpoint size:        {model_size_mb:.2f} MB")

print()
print("CPU INFERENCE")
print("-" * 70)
print(f"Average inference time: {average_time_ms:.2f} ms/image")
print(f"Throughput:             {images_per_second:.2f} images/second")

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)