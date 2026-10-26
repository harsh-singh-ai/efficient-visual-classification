from pathlib import Path
from PIL import Image
import csv
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]

EXPECTED_COUNTS = {
    "train": 14,
    "val": 3,
    "test": 3,
}


def main():

    print("=" * 70)
    print("FINAL DATASET VERIFICATION")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_files = {}
    verification_rows = []

    total_images = 0
    invalid_images = 0

    # ------------------------------------------------------------
    # Check counts and image readability
    # ------------------------------------------------------------

    for split in ["train", "val", "test"]:

        print()
        print(f"{split.upper()} SET")
        print("-" * 50)

        for class_name in CLASSES:

            class_dir = PROCESSED_DIR / split / class_name
            images = sorted(class_dir.glob("*.jpg"))

            expected = EXPECTED_COUNTS[split]
            actual = len(images)

            status = "PASS" if actual == expected else "FAIL"

            print(
                f"{class_name:<12} "
                f"{actual:>3} / {expected:<3} "
                f"{status}"
            )

            for image_path in images:

                total_images += 1

                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    with Image.open(image_path) as image:
                        width, height = image.size

                    verification_rows.append({
                        "split": split,
                        "class": class_name,
                        "filename": image_path.name,
                        "width": width,
                        "height": height,
                        "status": "valid",
                    })

                    key = image_path.name

                    if key not in all_files:
                        all_files[key] = []

                    all_files[key].append(split)

                except Exception as error:

                    invalid_images += 1

                    verification_rows.append({
                        "split": split,
                        "class": class_name,
                        "filename": image_path.name,
                        "width": "",
                        "height": "",
                        "status": f"invalid: {error}",
                    })

    # ------------------------------------------------------------
    # Check split overlap
    # ------------------------------------------------------------

    duplicate_files = {
        filename: splits
        for filename, splits in all_files.items()
        if len(splits) > 1
    }

    # ------------------------------------------------------------
    # Save verification metrics
    # ------------------------------------------------------------

    metrics_file = METRICS_DIR / "dataset_verification.csv"

    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "split",
            "class",
            "filename",
            "width",
            "height",
            "status",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(verification_rows)

    # ------------------------------------------------------------
    # Create class distribution figure
    # ------------------------------------------------------------

    train_counts = []
    val_counts = []
    test_counts = []

    for class_name in CLASSES:

        train_counts.append(
            len(
                list(
                    (PROCESSED_DIR / "train" / class_name).glob("*.jpg")
                )
            )
        )

        val_counts.append(
            len(
                list(
                    (PROCESSED_DIR / "val" / class_name).glob("*.jpg")
                )
            )
        )

        test_counts.append(
            len(
                list(
                    (PROCESSED_DIR / "test" / class_name).glob("*.jpg")
                )
            )
        )

    x = range(len(CLASSES))

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - 0.2 for i in x],
        train_counts,
        width=0.2,
        label="Train"
    )

    plt.bar(
        x,
        val_counts,
        width=0.2,
        label="Validation"
    )

    plt.bar(
        [i + 0.2 for i in x],
        test_counts,
        width=0.2,
        label="Test"
    )

    plt.xticks(list(x), CLASSES)
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    plt.title("Final Dataset Class Distribution")
    plt.legend()
    plt.tight_layout()

    figure_file = FIGURES_DIR / "dataset_class_distribution.png"
    plt.savefig(figure_file, dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"Total images checked: {total_images}")
    print(f"Invalid images:       {invalid_images}")
    print(f"Split overlap files:  {len(duplicate_files)}")

    if invalid_images == 0:
        print("Image integrity:      PASS")
    else:
        print("Image integrity:      FAIL")

    if len(duplicate_files) == 0:
        print("Data leakage check:   PASS")
    else:
        print("Data leakage check:   FAIL")

    print()
    print(f"Metrics saved to: {metrics_file}")
    print(f"Figure saved to:  {figure_file}")

    print()
    print("=" * 70)

    if (
        total_images == 120
        and invalid_images == 0
        and len(duplicate_files) == 0
    ):
        print("FINAL DATASET STATUS: READY FOR MODEL TRAINING")
    else:
        print("FINAL DATASET STATUS: REVIEW REQUIRED")

    print("=" * 70)


if __name__ == "__main__":
    main()