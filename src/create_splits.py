from pathlib import Path
import random
import shutil
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SPLIT_METADATA_FILE = (
    BASE_DIR / "data" / "split_metadata.csv"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]

IMAGES_PER_CLASS = 38

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# FUNCTIONS
# ============================================================

def create_directories():

    for split in ["train", "val", "test"]:

        for class_name in CLASSES:

            directory = (
                PROCESSED_DIR
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


def get_images(class_name):

    class_dir = RAW_DIR / class_name

    images = sorted(
        class_dir.glob("*.jpg")
    )

    if len(images) < IMAGES_PER_CLASS:

        raise ValueError(
            f"Class '{class_name}' has only "
            f"{len(images)} images. "
            f"Need at least "
            f"{IMAGES_PER_CLASS}."
        )

    return images


def split_images(images):

    # Make a copy so the original list
    # is not modified.
    images = list(images)

    random.shuffle(images)

    # Select exactly 38 images.
    selected = images[
        :IMAGES_PER_CLASS
    ]

    # Calculate split sizes.
    train_count = round(
        IMAGES_PER_CLASS * TRAIN_RATIO
    )

    val_count = round(
        IMAGES_PER_CLASS * VAL_RATIO
    )

    # Test gets whatever remains.
    test_count = (
        IMAGES_PER_CLASS
        - train_count
        - val_count
    )

    train_images = selected[
        :train_count
    ]

    val_images = selected[
        train_count:
        train_count + val_count
    ]

    test_images = selected[
        train_count + val_count:
    ]

    return (
        train_images,
        val_images,
        test_images
    )


def copy_images(
    images,
    split,
    class_name,
    metadata_rows
):

    destination_dir = (
        PROCESSED_DIR
        / split
        / class_name
    )

    for image_path in images:

        destination_path = (
            destination_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_path
        )

        metadata_rows.append({
            "filename": str(
                destination_path.relative_to(
                    BASE_DIR
                )
            ),
            "class": class_name,
            "split": split,
            "source_file": str(
                image_path.relative_to(
                    BASE_DIR
                )
            ),
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING DATASET SPLITS")
    print("=" * 60)

    # Make the experiment reproducible.
    random.seed(RANDOM_SEED)

    create_directories()

    metadata_rows = []

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASSES:

        print()
        print("-" * 60)
        print(f"Processing class: {class_name}")
        print("-" * 60)

        images = get_images(
            class_name
        )

        print(
            f"Available images: "
            f"{len(images)}"
        )

        train_images, val_images, test_images = (
            split_images(images)
        )

        print(
            f"Selected: "
            f"{len(train_images) + len(val_images) + len(test_images)}"
        )

        print(
            f"Train: {len(train_images)}"
        )

        print(
            f"Validation: {len(val_images)}"
        )

        print(
            f"Test: {len(test_images)}"
        )

        copy_images(
            train_images,
            "train",
            class_name,
            metadata_rows
        )

        copy_images(
            val_images,
            "val",
            class_name,
            metadata_rows
        )

        copy_images(
            test_images,
            "test",
            class_name,
            metadata_rows
        )

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

    # ========================================================
    # SAVE SPLIT METADATA
    # ========================================================

    with open(
        SPLIT_METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "filename",
            "class",
            "split",
            "source_file",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"Training images:   {total_train}"
    )

    print(
        f"Validation images: {total_val}"
    )

    print(
        f"Test images:       {total_test}"
    )

    print(
        f"Total images:      "
        f"{total_train + total_val + total_test}"
    )

    print()
    print(
        f"Split metadata: "
        f"{SPLIT_METADATA_FILE}"
    )

    print()
    print("Random seed:", RANDOM_SEED)

    print("=" * 60)


if __name__ == "__main__":
    main()