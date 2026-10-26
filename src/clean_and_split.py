from pathlib import Path
import random
import shutil
import csv

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

IMAGE_METADATA_FILE = BASE_DIR / "data" / "image_metadata.csv"
SPLIT_METADATA_FILE = BASE_DIR / "data" / "split_metadata.csv"
CLEANING_LOG_FILE = BASE_DIR / "data" / "cleaning_log.csv"

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]

# Clearly unsuitable images identified during visual inspection.
EXCLUDED_IMAGES = {
    "book": {
        "book_0019.jpg",
        "book_0021.jpg",
        "book_0022.jpg",
        "book_0032.jpg",
        "book_0047.jpg",
    },

    "bottle": {
        "bottle_0001.jpg",
        "bottle_0002.jpg",
        "bottle_0014.jpg",
        "bottle_0016.jpg",
        "bottle_0018.jpg",
        "bottle_0034.jpg",
        "bottle_0039.jpg",
    },

    "cup": {
        "cup_0011.jpg",
        "cup_0022.jpg",
        "cup_0027.jpg",
        "cup_0033.jpg",
        "cup_0036.jpg",
        "cup_0039.jpg",
        "cup_0040.jpg",
        "cup_0046.jpg",
    },

    "keyboard": {
        "keyboard_0002.jpg",
        "keyboard_0005.jpg",
        "keyboard_0032.jpg",
        "keyboard_0034.jpg",
        "keyboard_0035.jpg",
        "keyboard_0036.jpg",
        "keyboard_0040.jpg",
        "keyboard_0041.jpg",
        "keyboard_0042.jpg",
        "keyboard_0043.jpg",
    },

    "mouse": {
        "mouse_0005.jpg",
        "mouse_0006.jpg",
        "mouse_0007.jpg",
        "mouse_0019.jpg",
        "mouse_0027.jpg",
    },

    "phone": {
        "phone_0005.jpg",
        "phone_0006.jpg",
        "phone_0008.jpg",
        "phone_0009.jpg",
        "phone_0011.jpg",
        "phone_0015.jpg",
        "phone_0017.jpg",
        "phone_0019.jpg",
        "phone_0020.jpg",
        "phone_0021.jpg",
        "phone_0023.jpg",
        "phone_0024.jpg",
        "phone_0025.jpg",
        "phone_0028.jpg",
        "phone_0035.jpg",
    },
}

IMAGES_PER_CLASS = 20

TRAIN_COUNT = 14
VAL_COUNT = 3
TEST_COUNT = 3

RANDOM_SEED = 42


def clear_processed_directory():
    """Remove previous processed train/val/test data."""
    if PROCESSED_DIR.exists():
        for split in ["train", "val", "test"]:
            split_dir = PROCESSED_DIR / split

            if split_dir.exists():
                shutil.rmtree(split_dir)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            (PROCESSED_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


def load_image_metadata():
    """Load original Wikimedia metadata."""
    metadata = {}

    if not IMAGE_METADATA_FILE.exists():
        return metadata

    with open(
        IMAGE_METADATA_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            key = (row["class"], row["filename"].split("/")[-1])
            metadata[key] = row

    return metadata


def get_clean_images(class_name):
    """Return images that were not manually excluded."""

    class_dir = RAW_DIR / class_name

    all_images = sorted(class_dir.glob("*.jpg"))

    excluded = EXCLUDED_IMAGES.get(class_name, set())

    clean_images = [
        image
        for image in all_images
        if image.name not in excluded
    ]

    return all_images, clean_images


def main():

    print("=" * 70)
    print("CLEANING AND CREATING FINAL DATASET")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    clear_processed_directory()

    original_metadata = load_image_metadata()

    split_rows = []
    cleaning_rows = []

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASSES:

        print()
        print("-" * 70)
        print(f"Class: {class_name}")
        print("-" * 70)

        all_images, clean_images = get_clean_images(class_name)

        excluded = EXCLUDED_IMAGES.get(class_name, set())

        print(f"Original images: {len(all_images)}")
        print(f"Excluded images: {len(excluded)}")
        print(f"Clean candidates: {len(clean_images)}")

        if len(clean_images) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Not enough clean images for class '{class_name}'. "
                f"Need {IMAGES_PER_CLASS}, found {len(clean_images)}."
            )

        # Randomly select exactly 20 clean images.
        selected_images = clean_images.copy()
        random.shuffle(selected_images)
        selected_images = selected_images[:IMAGES_PER_CLASS]

        # Create balanced split.
        train_images = selected_images[:TRAIN_COUNT]
        val_images = selected_images[
            TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
        ]
        test_images = selected_images[
            TRAIN_COUNT + VAL_COUNT:
        ]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        selected_names = {
            image.name
            for image in selected_images
        }

        # Record cleaning decisions.
        for image in all_images:

            if image.name in excluded:
                status = "excluded"
                reason = "Failed visual quality/class relevance inspection"

            elif image.name in selected_names:
                status = "selected"
                reason = "Passed visual inspection and selected for final dataset"

            else:
                status = "not_selected"
                reason = "Clean candidate but not selected after balancing"

            cleaning_rows.append({
                "class": class_name,
                "filename": image.name,
                "status": status,
                "reason": reason,
            })

        # Copy selected images into train/val/test.
        for split_name, images in splits.items():

            for image in images:

                destination = (
                    PROCESSED_DIR
                    / split_name
                    / class_name
                    / image.name
                )

                shutil.copy2(image, destination)

                metadata_key = (class_name, image.name)
                original = original_metadata.get(metadata_key, {})

                split_rows.append({
                    "filename": str(
                        destination.relative_to(BASE_DIR)
                    ),
                    "class": class_name,
                    "split": split_name,
                    "source_file": str(
                        image.relative_to(BASE_DIR)
                    ),
                    "source": original.get("source", ""),
                    "source_title": original.get("source_title", ""),
                    "source_url": original.get("source_url", ""),
                    "author": original.get("author", ""),
                    "license": original.get("license", ""),
                    "random_seed": RANDOM_SEED,
                })

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

        print(f"Final selected: {len(selected_images)}")
        print(f"Train: {len(train_images)}")
        print(f"Validation: {len(val_images)}")
        print(f"Test: {len(test_images)}")

    # Save split metadata.
    with open(
        SPLIT_METADATA_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "filename",
            "class",
            "split",
            "source_file",
            "source",
            "source_title",
            "source_url",
            "author",
            "license",
            "random_seed",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(split_rows)

    # Save cleaning log.
    with open(
        CLEANING_LOG_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "class",
            "filename",
            "status",
            "reason",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(cleaning_rows)

    print()
    print("=" * 70)
    print("FINAL DATASET CREATED")
    print("=" * 70)

    print(f"Training images:   {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Test images:       {total_test}")
    print(f"Total images:      {total_train + total_val + total_test}")

    print()
    print("Class distribution:")
    print("  6 classes × 20 images = 120 images")

    print()
    print(f"Split metadata: {SPLIT_METADATA_FILE}")
    print(f"Cleaning log:  {CLEANING_LOG_FILE}")
    print()
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()