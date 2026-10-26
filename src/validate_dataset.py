from pathlib import Path
from PIL import Image
import hashlib


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]


def file_hash(path):
    """Return an MD5 hash of the file contents."""
    md5 = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


def validate_images():

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    total_images = 0
    valid_images = 0
    invalid_images = []
    duplicate_images = []

    hashes = {}

    for class_name in CLASSES:

        class_dir = RAW_DIR / class_name

        image_files = sorted(
            class_dir.glob("*.jpg")
        )

        print()
        print(f"Class: {class_name}")
        print(f"Images found: {len(image_files)}")

        for image_path in image_files:

            total_images += 1

            try:

                # Check that the image can be opened.
                with Image.open(image_path) as image:

                    image.verify()

                # Reopen after verify() because verify()
                # closes the image internally.
                with Image.open(image_path) as image:

                    width, height = image.size

                # Reject extremely small images.
                if width < 150 or height < 150:

                    invalid_images.append(
                        (
                            str(image_path),
                            f"Too small: {width}x{height}"
                        )
                    )

                    continue

                # Detect exact duplicate files.
                image_hash = file_hash(image_path)

                if image_hash in hashes:

                    duplicate_images.append(
                        (
                            str(image_path),
                            hashes[image_hash]
                        )
                    )

                else:

                    hashes[image_hash] = str(
                        image_path
                    )

                valid_images += 1

            except Exception as error:

                invalid_images.append(
                    (
                        str(image_path),
                        str(error)
                    )
                )

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total images checked: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {len(invalid_images)}")
    print(f"Exact duplicates: {len(duplicate_images)}")

    if invalid_images:

        print()
        print("INVALID IMAGES:")

        for path, reason in invalid_images:

            print(
                f"- {path} | {reason}"
            )

    if duplicate_images:

        print()
        print("DUPLICATE IMAGES:")

        for duplicate, original in duplicate_images:

            print(
                f"- {duplicate}"
            )

            print(
                f"  Same as: {original}"
            )

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_images()