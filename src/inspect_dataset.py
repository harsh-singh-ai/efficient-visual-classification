from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"

CLASSES = [
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "phone",
]

THUMB_SIZE = 220
COLS = 4
ROWS = 7

FONT_SIZE = 14


def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            FONT_SIZE
        )
    except:
        font = ImageFont.load_default()

    print("=" * 60)
    print("CREATING LABELED DATASET CONTACT SHEETS")
    print("=" * 60)

    for class_name in CLASSES:

        class_dir = (
            PROCESSED_DIR
            / "train"
            / class_name
        )

        images = sorted(
            class_dir.glob("*.jpg")
        )

        print(
            f"{class_name}: {len(images)} images"
        )

        sheet_width = COLS * THUMB_SIZE
        sheet_height = ROWS * (THUMB_SIZE + 30)

        sheet = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for index, image_path in enumerate(images):

            row = index // COLS
            col = index % COLS

            x = col * THUMB_SIZE
            y = row * (THUMB_SIZE + 30)

            try:
                image = Image.open(
                    image_path
                ).convert("RGB")

                image.thumbnail(
                    (THUMB_SIZE - 10, THUMB_SIZE - 10)
                )

                image_x = (
                    x
                    + (THUMB_SIZE - image.width) // 2
                )

                image_y = (
                    y
                    + (THUMB_SIZE - image.height) // 2
                )

                sheet.paste(
                    image,
                    (image_x, image_y)
                )

                draw.text(
                    (x + 5, y + THUMB_SIZE),
                    image_path.name,
                    fill="black",
                    font=font
                )

            except Exception as error:

                draw.text(
                    (x + 5, y + 5),
                    f"ERROR: {image_path.name}",
                    fill="red",
                    font=font
                )

        output_path = (
            FIGURES_DIR
            / f"{class_name}_inspection.jpg"
        )

        sheet.save(
            output_path,
            quality=95
        )

        print(
            f"  Saved: {output_path.name}"
        )

    print()
    print("=" * 60)
    print("CONTACT SHEETS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()