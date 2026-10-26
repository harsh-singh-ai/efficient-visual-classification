import csv
import io
import time
from pathlib import Path

import requests
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
METADATA_FILE = BASE_DIR / "data" / "image_metadata.csv"

IMAGES_PER_CLASS = 100

CLASSES = {
    "bottle": "bottle",
    "cup": "cup",
    "book": "book",
    "mouse": "computer mouse",
    "keyboard": "computer keyboard",
    "phone": "mobile phone",
}

API_URL = "https://commons.wikimedia.org/w/api.php"

ALLOWED_LICENSES = {
    "CC0",
    "Public domain",
    "CC BY",
    "CC BY-SA",
    "CC BY 2.0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 2.0",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
}

HEADERS = {
    "User-Agent": (
        "EfficientVisualClassification/1.0 "
        "(University of Aizu student project)"
    )
}


def search_commons(search_term, limit=200):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 500),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 640,
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    return list(pages.values())


def get_metadata(info):
    metadata = info.get("extmetadata", {})

    def value(key):
        item = metadata.get(key, {})
        return item.get("value", "")

    license_name = value("LicenseShortName")
    artist = value("Artist")
    description = value("ImageDescription")
    categories = value("Categories")

    return {
        "license": license_name,
        "artist": artist,
        "description": description,
        "categories": categories,
    }


def license_is_allowed(license_name):
    if not license_name:
        return False

    license_name = license_name.strip()

    if license_name in ALLOWED_LICENSES:
        return True

    upper = license_name.upper()

    if "CC BY 4.0" in upper:
        return True

    if "CC BY-SA 4.0" in upper:
        return True

    if "CC0" in upper:
        return True

    if "PUBLIC DOMAIN" in upper:
        return True

    return False


def download_image(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content))

    image.load()

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def save_image(image, path):
    image.save(
        path,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def main():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    total_downloaded = 0

    for class_name, search_term in CLASSES.items():

        class_dir = RAW_DIR / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = list(
            class_dir.glob("*.jpg")
        )

        print()
        print("=" * 60)
        print(f"Class: {class_name}")
        print(f"Existing images: {len(existing)}")
        print("=" * 60)

        if len(existing) >= IMAGES_PER_CLASS:

            print(
                "Enough images already exist. Skipping."
            )

            continue

        needed = (
            IMAGES_PER_CLASS
            - len(existing)
        )

        pages = search_commons(
            search_term,
            limit=300,
        )

        print(
            f"Search results: {len(pages)}"
        )

        downloaded = 0

        seen_titles = set()

        for page in pages:

            if downloaded >= needed:
                break

            title = page.get(
                "title",
                "",
            )

            if title in seen_titles:
                continue

            seen_titles.add(title)

            imageinfo = page.get(
                "imageinfo",
                [],
            )

            if not imageinfo:
                continue

            info = imageinfo[0]

            mime = info.get(
                "mime",
                "",
            )

            if not mime.startswith(
                "image/"
            ):
                continue

            metadata = get_metadata(info)

            license_name = metadata[
                "license"
            ]

            if not license_is_allowed(
                license_name
            ):
                continue

            image_url = (
                info.get("thumburl")
                or info.get("url")
            )

            if not image_url:
                continue

            filename = (
                f"{class_name}_"
                f"{len(existing) + downloaded + 1:04d}.jpg"
            )

            output_path = (
                class_dir / filename
            )

            try:

                image = download_image(
                    image_url
                )

                width, height = image.size

                if width < 150 or height < 150:
                    continue

                save_image(
                    image,
                    output_path,
                )

                metadata_rows.append({
                    "filename": str(
                        output_path.relative_to(
                            BASE_DIR
                        )
                    ),
                    "class": class_name,
                    "source": "Wikimedia Commons",
                    "source_title": title,
                    "source_url": (
                        "https://commons.wikimedia.org/wiki/"
                        + title.replace(" ", "_")
                    ),
                    "author": metadata[
                        "artist"
                    ],
                    "license": license_name,
                    "width": width,
                    "height": height,
                    "search_term": search_term,
                })

                downloaded += 1

                total_downloaded += 1

                print(
                    f"[{downloaded:03d}/{needed:03d}] "
                    f"{filename} | "
                    f"{license_name}"
                )

                time.sleep(0.2)

            except Exception as error:

                print(
                    f"Skipped {title}: {error}"
                )

        print(
            f"Finished {class_name}: "
            f"{downloaded} new images"
        )

    fieldnames = [
        "filename",
        "class",
        "source",
        "source_title",
        "source_url",
        "author",
        "license",
        "width",
        "height",
        "search_term",
    ]

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    print()
    print("=" * 60)
    print("DATASET DOWNLOAD COMPLETE")
    print("=" * 60)
    print(
        f"New images downloaded: "
        f"{total_downloaded}"
    )
    print(
        f"Metadata file: "
        f"{METADATA_FILE}"
    )
    print()


if __name__ == "__main__":
    main()