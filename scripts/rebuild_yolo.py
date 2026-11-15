from pathlib import Path
import sys
import shutil
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path
from src.data.prepare import parse_voc


def main():
    cfg = load_config()

    label_map = cfg["labels"]
    raw_root = resolve_path(cfg["data"]["raw_root"])
    processed = resolve_path(cfg["data"]["processed_root"])

    class_names = list(dict.fromkeys(label_map.values()))
    class_to_id = {name: i for i, name in enumerate(class_names)}

    # Read existing verified classification splits.
    csv_root = processed / "classification"
    splits = {
        "train": pd.read_csv(csv_root / "train.csv"),
        "val": pd.read_csv(csv_root / "val.csv"),
    }

    print("Classification split sizes:")
    for name, df in splits.items():
        print(f"{name}: {len(df)}")

    # Index XML annotations once.
    print("\nIndexing XML annotations...")
    xml_index = {}

    for xml_file in raw_root.rglob("*.xml"):
        if xml_file.stem in xml_index:
            raise RuntimeError(
                f"Duplicate XML stem: {xml_file.stem}"
            )
        xml_index[xml_file.stem] = xml_file

    print(f"Indexed {len(xml_index)} XML files.")

    # Check image and label filename collisions.
    for split_name, df in splits.items():
        paths = [Path(p) for p in df["image"]]
        stems = [p.stem.lower() for p in paths]

        if len(stems) != len(set(stems)):
            raise RuntimeError(
                f"Duplicate image stems in {split_name} split."
            )

        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(p)

            if p.stem not in xml_index:
                raise FileNotFoundError(
                    f"No XML annotation found for {p}"
                )

    # Back up incomplete YOLO directories.
    yolo_root = processed / "yolo"
    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for folder in [
        yolo_root / "images" / "train",
        yolo_root / "images" / "val",
        yolo_root / "labels" / "train",
        yolo_root / "labels" / "val",
    ]:
        if folder.exists():
            backup = folder.with_name(
                folder.name + "_backup_" + backup_suffix
            )
            folder.rename(backup)
            print(f"Backed up {folder}")

        folder.mkdir(parents=True, exist_ok=True)

    # Generate YOLO images and labels.
    for split_name, df in splits.items():
        print(f"\nBuilding {split_name} split...")

        for count, (_, row) in enumerate(df.iterrows(), start=1):
            image_path = Path(row["image"])
            xml_path = xml_index[image_path.stem]

            objects = parse_voc(xml_path, label_map)

            if not objects:
                raise RuntimeError(
                    f"No valid objects found in {xml_path}"
                )

            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

            image_out = (
                yolo_root / "images" / split_name / image_path.name
            )

            label_out = (
                yolo_root / "labels" / split_name
                / f"{image_path.stem}.txt"
            )

            shutil.copy2(image_path, image_out)

            with open(label_out, "w", encoding="utf-8") as f:
                for obj in objects:
                    cls = class_to_id[obj["class_name"]]

                    xmin, xmax = obj["xmin"], obj["xmax"]
                    ymin, ymax = obj["ymin"], obj["ymax"]

                    if not (
                        0 <= xmin < xmax <= width
                        and 0 <= ymin < ymax <= height
                    ):
                        raise ValueError(
                            f"Invalid bounding box in {xml_path}: {obj}"
                        )

                    xc = ((xmin + xmax) / 2) / width
                    yc = ((ymin + ymax) / 2) / height
                    bw = (xmax - xmin) / width
                    bh = (ymax - ymin) / height

                    f.write(
                        f"{cls} {xc:.6f} {yc:.6f} "
                        f"{bw:.6f} {bh:.6f}\n"
                    )

            if count % 100 == 0:
                print(f"{split_name}: {count}/{len(df)}")

        print(f"Completed {split_name}: {len(df)} images.")

    # Write YOLO configuration.
    yaml_text = (
        f"path: {yolo_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        "names: [" + ", ".join(repr(x) for x in class_names) + "]\n"
    )

    (yolo_root / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    print("\nYOLO dataset rebuild completed.")


if __name__ == "__main__":
    main()
