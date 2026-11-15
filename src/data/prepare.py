
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import json
import shutil
import random
import pandas as pd
from PIL import Image


def parse_voc(xml_path, label_map):
    root = ET.parse(xml_path).getroot()
    objects = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()

        if name not in label_map:
            continue

        b = obj.find("bndbox")
        if b is None:
            continue

        try:
            xmin = float(b.findtext("xmin"))
            ymin = float(b.findtext("ymin"))
            xmax = float(b.findtext("xmax"))
            ymax = float(b.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        objects.append({
            "raw": name,
            "class_name": label_map[name],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return objects


def prepare_dataset(
    raw_root,
    out_root,
    label_map,
    seed=42,
    val_size=0.15,
    test_size=0.15
):
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("Invalid validation/test split proportions.")

    # Collect unique image paths.
    image_files = sorted({
        p.resolve()
        for p in raw_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    })

    print(f"Found {len(image_files)} unique image files.")

    # Build XML annotation index ONCE.
    xml_index = {}

    for xml_file in raw_root.rglob("*"):
        if xml_file.is_file() and xml_file.suffix.lower() == ".xml":
            stem = xml_file.stem

            if stem in xml_index:
                raise RuntimeError(
                    f"Duplicate XML filename detected: {stem}\n"
                    f"{xml_index[stem]}\n{xml_file}"
                )

            xml_index[stem] = xml_file

    print(f"Indexed {len(xml_index)} XML annotation files.")

    records = []
    yolo_records = []

    class_names = list(dict.fromkeys(label_map.values()))
    class_to_id = {
        name: i for i, name in enumerate(class_names)
    }

    # Process images and their annotations.
    for count, image_path in enumerate(image_files, start=1):

        # Fast dictionary lookup instead of recursive search.
        xml_path = xml_index.get(image_path.stem)

        if xml_path is None:
            continue

        objects = parse_voc(xml_path, label_map)

        if not objects:
            continue

        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            continue

        # Dominant class based on total bounding-box area.
        areas = {}

        for obj in objects:
            area = (
                max(0, obj["xmax"] - obj["xmin"])
                * max(0, obj["ymax"] - obj["ymin"])
            )

            areas[obj["class_name"]] = (
                areas.get(obj["class_name"], 0) + area
            )

        dominant = max(areas, key=areas.get)

        records.append({
            "image": str(image_path),
            "label": class_to_id[dominant],
            "class_name": dominant,
            "width": w,
            "height": h,
            "num_objects": len(objects),
        })

        yolo_records.append((image_path, objects, w, h))

        if count % 1000 == 0:
            print(f"Processed {count}/{len(image_files)} images.")

    if not records:
        raise RuntimeError(
            f"No annotated images found under {raw_root}."
        )

    df = pd.DataFrame(records)

    if not df["image"].is_unique:
        raise RuntimeError("Duplicate image paths detected.")

    print(f"Prepared {len(df)} annotated unique images.")

    # Split unique images.
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)

    n = len(indices)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    # Output directories.
    processed = out_root

    (processed / "classification").mkdir(
        parents=True, exist_ok=True
    )

    # Safely back up old YOLO split directories.
    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    yolo_folders = [
        processed / "yolo/images/train",
        processed / "yolo/images/val",
        processed / "yolo/labels/train",
        processed / "yolo/labels/val",
    ]

    for folder in yolo_folders:
        if folder.exists():
            backup_folder = folder.with_name(
                f"{folder.name}_backup_{backup_suffix}"
            )

            folder.rename(backup_folder)
            print(f"Backed up: {folder}")

        folder.mkdir(parents=True, exist_ok=True)

    # Save classification CSVs.
    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    for name, idxs in split_map.items():
        df.loc[sorted(idxs)].to_csv(
            processed / "classification" / f"{name}.csv",
            index=False
        )

    # Verify split isolation.
    train_paths = set(df.loc[train_idx, "image"])
    val_paths = set(df.loc[val_idx, "image"])
    test_paths = set(df.loc[test_idx, "image"])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # Index YOLO records.
    yolo_by_path = {
        str(p.resolve()): (p, objs, w, h)
        for p, objs, w, h in yolo_records
    }

    # Check for duplicate filenames within each YOLO split.
    for split_name, idxs in [
        ("train", train_idx),
        ("val", val_idx)
    ]:
        filenames = [
            yolo_by_path[df.loc[idx, "image"]][0].name.lower()
            for idx in idxs
        ]

        if len(filenames) != len(set(filenames)):
            raise RuntimeError(
                f"Duplicate image filenames detected in YOLO {split_name} split."
            )

    # Generate YOLO train and validation datasets.
    for split_name, idxs in [
        ("train", train_idx),
        ("val", val_idx)
    ]:

        for idx in idxs:
            row = df.loc[idx]
            src, objs, w, h = yolo_by_path[row["image"]]

            dst_img = (
                processed / "yolo" / "images"
                / split_name / src.name
            )

            shutil.copy2(src, dst_img)

            label_path = (
                processed / "yolo" / "labels"
                / split_name / f"{src.stem}.txt"
            )

            with open(label_path, "w", encoding="utf-8") as f:
                for obj in objs:
                    cls = class_to_id[obj["class_name"]]

                    xc = ((obj["xmin"] + obj["xmax"]) / 2) / w
                    yc = ((obj["ymin"] + obj["ymax"]) / 2) / h
                    bw = (obj["xmax"] - obj["xmin"]) / w
                    bh = (obj["ymax"] - obj["ymin"]) / h

                    f.write(
                        f"{cls} {xc:.6f} {yc:.6f} "
                        f"{bw:.6f} {bh:.6f}\n"
                    )

    # Generate YOLO configuration.
    (processed / "yolo" / "data.yaml").write_text(
        "path: "
        + str((processed / "yolo").resolve()).replace("\\", "/")
        + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        "names: [" + ", ".join(repr(x) for x in class_names) + "]\n",
        encoding="utf-8"
    )

    # Save class mapping and manifest.
    (processed / "classes.json").write_text(
        json.dumps(class_to_id, indent=2),
        encoding="utf-8"
    )

    df.to_csv(processed / "manifest.csv", index=False)

    print("\nDataset preparation completed.")
    print("\nClass distribution:")
    print(df["class_name"].value_counts())
    print("\nSplit sizes:")
    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")

    return df
