from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "GC10-DET"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
META_DIR = PROCESSED_DIR / "meta"

RANDOM_SEED = 20260430
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
OUTPUT_SIZE = 1024
MIN_RAW_BOX_SIDE = 5.0
MIN_OUTPUT_BOX_SIDE = 2.0
JPEG_QUALITY = 92

CLASS_DEFS = [
    ("1_chongkong", "punching_hole", "冲孔"),
    ("2_hanfeng", "welding_line", "焊缝"),
    ("3_yueyawan", "crescent_gap", "月牙弯"),
    ("4_shuiban", "water_spot", "水斑"),
    ("5_youban", "oil_spot", "油斑"),
    ("6_siban", "silk_spot", "丝斑"),
    ("7_yiwu", "inclusion", "异物"),
    ("8_yahen", "rolled_pit", "压痕"),
    ("9_zhehen", "crease", "折痕"),
    ("10_yaozhe", "waist_folding", "腰折"),
]

RAW_NAME_TO_CLASS_ID = {raw: idx for idx, (raw, _, _) in enumerate(CLASS_DEFS)}
RAW_NAME_TO_CLASS_ID["10_yaozhed"] = RAW_NAME_TO_CLASS_ID["10_yaozhe"]


@dataclass
class RawBox:
    raw_name: str
    class_id: int | None
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    status: str
    reason: str = ""


@dataclass
class Record:
    record_id: str
    source_xml: Path
    image_path: Path
    raw_folder: str
    raw_filename: str
    width: int
    height: int
    raw_boxes: list[RawBox] = field(default_factory=list)
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    status: str = "raw"
    remove_reason: str = ""
    md5: str = ""
    dhash: int = 0
    split: str = ""
    output_stem: str = ""


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR / "images" / "train",
        PROCESSED_DIR / "images" / "val",
        PROCESSED_DIR / "images" / "test",
        PROCESSED_DIR / "labels" / "train",
        PROCESSED_DIR / "labels" / "val",
        PROCESSED_DIR / "labels" / "test",
        META_DIR,
        FIGURES_DIR / "augmentation_examples",
        FIGURES_DIR / "samples",
        FIGURES_DIR / "clean",
        FIGURES_DIR / "square",
        FIGURES_DIR / "chart",
        FIGURES_DIR / "audit",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_outputs() -> None:
    for path in [PROCESSED_DIR, FIGURES_DIR]:
        if path.exists():
            shutil.rmtree(path)
    ensure_dirs()


def require_raw_dataset() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "Missing raw dataset directory: data/raw/GC10-DET. "
            "Place the original GC10-DET files there before running this script."
        )
    if not any((RAW_DIR / str(i)).exists() for i in range(1, 11)) or not (RAW_DIR / "lable").exists():
        raise FileNotFoundError(
            "data/raw/GC10-DET should contain class folders 1..10 and the original lable/ XML folder."
        )


def image_files() -> list[Path]:
    return sorted(
        p
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def xml_files() -> list[Path]:
    return sorted(p for p in RAW_DIR.rglob("*.xml") if p.is_file())


def parse_int(text: str | None, default: int = 0) -> int:
    try:
        return int(float(text or default))
    except ValueError:
        return default


def match_image(root: ET.Element, by_rel: dict[str, Path], by_name: dict[str, list[Path]]) -> tuple[Path | None, str]:
    folder = root.findtext("folder", "").strip()
    filename = root.findtext("filename", "").strip()
    rel = f"{folder}/{filename}".replace("\\", "/")
    if rel in by_rel:
        return by_rel[rel], "exact_folder_filename"
    candidates = by_name.get(filename, [])
    if len(candidates) == 1:
        return candidates[0], "filename_unique_folder_mismatch"
    if len(candidates) > 1:
        for candidate in candidates:
            if candidate.parent.name == folder:
                return candidate, "ambiguous_filename_resolved_by_folder"
        return sorted(candidates)[0], "ambiguous_filename_used_first_sorted"
    return None, "missing_image"


def read_records() -> tuple[list[Record], dict]:
    images = image_files()
    xmls = xml_files()
    by_rel = {p.relative_to(RAW_DIR).as_posix(): p for p in images}
    by_name: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        by_name[p.name].append(p)

    records: list[Record] = []
    raw_class_counter: Counter[str] = Counter()
    match_counter: Counter[str] = Counter()
    xml_parse_errors = []
    for xml_path in xmls:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            xml_parse_errors.append({"xml": relpath(xml_path), "error": str(exc)})
            continue

        image_path, match_status = match_image(root, by_rel, by_name)
        match_counter[match_status] += 1
        folder = root.findtext("folder", "").strip()
        filename = root.findtext("filename", "").strip()
        width = parse_int(root.findtext("size/width"))
        height = parse_int(root.findtext("size/height"))
        record_id = xml_path.stem
        record = Record(
            record_id=record_id,
            source_xml=xml_path,
            image_path=image_path or Path(),
            raw_folder=folder,
            raw_filename=filename,
            width=width,
            height=height,
        )
        if image_path is None:
            record.status = "removed"
            record.remove_reason = "missing_image"
            records.append(record)
            continue

        for obj in root.findall("object"):
            raw_name = (obj.findtext("name", "") or "").strip()
            raw_class_counter[raw_name] += 1
            box = obj.find("bndbox")
            if box is None:
                record.raw_boxes.append(RawBox(raw_name, None, 0, 0, 0, 0, "removed", "missing_bndbox"))
                continue
            xmin = float(parse_int(box.findtext("xmin")))
            ymin = float(parse_int(box.findtext("ymin")))
            xmax = float(parse_int(box.findtext("xmax")))
            ymax = float(parse_int(box.findtext("ymax")))
            class_id = RAW_NAME_TO_CLASS_ID.get(raw_name)
            status = "candidate" if class_id is not None else "removed"
            reason = "" if class_id is not None else "unknown_class"
            record.raw_boxes.append(RawBox(raw_name, class_id, xmin, ymin, xmax, ymax, status, reason))
        records.append(record)

    annotated_by_xml = len([r for r in records if r.image_path])
    raw_summary = {
        "raw_image_files": len(images),
        "raw_xml_files": len(xmls),
        "raw_xml_records_with_image": annotated_by_xml,
        "raw_unannotated_image_files_estimate": max(0, len(images) - annotated_by_xml),
        "raw_object_count": int(sum(raw_class_counter.values())),
        "raw_class_counter": dict(sorted(raw_class_counter.items())),
        "xml_parse_errors": xml_parse_errors,
        "match_counter": dict(match_counter),
        "duplicate_image_basenames": int(sum(1 for paths in by_name.values() if len(paths) > 1)),
    }
    return records, raw_summary


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 16) -> int:
    gray = ImageOps.grayscale(img).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    value = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def validate_boxes(record: Record, stage_counter: Counter[str]) -> None:
    valid_boxes: list[tuple[int, float, float, float, float]] = []
    for raw_box in record.raw_boxes:
        if raw_box.class_id is None:
            stage_counter["boxes_removed_unknown_class"] += 1
            continue
        x1, y1, x2, y2 = raw_box.xmin, raw_box.ymin, raw_box.xmax, raw_box.ymax
        if x1 < 0 or y1 < 0 or x2 > record.width or y2 > record.height:
            raw_box.status = "removed"
            raw_box.reason = "out_of_bounds"
            stage_counter["boxes_removed_out_of_bounds"] += 1
            continue
        if x2 <= x1 or y2 <= y1:
            raw_box.status = "removed"
            raw_box.reason = "zero_or_negative_area"
            stage_counter["boxes_removed_zero_area"] += 1
            continue
        if (x2 - x1) < MIN_RAW_BOX_SIDE or (y2 - y1) < MIN_RAW_BOX_SIDE:
            raw_box.status = "removed"
            raw_box.reason = "too_small"
            stage_counter["boxes_removed_too_small"] += 1
            continue
        raw_box.status = "kept"
        valid_boxes.append((raw_box.class_id, x1, y1, x2, y2))
    record.boxes = valid_boxes


def clean_records(records: list[Record]) -> tuple[list[Record], dict, list[dict]]:
    stage_counter: Counter[str] = Counter()
    removed_rows: list[dict] = []
    opened_records: list[Record] = []
    for record in records:
        if not record.image_path:
            removed_rows.append(removed_row(record, "missing_image"))
            continue
        try:
            with Image.open(record.image_path) as img:
                img.verify()
            with Image.open(record.image_path) as img:
                img = img.convert("RGB")
                record.width, record.height = img.size
                record.md5 = compute_md5(record.image_path)
                record.dhash = compute_dhash(img)
        except Exception as exc:
            record.status = "removed"
            record.remove_reason = "corrupt_or_unreadable_image"
            removed_rows.append(removed_row(record, f"corrupt_or_unreadable_image: {exc}"))
            continue
        opened_records.append(record)

    stage_counter["records_after_corrupt_check"] = len(opened_records)

    exact_seen: dict[str, Record] = {}
    dhash_buckets: list[Record] = []
    deduped: list[Record] = []
    for record in opened_records:
        if record.md5 in exact_seen:
            record.status = "removed"
            record.remove_reason = f"exact_duplicate_of:{exact_seen[record.md5].record_id}"
            removed_rows.append(removed_row(record, record.remove_reason))
            continue
        exact_seen[record.md5] = record

        near_source = None
        for kept in dhash_buckets:
            if hamming(record.dhash, kept.dhash) <= 2:
                near_source = kept
                break
        if near_source is not None:
            record.status = "removed"
            record.remove_reason = f"near_duplicate_of:{near_source.record_id}"
            removed_rows.append(removed_row(record, record.remove_reason))
            continue
        dhash_buckets.append(record)
        deduped.append(record)
    stage_counter["records_removed_exact_duplicate"] = len(opened_records) - len(exact_seen)
    stage_counter["records_after_duplicate_check"] = len(deduped)

    kept: list[Record] = []
    for record in deduped:
        validate_boxes(record, stage_counter)
        if not record.boxes:
            record.status = "removed"
            record.remove_reason = "no_valid_annotation_after_label_filter"
            removed_rows.append(removed_row(record, record.remove_reason))
            continue
        record.status = "kept_clean"
        kept.append(record)

    stage_counter["records_after_annotation_filter"] = len(kept)
    stage_counter["boxes_after_annotation_filter"] = sum(len(r.boxes) for r in kept)
    return kept, dict(stage_counter), removed_rows


def removed_row(record: Record, reason: str) -> dict:
    return {
        "record_id": record.record_id,
        "image_path": relpath(record.image_path) if record.image_path else "",
        "xml_path": relpath(record.source_xml),
        "raw_folder": record.raw_folder,
        "raw_filename": record.raw_filename,
        "reason": reason,
    }


def stratified_split(records: list[Record]) -> None:
    rng = random.Random(RANDOM_SEED)
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[source_key(record.raw_filename)].append(record)

    by_primary: dict[int, list[list[Record]]] = defaultdict(list)
    for items in groups.values():
        class_votes = Counter(box[0] for record in items for box in record.boxes)
        primary = class_votes.most_common(1)[0][0]
        by_primary[primary].append(items)

    for class_id, group_items in by_primary.items():
        rng.shuffle(group_items)
        n = len(group_items)
        train_end = int(round(n * SPLIT_RATIO["train"]))
        val_end = train_end + int(round(n * SPLIT_RATIO["val"]))
        for idx, items in enumerate(group_items):
            if idx < train_end:
                split = "train"
            elif idx < val_end:
                split = "val"
            else:
                split = "test"
            for record in items:
                record.split = split
    rng.shuffle(records)
    for idx, record in enumerate(records, start=1):
        record.output_stem = f"{idx:05d}"


def source_key(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "img":
        return "_".join(parts[:3])
    return stem


def pad_to_square_and_resize(img: Image.Image, boxes: list[tuple[int, float, float, float, float]]) -> tuple[Image.Image, list[tuple[int, float, float, float, float]], dict]:
    w, h = img.size
    side = max(w, h)
    pad_x = (side - w) / 2.0
    pad_y = (side - h) / 2.0
    square = Image.new("RGB", (side, side), (114, 114, 114))
    square.paste(img, (int(round(pad_x)), int(round(pad_y))))
    scale = OUTPUT_SIZE / side
    resized = square.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
    new_boxes = []
    for class_id, x1, y1, x2, y2 in boxes:
        nx1 = (x1 + pad_x) * scale
        ny1 = (y1 + pad_y) * scale
        nx2 = (x2 + pad_x) * scale
        ny2 = (y2 + pad_y) * scale
        new_boxes.append((class_id, nx1, ny1, nx2, ny2))
    meta = {"pad_x": pad_x, "pad_y": pad_y, "side": side, "scale": scale}
    return resized, filter_output_boxes(new_boxes), meta


def filter_output_boxes(boxes: list[tuple[int, float, float, float, float]]) -> list[tuple[int, float, float, float, float]]:
    filtered = []
    for class_id, x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(float(OUTPUT_SIZE), x1))
        y1 = max(0.0, min(float(OUTPUT_SIZE), y1))
        x2 = max(0.0, min(float(OUTPUT_SIZE), x2))
        y2 = max(0.0, min(float(OUTPUT_SIZE), y2))
        if x2 - x1 >= MIN_OUTPUT_BOX_SIDE and y2 - y1 >= MIN_OUTPUT_BOX_SIDE:
            filtered.append((class_id, x1, y1, x2, y2))
    return filtered


def write_yolo_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = []
    for class_id, x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2.0) / OUTPUT_SIZE
        yc = ((y1 + y2) / 2.0) / OUTPUT_SIZE
        bw = (x2 - x1) / OUTPUT_SIZE
        bh = (y2 - y1) / OUTPUT_SIZE
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_processed(records: list[Record]) -> tuple[list[dict], dict[str, tuple[Image.Image, list[tuple[int, float, float, float, float]], Record]]]:
    rows: list[dict] = []
    cache: dict[str, tuple[Image.Image, list[tuple[int, float, float, float, float]], Record]] = {}
    for record in records:
        with Image.open(record.image_path) as img:
            img = img.convert("RGB")
            square, square_boxes, square_meta = pad_to_square_and_resize(img, record.boxes)
        if not square_boxes:
            record.status = "removed"
            record.remove_reason = "no_valid_annotation_after_square_transform"
            continue
        image_out = PROCESSED_DIR / "images" / record.split / f"{record.output_stem}.jpg"
        label_out = PROCESSED_DIR / "labels" / record.split / f"{record.output_stem}.txt"
        square.save(image_out, quality=JPEG_QUALITY, optimize=True)
        write_yolo_label(label_out, square_boxes)
        cache[record.output_stem] = (square.copy(), square_boxes, record)
        for order, (class_id, x1, y1, x2, y2) in enumerate(square_boxes, start=1):
            rows.append(
                {
                    "image_id": record.output_stem,
                    "split": record.split,
                    "source_image": relpath(record.image_path),
                    "source_xml": relpath(record.source_xml),
                    "box_index": order,
                    "class_id": class_id,
                    "raw_class_name": CLASS_DEFS[class_id][0],
                    "class_name": CLASS_DEFS[class_id][1],
                    "class_name_cn": CLASS_DEFS[class_id][2],
                    "xmin": round(x1, 3),
                    "ymin": round(y1, 3),
                    "xmax": round(x2, 3),
                    "ymax": round(y2, 3),
                    "width": round(x2 - x1, 3),
                    "height": round(y2 - y1, 3),
                    "area": round((x2 - x1) * (y2 - y1), 3),
                    "pad_x_raw": square_meta["pad_x"],
                    "pad_y_raw": square_meta["pad_y"],
                    "scale": square_meta["scale"],
                }
            )
    return rows, cache


def augment_dataset(cache: dict[str, tuple[Image.Image, list[tuple[int, float, float, float, float]], Record]]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    rng = random.Random(RANDOM_SEED + 17)
    train_items = sorted((stem, item) for stem, item in cache.items() if item[2].split == "train")
    methods = ["flip", "light", "scale", "rotate"]
    skipped = Counter()
    for idx, (stem, (img, boxes, record)) in enumerate(train_items):
        method = methods[idx % len(methods)]
        aug_img, aug_boxes = apply_augmentation(img, boxes, method, rng)
        aug_boxes = filter_output_boxes(aug_boxes)
        if not aug_boxes:
            skipped[method] += 1
            continue
        aug_stem = f"{stem}_{method}"
        image_out = PROCESSED_DIR / "images" / "train" / f"{aug_stem}.jpg"
        label_out = PROCESSED_DIR / "labels" / "train" / f"{aug_stem}.txt"
        aug_img.save(image_out, quality=JPEG_QUALITY, optimize=True)
        write_yolo_label(label_out, aug_boxes)
        for order, (class_id, x1, y1, x2, y2) in enumerate(aug_boxes, start=1):
            rows.append(
                {
                    "image_id": aug_stem,
                    "split": "train",
                    "source_image": relpath(record.image_path),
                    "source_xml": relpath(record.source_xml),
                    "box_index": order,
                    "class_id": class_id,
                    "raw_class_name": CLASS_DEFS[class_id][0],
                    "class_name": CLASS_DEFS[class_id][1],
                    "class_name_cn": CLASS_DEFS[class_id][2],
                    "xmin": round(x1, 3),
                    "ymin": round(y1, 3),
                    "xmax": round(x2, 3),
                    "ymax": round(y2, 3),
                    "width": round(x2 - x1, 3),
                    "height": round(y2 - y1, 3),
                    "area": round((x2 - x1) * (y2 - y1), 3),
                    "augmentation": method,
                }
            )
    create_examples(train_items, methods)
    return rows, {"methods": methods, "skipped": dict(skipped), "created_images": len({r["image_id"] for r in rows})}


def apply_augmentation(
    img: Image.Image,
    boxes: list[tuple[int, float, float, float, float]],
    method: str,
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    if method == "flip":
        out = ImageOps.mirror(img)
        new_boxes = [(cid, OUTPUT_SIZE - x2, y1, OUTPUT_SIZE - x1, y2) for cid, x1, y1, x2, y2 in boxes]
        return out, new_boxes
    if method == "light":
        brightness = rng.uniform(0.82, 1.18)
        contrast = rng.uniform(0.86, 1.16)
        out = ImageEnhance.Brightness(img).enhance(brightness)
        out = ImageEnhance.Contrast(out).enhance(contrast)
        return out, list(boxes)
    if method == "scale":
        scale = rng.uniform(0.86, 1.14)
        new_size = max(1, int(round(OUTPUT_SIZE * scale)))
        resized = img.resize((new_size, new_size), Image.Resampling.BICUBIC)
        max_shift = int(OUTPUT_SIZE * 0.08)
        dx = int(round((OUTPUT_SIZE - new_size) / 2 + rng.randint(-max_shift, max_shift)))
        dy = int(round((OUTPUT_SIZE - new_size) / 2 + rng.randint(-max_shift, max_shift)))
        canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), (114, 114, 114))
        paste_x = max(0, dx)
        paste_y = max(0, dy)
        crop_x = max(0, -dx)
        crop_y = max(0, -dy)
        crop_w = min(new_size - crop_x, OUTPUT_SIZE - paste_x)
        crop_h = min(new_size - crop_y, OUTPUT_SIZE - paste_y)
        if crop_w > 0 and crop_h > 0:
            canvas.paste(resized.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)), (paste_x, paste_y))
        new_boxes = [(cid, x1 * scale + dx, y1 * scale + dy, x2 * scale + dx, y2 * scale + dy) for cid, x1, y1, x2, y2 in boxes]
        return canvas, new_boxes
    if method == "rotate":
        angle = rng.uniform(-7.0, 7.0)
        out = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(114, 114, 114))
        new_boxes = [rotate_box(box, angle) for box in boxes]
        return out, new_boxes
    raise ValueError(f"Unknown augmentation method: {method}")


def rotate_box(box: tuple[int, float, float, float, float], angle: float) -> tuple[int, float, float, float, float]:
    cid, x1, y1, x2, y2 = box
    cx = cy = OUTPUT_SIZE / 2.0
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    rotated = []
    for x, y in corners:
        tx, ty = x - cx, y - cy
        rx = tx * cos_t - ty * sin_t + cx
        ry = tx * sin_t + ty * cos_t + cy
        rotated.append((rx, ry))
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return cid, min(xs), min(ys), max(xs), max(ys)


def font(size: int = 18):
    for name in ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(img: Image.Image, boxes: list[tuple[int, float, float, float, float]], title: str = "") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (0, 109, 119),
        (214, 40, 40),
        (58, 134, 255),
        (106, 153, 78),
    ]
    fnt = font(16)
    for cid, x1, y1, x2, y2 in boxes:
        color = palette[cid % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{cid}:{CLASS_DEFS[cid][1]}"
        label_box = draw.textbbox((0, 0), label, font=fnt)
        tw = label_box[2] - label_box[0]
        th = label_box[3] - label_box[1]
        draw.rectangle((x1, max(0, y1 - th - 6), x1 + tw + 8, max(th + 6, y1)), fill=color)
        draw.text((x1 + 4, max(0, y1 - th - 3)), label, fill=(255, 255, 255), font=fnt)
    if title:
        title_font = font(24)
        draw.rectangle((0, 0, out.width, 42), fill=(20, 20, 20))
        draw.text((14, 9), title, fill=(255, 255, 255), font=title_font)
    return out


def make_contact_sheet(items: list[Image.Image], cols: int, cell_size: tuple[int, int], bg=(245, 245, 245)) -> Image.Image:
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_size[0], rows * cell_size[1]), bg)
    for idx, img in enumerate(items):
        thumb = ImageOps.contain(img, cell_size, Image.Resampling.LANCZOS)
        x = (idx % cols) * cell_size[0] + (cell_size[0] - thumb.width) // 2
        y = (idx // cols) * cell_size[1] + (cell_size[1] - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    return sheet


def create_examples(train_items: list[tuple[str, tuple[Image.Image, list[tuple[int, float, float, float, float]], Record]]], methods: list[str]) -> None:
    if not train_items:
        return
    rng = random.Random(RANDOM_SEED + 37)
    selected = train_items[: min(4, len(train_items))]
    panels = []
    for stem, (img, boxes, _) in selected:
        panels.append(draw_boxes(img, boxes, f"original {stem}"))
        for method in methods:
            aug_img, aug_boxes = apply_augmentation(img, boxes, method, rng)
            panels.append(draw_boxes(aug_img, filter_output_boxes(aug_boxes), method))
    sheet = make_contact_sheet(panels, cols=5, cell_size=(360, 360))
    sheet.save(FIGURES_DIR / "augmentation_examples" / "examples.jpg", quality=JPEG_QUALITY)


def create_visualizations(records: list[Record], cache: dict[str, tuple[Image.Image, list[tuple[int, float, float, float, float]], Record]], all_rows: pd.DataFrame, removed_rows: list[dict]) -> None:
    by_class: dict[int, list[str]] = defaultdict(list)
    for _, row in all_rows.drop_duplicates(["image_id", "class_id"]).iterrows():
        if len(by_class[int(row.class_id)]) < 8:
            by_class[int(row.class_id)].append(str(row.image_id))

    for class_id, stems in by_class.items():
        images = []
        for stem in stems[:5]:
            split = str(all_rows.loc[all_rows["image_id"] == stem, "split"].iloc[0])
            path = PROCESSED_DIR / "images" / split / f"{stem}.jpg"
            label_rows = all_rows[all_rows["image_id"] == stem]
            boxes = [
                (int(r.class_id), float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))
                for r in label_rows.itertuples()
            ]
            with Image.open(path) as img:
                images.append(draw_boxes(img.convert("RGB"), boxes, f"{CLASS_DEFS[class_id][1]} / {stem}"))
        if images:
            class_dir = FIGURES_DIR / "samples" / f"{class_id + 1:02d}_{CLASS_DEFS[class_id][1]}"
            class_dir.mkdir(parents=True, exist_ok=True)
            make_contact_sheet(images, cols=min(5, len(images)), cell_size=(360, 360)).save(
                class_dir / "samples.jpg", quality=JPEG_QUALITY
            )

    first = next(iter(cache.values()), None)
    if first:
        square_img, square_boxes, record = first
        with Image.open(record.image_path) as raw_img:
            raw_img = raw_img.convert("RGB")
            raw_panel = draw_boxes(raw_img, record.boxes, "raw image and VOC boxes")
        square_panel = draw_boxes(square_img, square_boxes, "square padded 1024x1024")
        make_contact_sheet([raw_panel, square_panel], cols=2, cell_size=(520, 520)).save(
            FIGURES_DIR / "square" / "before_after.jpg", quality=JPEG_QUALITY
        )

    cleaning_panels = []
    removed_unknown = [r for r in removed_rows if "unknown_class" in r.get("reason", "") or "no_valid_annotation" in r.get("reason", "")]
    if removed_unknown:
        path = ROOT / removed_unknown[0]["image_path"]
        if path.exists():
            with Image.open(path) as img:
                cleaning_panels.append(draw_text_panel(img.convert("RGB"), "removed: no valid class after cleaning"))
    if first:
        cleaning_panels.append(draw_boxes(first[0], first[1], "kept: valid label after cleaning"))
    if cleaning_panels:
        make_contact_sheet(cleaning_panels, cols=len(cleaning_panels), cell_size=(520, 520)).save(
            FIGURES_DIR / "clean" / "before_after.jpg", quality=JPEG_QUALITY
        )


def draw_text_panel(img: Image.Image, text: str) -> Image.Image:
    out = ImageOps.contain(img.copy(), (1024, 1024), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1024, 1024), (114, 114, 114))
    canvas.paste(out, ((1024 - out.width) // 2, (1024 - out.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 48), fill=(20, 20, 20))
    draw.text((16, 12), text, fill=(255, 255, 255), font=font(24))
    return canvas


def write_raw_annotation_summary(records: list[Record]) -> pd.DataFrame:
    rows = []
    for record in records:
        if not record.raw_boxes:
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_image": relpath(record.image_path) if record.image_path else "",
                    "source_xml": relpath(record.source_xml),
                    "raw_folder": record.raw_folder,
                    "raw_filename": record.raw_filename,
                    "raw_class_name": "",
                    "normalized_class_id": "",
                    "normalized_class_name": "",
                    "xmin": "",
                    "ymin": "",
                    "xmax": "",
                    "ymax": "",
                    "box_width": "",
                    "box_height": "",
                    "status": "no_object_in_xml",
                    "reason": "",
                }
            )
            continue
        for raw_box in record.raw_boxes:
            cid = raw_box.class_id
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_image": relpath(record.image_path) if record.image_path else "",
                    "source_xml": relpath(record.source_xml),
                    "raw_folder": record.raw_folder,
                    "raw_filename": record.raw_filename,
                    "raw_class_name": raw_box.raw_name,
                    "normalized_class_id": cid if cid is not None else "",
                    "normalized_class_name": CLASS_DEFS[cid][1] if cid is not None else "",
                    "xmin": raw_box.xmin,
                    "ymin": raw_box.ymin,
                    "xmax": raw_box.xmax,
                    "ymax": raw_box.ymax,
                    "box_width": raw_box.xmax - raw_box.xmin,
                    "box_height": raw_box.ymax - raw_box.ymin,
                    "status": raw_box.status,
                    "reason": raw_box.reason,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(META_DIR / "raw_annotation_summary.csv", index=False, encoding="utf-8-sig")
    return df


def write_data_yaml() -> None:
    names = [item[1] for item in CLASS_DEFS]
    text = [
        f"path: {PROCESSED_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    text.extend([f"  {idx}: {name}" for idx, name in enumerate(names)])
    (PROCESSED_DIR / "data.yaml").write_text("\n".join(text) + "\n", encoding="utf-8")


def final_label_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(META_DIR / "processed_annotations.csv", index=False, encoding="utf-8-sig")
    return df


def count_images_by_split() -> dict[str, int]:
    return {
        split: len(list((PROCESSED_DIR / "images" / split).glob("*.jpg")))
        for split in ["train", "val", "test"]
    }


def create_charts(
    split_stats: pd.DataFrame,
    class_stats: pd.DataFrame,
    size_stats: pd.DataFrame,
    stage_stats: pd.DataFrame,
    granularity_stats: pd.DataFrame,
) -> None:
    chart_dir = FIGURES_DIR / "chart"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.4,
        "axes.edgecolor": "#1a1a1a",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#1a1a1a",
        "legend.fancybox": False,
    })
    bar_kw = {"edgecolor": "#1a1a1a", "linewidth": 0.8}

    def _polish(ax: plt.Axes, *, y_minor: bool = True) -> None:
        if y_minor:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    split_stats.plot(x="split", y="image_count", kind="bar", legend=False, color="#2a9d8f", ax=ax, **bar_kw)
    ax.set_title("Image count by split")
    ax.set_ylabel("images")
    ax.set_xlabel("split")
    plt.xticks(rotation=0)
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "split_image_count.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = list(range(len(class_stats)))
    width = 0.4
    ax.bar([i - width / 2 for i in x], class_stats["image_count"], width=width, color="#457b9d", label="image_count", **bar_kw)
    ax.bar([i + width / 2 for i in x], class_stats["annotation_count"], width=width, color="#e76f51", label="annotation_count", **bar_kw)
    ax.set_xticks(x)
    ax.set_xticklabels(class_stats["class_name"], rotation=35, ha="right")
    ax.set_title("Per-class image vs annotation count")
    ax.set_ylabel("count")
    ax.legend()
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "class_image_vs_annotation.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    class_stats.plot(x="class_name", y="annotation_count", kind="bar", legend=False, color="#457b9d", ax=ax, **bar_kw)
    ax.set_title("Annotations per class")
    ax.set_ylabel("annotations")
    ax.set_xlabel("class")
    plt.xticks(rotation=35, ha="right")
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "class_annotation_count.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    size_stats.plot(x="size_bucket", y="annotation_count", kind="bar", legend=False, color="#e76f51", ax=ax, **bar_kw)
    ax.set_title("Annotations by box size bucket (1024x1024 px²)")
    ax.set_ylabel("annotations")
    ax.set_xlabel("size bucket")
    plt.xticks(rotation=0)
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "size_bucket_count.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    stage_stats.plot(x="stage", y="image_count", kind="line", marker="o", legend=False, color="#264653", ax=ax, linewidth=1.8, markersize=7, markeredgecolor="#1a1a1a", markeredgewidth=0.8)
    ax.set_title("Images surviving each cleaning stage")
    ax.set_ylabel("images")
    ax.set_xlabel("stage")
    plt.xticks(rotation=25, ha="right")
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "cleaning_stage_image_count.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    granularity_stats.plot(x="defects_per_image", y="image_count", kind="bar", legend=False, color="#9b5de5", ax=ax, **bar_kw)
    ax.set_title("Image count by defects per image")
    ax.set_ylabel("images")
    ax.set_xlabel("defects per image")
    plt.xticks(rotation=0)
    _polish(ax)
    plt.tight_layout()
    plt.savefig(chart_dir / "defects_per_image_count.png", dpi=180)
    plt.close()


def build_stats(raw_summary: dict, stage_counter: dict, clean_records: list[Record], all_df: pd.DataFrame, aug_meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_counts = count_images_by_split()
    split_rows = []
    for split in ["train", "val", "test"]:
        sub = all_df[all_df["split"] == split]
        image_count = image_counts[split]
        split_rows.append(
            {
                "split": split,
                "image_count": image_count,
                "annotation_count": int(len(sub)),
                "avg_annotations_per_image": round(len(sub) / image_count, 3) if image_count else 0,
            }
        )
    split_stats = pd.DataFrame(split_rows)

    total_ann = len(all_df)
    class_rows = []
    for cid, (_, en, cn) in enumerate(CLASS_DEFS):
        sub = all_df[all_df["class_id"] == cid]
        class_rows.append(
            {
                "class_id": cid,
                "class_name": en,
                "class_name_cn": cn,
                "image_count": int(sub["image_id"].nunique()),
                "annotation_count": int(len(sub)),
                "annotation_ratio": round(len(sub) / total_ann, 4) if total_ann else 0,
            }
        )
    class_stats = pd.DataFrame(class_rows)

    def size_bucket(area: float) -> str:
        if area < 32 * 32:
            return "small"
        if area < 96 * 96:
            return "medium"
        return "large"

    size_stats = (
        all_df.assign(size_bucket=all_df["area"].apply(size_bucket))
        .groupby("size_bucket", as_index=False)
        .agg(annotation_count=("image_id", "count"))
    )
    size_stats["threshold"] = size_stats["size_bucket"].map(
        {
            "small": "area < 32^2 px",
            "medium": "32^2 <= area < 96^2 px",
            "large": "area >= 96^2 px",
        }
    )
    order = {"small": 0, "medium": 1, "large": 2}
    size_stats = size_stats.sort_values("size_bucket", key=lambda s: s.map(order))

    clean_image_count = len(clean_records)
    clean_box_count = sum(len(r.boxes) for r in clean_records)
    augmented_images = int(aug_meta["created_images"])
    stage_rows = [
        {
            "stage": "raw",
            "image_count": raw_summary["raw_image_files"],
            "annotation_count": raw_summary["raw_object_count"],
            "note": "原始图片文件与 VOC 标注框",
        },
        {
            "stage": "after_corrupt_check",
            "image_count": stage_counter["records_after_corrupt_check"],
            "annotation_count": raw_summary["raw_object_count"],
            "note": "完成图片可读性检查",
        },
        {
            "stage": "after_duplicate_check",
            "image_count": stage_counter["records_after_duplicate_check"],
            "annotation_count": raw_summary["raw_object_count"],
            "note": "删除完全重复与近似重复记录",
        },
        {
            "stage": "after_invalid_label_filter",
            "image_count": clean_image_count,
            "annotation_count": clean_box_count,
            "note": "删除未知、越界、零面积与过小标注",
        },
        {
            "stage": "after_square_transform",
            "image_count": clean_image_count,
            "annotation_count": clean_box_count,
            "note": "padding 为正方形并缩放到 1024",
        },
        {
            "stage": "after_train_augmentation",
            "image_count": int(sum(image_counts.values())),
            "annotation_count": int(len(all_df)),
            "note": "对符合条件的训练图生成增强样本",
        },
    ]
    stage_stats = pd.DataFrame(stage_rows)

    per_image = all_df.groupby("image_id").size()
    granularity_rows = [
        {"defects_per_image": "1", "image_count": int((per_image == 1).sum())},
        {"defects_per_image": "2-3", "image_count": int(((per_image >= 2) & (per_image <= 3)).sum())},
        {"defects_per_image": "4+", "image_count": int((per_image >= 4).sum())},
    ]
    granularity_stats = pd.DataFrame(granularity_rows)

    for name, df in [
        ("split_stats.csv", split_stats),
        ("class_stats.csv", class_stats),
        ("size_stats.csv", size_stats),
        ("stage_stats.csv", stage_stats),
        ("image_granularity_stats.csv", granularity_stats),
    ]:
        df.to_csv(META_DIR / name, index=False, encoding="utf-8-sig")

    create_charts(split_stats, class_stats, size_stats, stage_stats, granularity_stats)
    return split_stats, class_stats, size_stats, stage_stats, granularity_stats


def leakage_audit(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = all_df[["image_id", "source_image"]].drop_duplicates().set_index("image_id")["source_image"].to_dict()
    augmented_ids = set()
    if "augmentation" in all_df.columns:
        augmented_ids = set(all_df.loc[all_df["augmentation"].notna(), "image_id"].astype(str))
    image_paths = []
    for split in ["train", "val", "test"]:
        for path in (PROCESSED_DIR / "images" / split).glob("*.jpg"):
            with Image.open(path) as img:
                img = img.convert("RGB")
                src = sources.get(path.stem, "")
                image_paths.append(
                    {
                        "image_id": path.stem,
                        "split": split,
                        "path": relpath(path),
                        "md5": compute_md5(path),
                        "dhash": compute_dhash(img),
                        "source_key": source_key(Path(src).name) if src else "",
                    }
                )
    by_md5 = defaultdict(list)
    for item in image_paths:
        by_md5[item["md5"]].append(item)
    for md5, items in by_md5.items():
        splits = {item["split"] for item in items}
        if len(splits) > 1:
            rows.append(
                {
                    "check": "exact_duplicate_across_splits",
                    "severity": "high",
                    "sample_a": items[0]["path"],
                    "sample_b": items[1]["path"],
                    "detail": f"相同 MD5 同时出现在 {sorted(splits)}",
                }
            )
    clean_items = [item for item in image_paths if item["image_id"] not in augmented_ids]
    for i, a in enumerate(clean_items):
        for b in clean_items[i + 1 :]:
            if a["split"] == b["split"]:
                continue
            if a["source_key"] != b["source_key"]:
                continue
            dist = hamming(a["dhash"], b["dhash"])
            if dist <= 2:
                rows.append(
                    {
                        "check": "near_duplicate_across_splits",
                        "severity": "medium",
                        "sample_a": a["path"],
                        "sample_b": b["path"],
                        "detail": f'同一采集键 {a["source_key"]} 的 dHash 汉明距离为 {dist}',
                    }
                )
                break
    source_splits: dict[str, set[str]] = defaultdict(set)
    for item in clean_items:
        if item["source_key"]:
            source_splits[item["source_key"]].add(item["split"])
    for key, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            samples = [item for item in clean_items if item["source_key"] == key]
            rows.append(
                {
                    "check": "source_sequence_across_splits",
                    "severity": "medium",
                    "sample_a": samples[0]["path"],
                    "sample_b": samples[-1]["path"],
                    "detail": f"采集键 {key} 同时出现在 {sorted(splits)}",
                }
            )
    if not rows:
        rows.append(
            {
                "check": "leakage",
                "severity": "pass",
                "sample_a": "",
                "sample_b": "",
                "detail": "未发现跨最终 train/val/test 的完全重复、同源近似重复或采集序列泄漏。",
            }
        )
    return pd.DataFrame(rows)


def quality_audit(raw_summary: dict, all_df: pd.DataFrame, removed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unknown_removed = removed_df[removed_df["reason"].str.contains("no_valid_annotation|unknown_class", na=False)]
    if not unknown_removed.empty:
        rows.append(
            {
                "check": "dirty_label",
                "severity": "medium",
                "sample": unknown_removed.iloc[0]["image_path"],
                "detail": "原始标签未知或清洗后无有效标签，已在清洗阶段删除。",
            }
        )
    if raw_summary["match_counter"].get("filename_unique_folder_mismatch", 0):
        rows.append(
            {
                "check": "metadata_mismatch",
                "severity": "medium",
                "sample": "data/processed/meta/raw_annotation_summary.csv",
                "detail": f'{raw_summary["match_counter"].get("filename_unique_folder_mismatch", 0)} 条 XML 需要通过文件名回补图片路径。',
            }
        )
    large = (
        all_df.assign(area_ratio=all_df["area"] / float(OUTPUT_SIZE * OUTPUT_SIZE))
        .query("area_ratio > 0.45")
        .sort_values("area_ratio", ascending=False)
        .drop_duplicates(subset=["image_id"])
        .head(5)
    )
    for row in large.itertuples():
        rows.append(
            {
                "check": "large_or_loose_box",
                "severity": "low",
                "sample": f"data/processed/images/{row.split}/{row.image_id}.jpg",
                "detail": f"{row.class_name} 标注框面积占比 {row.area_ratio:.3f}，建议人工复查边界。",
            }
        )
    edge_mask = (
        (all_df["xmin"] <= 1)
        | (all_df["ymin"] <= 1)
        | (all_df["xmax"] >= OUTPUT_SIZE - 1)
        | (all_df["ymax"] >= OUTPUT_SIZE - 1)
    )
    edge = all_df[edge_mask].drop_duplicates(subset=["image_id"]).head(5)
    for row in edge.itertuples():
        rows.append(
            {
                "check": "edge_touching_box",
                "severity": "low",
                "sample": f"data/processed/images/{row.split}/{row.image_id}.jpg",
                "detail": f"{row.class_name} 标注框贴近输出图边界，需复查缺陷是否被截断。",
            }
        )
    rows.append(
        {
            "check": "shortcut_feature",
            "severity": "fixed",
            "sample": "data/processed/images/",
            "detail": "处理后文件名不保留原始类别目录或采集式命名，所有输出图统一为 1024x1024。",
        }
    )
    return pd.DataFrame(rows)

def write_audit_tables(raw_summary: dict, all_df: pd.DataFrame, removed_df: pd.DataFrame) -> None:
    leakage_df = leakage_audit(all_df)
    quality_df = quality_audit(raw_summary, all_df, removed_df)
    leakage_df.to_csv(META_DIR / "leakage_audit.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(META_DIR / "quality_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat([leakage_df, quality_df], ignore_index=True).to_csv(
        META_DIR / "audit_findings.csv", index=False, encoding="utf-8-sig"
    )
    create_audit_evidence_figures(all_df, removed_df)


def create_audit_evidence_figures(all_df: pd.DataFrame, removed_df: pd.DataFrame) -> None:
    audit_dir = FIGURES_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []

    loose = (
        all_df.assign(area_ratio=all_df["area"] / float(OUTPUT_SIZE * OUTPUT_SIZE))
        .query("area_ratio > 0.45")
        .sort_values("area_ratio", ascending=False)
        .drop_duplicates(subset=["image_id"])
        .head(2)
    )
    for row in loose.itertuples():
        panel = audit_panel_for_image(
            all_df,
            row.image_id,
            row.split,
            f"large box: {row.class_name} ratio={row.area_ratio:.2f}",
            highlight=(float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)),
        )
        if panel is not None:
            panels.append(panel)

    edge_mask = (
        (all_df["xmin"] <= 1)
        | (all_df["ymin"] <= 1)
        | (all_df["xmax"] >= OUTPUT_SIZE - 1)
        | (all_df["ymax"] >= OUTPUT_SIZE - 1)
    )
    edge_df = all_df[edge_mask].drop_duplicates(subset=["image_id"]).head(2)
    for row in edge_df.itertuples():
        panel = audit_panel_for_image(
            all_df,
            row.image_id,
            row.split,
            f"edge-touching: {row.class_name} {row.image_id}",
            highlight=(float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)),
        )
        if panel is not None:
            panels.append(panel)

    dirty = removed_df[removed_df["reason"].str.contains("no_valid_annotation|unknown_class", na=False)] if not removed_df.empty else removed_df
    if not dirty.empty:
        for path_str in dirty["image_path"].head(2):
            path = ROOT / str(path_str)
            if not path.exists():
                continue
            with Image.open(path) as img:
                panels.append(draw_text_panel(img.convert("RGB"), f"removed (dirty label): {Path(path_str).name}"))

    if panels:
        cols = min(3, len(panels))
        sheet = make_contact_sheet(panels, cols=cols, cell_size=(420, 420))
        sheet.save(audit_dir / "audit_findings.jpg", quality=JPEG_QUALITY)


def audit_panel_for_image(
    all_df: pd.DataFrame,
    image_id: str,
    split: str,
    title: str,
    highlight: tuple[float, float, float, float] | None = None,
) -> Image.Image | None:
    path = PROCESSED_DIR / "images" / split / f"{image_id}.jpg"
    if not path.exists():
        return None
    with Image.open(path) as img:
        base = img.convert("RGB")
    boxes = [
        (int(r.class_id), float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))
        for r in all_df[all_df["image_id"] == image_id].itertuples()
    ]
    panel = draw_boxes(base, boxes, title)
    if highlight is not None:
        hx1, hy1, hx2, hy2 = highlight
        ImageDraw.Draw(panel).rectangle(
            (max(0.0, hx1 - 6), max(0.0, hy1 - 6), min(float(OUTPUT_SIZE), hx2 + 6), min(float(OUTPUT_SIZE), hy2 + 6)),
            outline=(255, 32, 32),
            width=5,
        )
    return panel


def validate_final_dataset() -> dict:
    problems = []
    counts = {}
    for split in ["train", "val", "test"]:
        image_dir = PROCESSED_DIR / "images" / split
        label_dir = PROCESSED_DIR / "labels" / split
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))
        counts[f"{split}_images"] = len(images)
        counts[f"{split}_labels"] = len(labels)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            problems.append(f"{split}: image/label stem mismatch")
        for label in labels:
            content = label.read_text(encoding="utf-8").strip().splitlines()
            if not content:
                problems.append(f"empty label: {relpath(label)}")
            for line in content:
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"bad yolo line: {relpath(label)} -> {line}")
                    continue
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if cid < 0 or cid >= len(CLASS_DEFS):
                    problems.append(f"class id out of range: {relpath(label)}")
                if any(v < 0 or v > 1 for v in vals):
                    problems.append(f"normalized coord out of range: {relpath(label)}")
                if vals[2] <= 0 or vals[3] <= 0:
                    problems.append(f"non-positive normalized size: {relpath(label)}")
    result = {"counts": counts, "problem_count": len(problems), "problems": problems[:20]}
    (META_DIR / "validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    random.seed(RANDOM_SEED)
    require_raw_dataset()
    ensure_dirs()
    reset_outputs()
    records, raw_summary = read_records()
    raw_df = write_raw_annotation_summary(records)
    clean, stage_counter, removed_rows = clean_records(records)
    stratified_split(clean)
    clean_rows, cache = save_processed(clean)
    aug_rows, aug_meta = augment_dataset(cache)
    all_rows = clean_rows + aug_rows
    all_df = final_label_dataframe(all_rows)
    removed_df = pd.DataFrame(removed_rows)
    removed_df.to_csv(META_DIR / "removed_records.csv", index=False, encoding="utf-8-sig")
    split_stats, class_stats, size_stats, stage_stats, granularity_stats = build_stats(
        raw_summary, stage_counter, clean, all_df, aug_meta
    )
    create_visualizations(clean, cache, all_df, removed_rows)
    write_data_yaml()
    write_audit_tables(raw_summary, all_df, removed_df)
    validation = validate_final_dataset()
    pipeline_summary = {
        "raw_summary": raw_summary,
        "raw_annotation_rows": len(raw_df),
        "stage_counter": stage_counter,
        "augmentation": aug_meta,
        "validation": validation,
    }
    (META_DIR / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

