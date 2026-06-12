import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


AXIS_COLORS = {
    "subject": "#1f7a8c",
    "document_type": "#2d6cdf",
    "business_domain": "#198754",
    "modifier": "#c56a1a",
}
DEFAULT_COLOR = "#5d6b7a"
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def build_signal_overlay_images(image_paths, paddle_pages, classification_result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_pages = [_extract_ocr_items(page) for page in paddle_pages]
    signal_targets = _extract_signal_targets(classification_result)
    overlays = []

    for page_index, image_path in enumerate(image_paths):
        page_items = ocr_pages[page_index] if page_index < len(ocr_pages) else []
        page_matches = _match_signal_targets(page_items, signal_targets)
        output_path = output_dir / f"page_{page_index + 1:03d}_signals.png"
        draw_signal_overlay(image_path, output_path, page_matches)
        overlays.append(
            {
                "path": output_path,
                "matches": page_matches,
            }
        )

    return overlays


def draw_signal_overlay(image_path, output_path, matches):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGBA")

    draw_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(draw_layer)
    font = _load_font(max(12, round(image.width * 0.012)))

    for match in _group_draw_matches(matches):
        bbox = _clamp_bbox(match.get("bbox"), image.size)
        if not bbox:
            continue

        color = AXIS_COLORS.get(match.get("axis"), DEFAULT_COLOR)
        rgb = _hex_to_rgb(color)
        line_width = max(3, round(image.width * 0.0024))
        draw.rounded_rectangle(
            bbox,
            radius=max(4, line_width + 2),
            outline=rgb + (255,),
            width=line_width,
            fill=rgb + (42,),
        )
        _draw_label(draw, bbox, _label_for_match(match), font, rgb, image.size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, draw_layer).convert("RGB").save(output_path, "PNG")


def _extract_signal_targets(classification_result):
    if not isinstance(classification_result, dict):
        return []

    targets = []
    for axis, axis_result in classification_result.items():
        if not isinstance(axis_result, dict):
            continue

        signals = axis_result.get("signals")
        if isinstance(signals, str):
            signals = [signals]
        if not isinstance(signals, list):
            continue

        for signal in signals:
            signal_text = _clean_text(signal)
            signal_key = _match_key(signal_text)
            if signal_key:
                targets.append(
                    {
                        "axis": str(axis),
                        "signal": signal_text,
                        "match_key": signal_key,
                    }
                )
    return targets


def _match_signal_targets(ocr_items, signal_targets):
    matches = []
    seen = set()

    for target in signal_targets:
        for item in ocr_items:
            if not _signal_matches_text(target["match_key"], item["match_key"]):
                continue

            dedupe_key = (target["axis"], target["signal"], tuple(round(value, 1) for value in item["bbox"]))
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            matches.append(
                {
                    "axis": target["axis"],
                    "signal": target["signal"],
                    "text": item["text"],
                    "bbox": item["bbox"],
                }
            )

    return matches


def _group_draw_matches(matches):
    groups = []
    group_by_bbox = {}

    for match in matches:
        bbox = match.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        group_key = tuple(round(float(value), 1) for value in bbox)
        group = group_by_bbox.get(group_key)
        if group is None:
            group = {
                "axis": match.get("axis"),
                "bbox": bbox,
                "signals_by_axis": {},
            }
            group_by_bbox[group_key] = group
            groups.append(group)

        axis = str(match.get("axis") or "signal")
        signal = _clean_text(match.get("signal"))
        if not signal:
            continue

        group["signals_by_axis"].setdefault(axis, [])
        if signal not in group["signals_by_axis"][axis]:
            group["signals_by_axis"][axis].append(signal)

    return groups


def _signal_matches_text(signal_key, text_key):
    if not signal_key or not text_key:
        return False
    return signal_key in text_key or (len(text_key) >= 2 and text_key in signal_key)


def _extract_ocr_items(paddle_payload):
    paddle_result = _first_page_payload(paddle_payload)
    if not isinstance(paddle_result, dict):
        return []

    paddle_result = paddle_result.get("res", paddle_result)
    if not isinstance(paddle_result, dict):
        return []

    rec_texts = paddle_result.get("rec_texts")
    if isinstance(rec_texts, list):
        return _extract_rec_text_items(paddle_result, rec_texts)

    result_items = paddle_result.get("result")
    if isinstance(result_items, list):
        return _extract_result_items(result_items)

    return []


def _first_page_payload(paddle_payload):
    if isinstance(paddle_payload, list):
        return paddle_payload[0] if paddle_payload else {}
    return paddle_payload


def _extract_rec_text_items(paddle_result, rec_texts):
    boxes = paddle_result.get("rec_boxes") or paddle_result.get("dt_boxes")
    polygons = paddle_result.get("rec_polys") or paddle_result.get("dt_polys")
    items = []

    for index, text in enumerate(rec_texts):
        clean_text = _clean_text(text)
        bbox = None

        if isinstance(boxes, list) and index < len(boxes):
            bbox = _box_to_bbox(boxes[index])
        if bbox is None and isinstance(polygons, list) and index < len(polygons):
            bbox = _polygon_to_bbox(polygons[index])

        if clean_text and bbox:
            items.append(_ocr_item(clean_text, bbox))

    return items


def _extract_result_items(result_items):
    items = []
    for result_item in result_items:
        if isinstance(result_item, (list, tuple)) and len(result_item) >= 2:
            bbox = _box_to_bbox(result_item[0])
            text = _clean_text(result_item[1])
        elif isinstance(result_item, dict):
            bbox = _box_to_bbox(result_item.get("bbox") or result_item.get("box"))
            text = _clean_text(result_item.get("text"))
        else:
            continue

        if text and bbox:
            items.append(_ocr_item(text, bbox))

    return items


def _ocr_item(text, bbox):
    return {
        "text": text,
        "match_key": _match_key(text),
        "bbox": bbox,
    }


def _box_to_bbox(box):
    if not isinstance(box, (list, tuple)):
        return None

    if len(box) == 4 and all(_is_number(value) for value in box):
        x1, y1, x2, y2 = [float(value) for value in box]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    return _polygon_to_bbox(box)


def _polygon_to_bbox(polygon):
    if not isinstance(polygon, (list, tuple)):
        return None

    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        if _is_number(point[0]) and _is_number(point[1]):
            points.append((float(point[0]), float(point[1])))

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _clamp_bbox(bbox, image_size):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    image_width, image_height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0, min(image_width - 1, x1))
    x2 = max(0, min(image_width - 1, x2))
    y1 = max(0, min(image_height - 1, y1))
    y2 = max(0, min(image_height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _draw_label(draw, bbox, label, font, rgb, image_size):
    image_width, image_height = image_size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding_x = 7
    padding_y = 4
    label_width = min(image_width - 2, text_width + padding_x * 2)
    label_height = text_height + padding_y * 2
    x = max(1, min(bbox[0], image_width - label_width - 1))
    y = bbox[1] - label_height - 4
    if y < 1:
        y = min(image_height - label_height - 1, bbox[3] + 4)

    label_bbox = [x, y, x + label_width, y + label_height]
    draw.rounded_rectangle(label_bbox, radius=5, fill=rgb + (232,))
    draw.text((x + padding_x, y + padding_y - 1), label, fill=(255, 255, 255, 255), font=font)


def _label_for_match(match):
    signals_by_axis = match.get("signals_by_axis")
    if isinstance(signals_by_axis, dict) and signals_by_axis:
        labels = []
        for axis, signals in signals_by_axis.items():
            labels.append(f"{axis}: {', '.join(signals)}")
        return " | ".join(labels)

    axis = str(match.get("axis") or "signal")
    signal = _clean_text(match.get("signal"))
    return f"{axis}: {signal}" if signal else axis


def _load_font(size):
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _match_key(value):
    normalized = unicodedata.normalize("NFKC", _clean_text(value)).lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def _clean_text(value):
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)
