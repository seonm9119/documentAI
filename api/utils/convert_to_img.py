import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSION = ".pdf"
OUTPUT_EXTENSION = ".png"
DEFAULT_DPI = 200


def convert_uploaded_file_to_images(source_path, output_dir, dpi=DEFAULT_DPI):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == PDF_EXTENSION:
        return _convert_pdf_to_images(source_path, output_dir, dpi)
    if suffix in IMAGE_EXTENSIONS:
        return [_convert_image_to_png(source_path, output_dir)]

    raise ValueError(f"Unsupported upload file type: {suffix or '(no extension)'}")


def _convert_pdf_to_images(source_path, output_dir, dpi):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF(fitz) is required to convert PDF files.") from exc

    image_paths = []
    document = fitz.open(str(source_path))
    try:
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        stem = _safe_stem(source_path)

        for page_index in range(len(document)):
            page_no = page_index + 1
            image_path = output_dir / f"{stem}_page_{page_no:03d}{OUTPUT_EXTENSION}"
            page = document.load_page(page_index)
            page.get_pixmap(matrix=matrix, alpha=False).save(str(image_path))
            image_paths.append(image_path)
    finally:
        document.close()

    return image_paths


def _convert_image_to_png(source_path, output_dir):
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("Pillow is required to convert image files.") from exc

    stem = _safe_stem(source_path)
    image_path = output_dir / f"{stem}_page_001{OUTPUT_EXTENSION}"

    try:
        with Image.open(source_path) as image:
            _to_rgb(image, Image).save(image_path, format="PNG")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image file: {source_path}") from exc

    return image_path


def _to_rgb(image, image_module):
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba_image = image.convert("RGBA")
        background = image_module.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        background.alpha_composite(rgba_image)
        return background.convert("RGB")
    return image.convert("RGB")


def _safe_stem(source_path):
    stem = Path(source_path).stem.strip()
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._-")
    return stem or "uploaded"
