import html
import re
import unicodedata


HANGUL_RANGES = (
    (0xAC00, 0xD7A3),
    (0x3130, 0x318F),
    (0x1100, 0x11FF),
)

DEEPSEEK_BLOCK_TOKEN_PATTERN = re.compile(
    r"<\|(?:det|ref|rec)\|>.*?<\|/(?:det|ref|rec)\|>",
    flags=re.I | re.S,
)
DEEPSEEK_SIMPLE_BLOCK_PATTERN = re.compile(
    r"<(?:det|ref|rec)>.*?</(?:det|ref|rec)>",
    flags=re.I | re.S,
)
DEEPSEEK_TOKEN_PATTERN = re.compile(
    r"</?(?:det|ref|rec)>|<\|[^>]+?\|>|<image>",
    flags=re.I,
)
HTML_LINE_BREAK_PATTERN = re.compile(r"<br\s*/?>|</(td|th|tr|table|p|div|li)>", flags=re.I)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def remove_extra_whitespace(text):
    lines = []
    for line in str(text or "").replace("\r", "\n").split("\n"):
        clean_line = " ".join(line.split()).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines).strip()


def remove_non_korean_english_special_chars(text):
    normalized_text = unicodedata.normalize("NFKC", str(text or ""))
    normalized_chars = []
    for char in normalized_text:
        if is_allowed_text_char(char):
            normalized_chars.append(char)
        else:
            normalized_chars.append(" ")
    return "".join(normalized_chars)


def remove_html_tags(text):
    normalized_text = html.unescape(str(text or ""))
    normalized_text = HTML_LINE_BREAK_PATTERN.sub("\n", normalized_text)
    return HTML_TAG_PATTERN.sub(" ", normalized_text)


def remove_deepseek_special_tokens(text):
    normalized_text = str(text or "")
    normalized_text = DEEPSEEK_BLOCK_TOKEN_PATTERN.sub("\n", normalized_text)
    normalized_text = DEEPSEEK_SIMPLE_BLOCK_PATTERN.sub("\n", normalized_text)
    return DEEPSEEK_TOKEN_PATTERN.sub(" ", normalized_text)


def normalize_ocr_text(text):
    normalized_text = remove_deepseek_special_tokens(text)
    normalized_text = remove_html_tags(normalized_text)
    normalized_text = remove_non_korean_english_special_chars(normalized_text)
    return remove_extra_whitespace(normalized_text)


def is_allowed_text_char(char):
    if not is_letter(char):
        return True
    if is_ascii_letter(char):
        return True
    return is_hangul_char(char)


def is_letter(char):
    return unicodedata.category(char).startswith("L")


def is_ascii_letter(char):
    return ("A" <= char <= "Z") or ("a" <= char <= "z")


def is_hangul_char(char):
    char_code = ord(char)
    return any(start <= char_code <= end for start, end in HANGUL_RANGES)
