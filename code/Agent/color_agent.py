
import time
import base64
import json
import re
from pathlib import Path
from typing import Any
import requests
import mimetypes


API_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGE_MODEL = "google/gemini-3.1-flash-image"
COUNT_MODEL = "google/gemini-2.5-flash-lite"    # cheap vision model for counting

PALETTE = {
    "red":     "#E6194B",
    "green":   "#3CB44B",
    "blue":    "#4363D8",
    "orange":  "#F58231",
    "purple":  "#911EB4",
    "cyan":    "#42D4F4",
    "yellow":  "#FFE119",
    "brown":   "#9A6324",
}
_PALETTE_LINES = "\n".join(f"- {name} {hex_}" for name, hex_ in PALETTE.items())

COLOR_PROMPT = f"""
This image is an ornamental motif drawn in black ink on white.
The ornamental motif could be in black stroke lines or filled black ink.

Recolour the stroke lines and black ink so that each KIND of ELEMENT has its own colour.
An element is a complete, self-contained shape that a designer would draw
as one piece and could reuse elsewhere. Not a fragment of a shape, and
not a group of several elements.

ELEMENT MATCHING:
- Identify complete, reusable motif elements.
- Elements that are approximately the same shape are the same kind,
  including rotated, mirrored, or scaled copies.
- Assign one consistent flat colour to every instance of the same kind.
- Assign different colours to different kinds.
- Be tolerant of handmade irregularities and imperfect lines. The drawing is hand-made, copies are only approximately
  identical, and lines are not perfectly clean. Shapes that look alike at a glance are the same kind.
- Keep every element whole. Never split one element across colours.

COLOURS:
- Use ONLY these colours, exactly as given, one per kind:
{_PALETTE_LINES}
- Use as few colours as there are kinds; you do not need to use all of them.
- Never use black, grey, white or any colour not in this list for the ink.

IMAGE PRESERVATION:
- Recolour existing black ink only.
- Preserve the original geometry, position, scale, and layout.
- Do not add, remove, move, redraw, or simplify any elements.
- Preserve all white pixels and white regions exactly as white. Do not fill enclosed white areas.
- Use flat, solid colours only. No gradients, shading, or texture.
- Do not draw darker outlines or edges around shapes; a stroke has one colour across its full width.
- Replace all black ink with assigned colours.
- Do not add text, labels, or a legend.

OUTPUT REQUIREMENTS:
- Return the recoloured image as an image output.
"""


RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


# Parsing functions
def _post_with_retry(body: dict, api_key: str, max_retries: int, timeout: int):
    """
    POST to OpenRouter with retries.
    Returns (data, attempts, error). On success error is None.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code >= 400:
            last_error = f"HTTP {response.status_code}: {response.text[:3000]}"
            if response.status_code not in RETRYABLE_STATUSES or attempt == max_retries:
                return None, attempt, last_error
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
            except ValueError:
                delay = 2 ** (attempt - 1)
            time.sleep(min(delay, 30))
            continue

        try:
            data = response.json()
        except ValueError:
            last_error = "Successful HTTP response contained invalid JSON."
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

        if not data.get("choices"):
            last_error = f"No choices in API response: {str(data)[:2000]}"
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

        return data, attempt, None

    return None, max_retries, last_error or "Request failed."


def _message(data: dict) -> dict:
    return (data.get("choices") or [{}])[0].get("message") or {}


def _extract_text(message: dict) -> str | None:
    content = message.get("content")
    parts = []
    if isinstance(content, str):
        parts.append(content.strip())
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" \
                    and isinstance(part.get("text"), str):
                parts.append(part["text"].strip())
    return "\n".join(p for p in parts if p) or None


def _cost(data: dict) -> tuple[float | None, dict]:
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    return (float(cost) if cost is not None else None), usage


def call_image(
    img_url: str,
    api_key: str,
    output_path: str,
    prompt: str = COLOR_PROMPT,
    model: str = IMAGE_MODEL,
    max_retries: int = 3,
    timeout: int = 180,
) -> dict[str, Any]:
    """Recolour an inked image. Returns the saved image path, no report."""
    body = {
        "model": model,
        "modalities": ["image", "text"],   # Gemini image models need both listed
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }],
    }

    def fail(error, attempts, data=None):
        cost, usage = _cost(data) if data else (None, None)
        return {"success": False, "output_path": None, "cost_usd": cost,
                "usage": usage, "error": error, "attempts": attempts}

    total_attempts = 0
    last_error = None
    # HTTP errors are retried inside _post_with_retry; this loop additionally
    # retries when the call succeeds but no image comes back.
    for _ in range(max_retries):
        data, attempts, error = _post_with_retry(body, api_key, max_retries, timeout)
        total_attempts += attempts
        if error:
            return fail(error, total_attempts)

        message = _message(data)
        images = message.get("images") or []
        image_url = images[0].get("image_url", {}).get("url") \
            if images and isinstance(images[0], dict) else None

        if not isinstance(image_url, str):
            last_error = ("No generated image in choices[0].message.images. "
                          f"Text received: {_extract_text(message)!r}")
            time.sleep(2)
            continue

        match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$",
                         image_url, flags=re.DOTALL)
        if not match:
            return fail(f"Unsupported image URL prefix: {image_url[:100]!r}",
                        total_attempts, data)

        media_type, b64 = match.groups()
        try:
            image_bytes = base64.b64decode(b64, validate=True)
        except ValueError as exc:
            return fail(f"Invalid base64 image: {exc}", total_attempts, data)

        extension = {"image/png": ".png", "image/jpeg": ".jpg",
                     "image/webp": ".webp"}.get(media_type, ".img")
        dest = Path(output_path)
        if not dest.suffix:
            dest = dest.with_suffix(extension)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)

        cost, usage = _cost(data)
        return {"success": True, "output_path": str(dest), "cost_usd": cost,
                "usage": usage, "error": None, "attempts": total_attempts}

    return fail(last_error or "Image editing failed.", total_attempts)

