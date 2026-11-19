import json
import re
import shutil
from pathlib import Path

from PIL import Image

from Agent.color_agent import (COLOR_PROMPT, IMAGE_MODEL, PALETTE, call_image,
                               _post_with_retry, _message, _extract_text, _cost)
from Process.render import from_img_to_url

JUDGE_MODEL = IMAGE_MODEL # same model as the colour agent, asked for text only
MAX_REDO = 5 # redos per image after the first attempt, so 6 in total 

JUDGE_PROMPT = f"""
You are reviewing the work of an image model that recoloured an ornamental motif.
Image 1 is the original motif in black ink on white. Image 2 is the recoloured result.

The task given to the model was:
- Each KIND of element gets its own flat colour. An element is a complete, self-contained
  shape that a designer would draw as one piece and could reuse elsewhere.
- Every copy of the same kind, including rotated, mirrored or scaled copies, gets the same colour.
- Every element is one colour, never split across colours.
- Only the existing black ink is recoloured: nothing added, removed, moved or redrawn,
  and enclosed white areas stay white.
- Colours only from this list: {", ".join(PALETTE)}.

WHAT COUNTS IS THE COLOUR OF THE INK LINES:
The result is used by keeping only the pixels that are black ink in image 1 and reading their
colour in image 2. Colour painted on the white areas is thrown away. So judge the lines
themselves, not the areas between them.
- Many motifs are line art: an element is drawn only by its outline and inner lines (petal
  lines, leaf veins). Then every line of the element, outline and inner lines alike, must be
  in the element's colour, and the space inside must stay white.
- Colouring the space inside a shape while its lines stay black, grey or dark is a failure,
  even if the image looks nicely coloured. For example, in a line drawing of a peony with
  leaves, the petal outlines must be red (or the peony's colour) and the leaf outlines and
  veins green (or the leaves' colour); a red-filled peony with dark outlines and green-filled
  leaves with dark veins fails, because all its lines are still dark.
- Lines of different kinds must be in different colours. Lines of different kinds that all
  end up in the same dark colour fail SEPARATED.
- Never ask the model to fill a shape solid or to colour the white inside an element.

Compare image 2 with image 1 and check:
1. CONSISTENT: all copies of the same kind have lines of the same colour.
2. SEPARATED: different kinds have lines of different colours.
3. WHOLE: no element has its lines split across several colours.
4. FAITHFUL: no element added, removed, moved or redrawn; enclosed white areas are still white,
   not filled.
5. CLEAN: lines in flat colours from the list, full stroke width; no black, grey or dark lines
   left, no dark outlines around coloured shapes.

Be practical. The drawing is hand-made, so copies are only approximately identical.
Where it is a matter of opinion whether two shapes are the same kind, or whether a part is
a separate element, accept the model's choice. Ignore anti-aliasing along stroke edges.

Answer with JSON only, no other text:
{{"score": <integer 1-10, 10 = perfect>,
  "verdict": "pass" or "redo",
  "problems": [{{"check": "CONSISTENT | SEPARATED | WHOLE | FAITHFUL | CLEAN",
                "where": "<which elements, where in the image>",
                "fix": "<what the model should do instead, naming colours from the list>"}}]}}
The verdict is "pass" exactly when the problems list is empty.
"""


def _parse_json(text):
    """First {...} block in the model's text, or None."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def call_judge(ink_url, colored_url, api_key, model=JUDGE_MODEL, max_retries=3, timeout=120):
    """Ask the judge model to review a recolour. Both images as data URLs."""
    body = {
        "model": model,
        "modalities": ["text"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": JUDGE_PROMPT},
                {"type": "text", "text": "Image 1 (original):"},
                {"type": "image_url", "image_url": {"url": ink_url}},
                {"type": "text", "text": "Image 2 (recoloured):"},
                {"type": "image_url", "image_url": {"url": colored_url}},
            ],
        }],
    }
    data, attempts, error = _post_with_retry(body, api_key, max_retries, timeout)
    if error:
        return {"success": False, "error": error, "cost_usd": None, "raw": None}
    cost, _ = _cost(data)
    text = _extract_text(_message(data))
    review = _parse_json(text)
    if not isinstance(review, dict) or "verdict" not in review:
        return {"success": False, "error": "judge did not return the JSON asked for",
                "cost_usd": cost, "raw": text}
    problems = [p for p in review.get("problems") or [] if isinstance(p, dict)]
    return {
        "success": True,
        "passed": review["verdict"] == "pass" and not problems,
        "score": review.get("score"),
        "problems": problems,
        "cost_usd": cost,
        "raw": text,
        "error": None,
    }


def redo_prompt(problems):
    """Colour prompt plus the judge's problems with the rejected attempt."""
    lines = "\n".join(f"- {p.get('where', '')}: {p.get('fix', '')}" for p in problems)
    return COLOR_PROMPT + f"""
FEEDBACK ON A PREVIOUS ATTEMPT:
A previous attempt at this image was rejected for these problems. Start again from this
black ink image, follow all rules above, and avoid these problems:
{lines}

Return the recoloured image as an image output.
"""


def recolor_with_judge(ink_img, api_key, out_dir, pattern_id, max_redo=MAX_REDO):
    out_dir = Path(out_dir)
    attempt_dir = out_dir / f"{pattern_id}" 
    attempt_dir.mkdir(parents=True, exist_ok=True)
    ink_url = from_img_to_url(ink_img)

    history, problems = [], []
    for n in range(max_redo + 1):
        prompt = redo_prompt(problems) if problems else COLOR_PROMPT
        call = call_image(ink_url, api_key, attempt_dir / f"{pattern_id}_{n}.png", prompt=prompt)
        record = {"attempt": n, "path": call["output_path"], "cost_usd": call["cost_usd"],
                  "feedback_given": problems, "error": call["error"]}
        history.append(record)
        if not call["success"]:
            print(f"{pattern_id} attempt {n}: colour agent failed: {call['error']}")
            continue    # nothing to judge; the next attempt reuses the same feedback

        colored_url = from_img_to_url(Image.open(call["output_path"]).convert("RGB"))
        judge = call_judge(ink_url, colored_url, api_key)
        record["judge"] = judge
        if not judge["success"]:
            print(f"{pattern_id} attempt {n}: judge failed: {judge['error']}")
            continue    # no review; redo with the previous feedback
        print(f"{pattern_id} attempt {n}: {'pass' if judge['passed'] else 'redo'}, "
              f"score {judge['score']}, {len(judge['problems'])} problems")
        if judge["passed"]:
            break
        problems = judge["problems"]

    judged = [r for r in history if (r.get("judge") or {}).get("success")]
    best = max(judged, key=lambda r: (r["judge"]["passed"], r["judge"]["score"] or 0),
               default=None)
    final_path = None
    if best is not None:
        final_path = out_dir / f"{pattern_id}{Path(best['path']).suffix}"
        shutil.copyfile(best["path"], final_path)
    costs = [r["cost_usd"] for r in history] + \
            [r["judge"]["cost_usd"] for r in history if r.get("judge")]
    result = {
        "pattern_id": pattern_id,
        "output_path": None if final_path is None else str(final_path),
        "passed": bool(best and best["judge"]["passed"]),
        "score": None if best is None else best["judge"]["score"],
        "chosen_attempt": None if best is None else best["attempt"],
        "n_attempts": len(history),
        "cost_usd": sum(c for c in costs if c is not None),
        "history": history,
    }
    (attempt_dir / f"{pattern_id}_log.json").write_text(json.dumps(result, indent=2, default=str))
    return result
