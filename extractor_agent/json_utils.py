"""
Shared JSON-parsing helpers for the extractor_agent pipeline.

LLMs occasionally wrap JSON in markdown code fences, prepend commentary, or
return truncated/malformed JSON. `extract_entities.py` already had a small
`safe_parse_json` to cope with this; this module lifts that logic out so
`extract_relations.py` (and anything else in the package) can reuse it
instead of doing a bare `json.loads()` that dies on the first malformed
response. Mirrors the fence-stripping/repair-retry pattern already used in
`extraction/service.py`'s `call_llm_json`.
"""

import json
import re


def strip_fences(text: str) -> str:
    """
    Remove ```json ... ``` / ``` ... ``` markdown code fences from LLM output.

    Handles fences anchored at the start/end of the string as well as a
    fenced block preceded by leading commentary (e.g. "Sure, here you go:\n```json\n...\n```").
    """
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_parse_json(text: str):
    """
    Parse JSON out of raw LLM output.

    Strips markdown fences first, then falls back to extracting the
    outermost {...} or [...] span in case the model added leading/trailing
    commentary around the JSON.
    """
    cleaned = strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch) + 1
        if start != -1 and end != 0:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                continue

    # Nothing salvageable — raise on the cleaned text so the caller sees
    # the real parse error rather than a confusing fallback message.
    return json.loads(cleaned)


def parse_json_with_repair(call_llm, prompt: str, *, max_retries: int = 2):
    """
    Call an LLM and guarantee the response parses as JSON.

    1. Send `prompt`, try `safe_parse_json` on the response.
    2. On failure, send a repair prompt containing the broken output and
       the exact parse error, asking the model to fix it.
    3. Repeat up to `max_retries` times, then re-raise the last error.

    Args:
        call_llm: callable(prompt: str) -> str. Makes one LLM call and
            returns the raw text response (no client/session state needed).
        prompt: the original extraction prompt.
        max_retries: how many repair attempts to allow.

    Returns:
        The parsed JSON object (dict or list, depending on the schema
        the prompt asked for).
    """
    raw_text = call_llm(prompt)
    last_error: json.JSONDecodeError | None = None

    try:
        return safe_parse_json(raw_text)
    except json.JSONDecodeError as exc:
        last_error = exc

    for _ in range(max_retries):
        repair_prompt = (
            "The following text was supposed to be valid JSON but it is "
            "malformed, truncated, or wrapped in markdown fences.\n\n"
            f"--- ERROR ---\n{last_error}\n\n"
            f"--- BROKEN OUTPUT ---\n{raw_text}\n\n"
            "Return ONLY the corrected, complete, valid JSON. "
            "Do NOT wrap it in markdown code fences. Do NOT add any commentary."
        )
        raw_text = call_llm(repair_prompt)
        try:
            return safe_parse_json(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc

    raise last_error
