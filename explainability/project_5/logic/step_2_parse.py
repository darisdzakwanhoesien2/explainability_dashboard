import json
import re

def extract_json_block(text):
    if not isinstance(text, str):
        return None

    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def normalize_json(obj):
    if obj is None:
        return []

    if isinstance(obj, dict):
        return [obj]

    if isinstance(obj, list):
        flat = []
        for item in obj:
            flat.extend(normalize_json(item))
        return flat

    return []


def is_valid_esg_object(d):
    return isinstance(d, dict) and "sentence" in d and "aspect" in d


def parse_esg_json(text):
    raw = extract_json_block(text)
    normalized = normalize_json(raw)
    return [x for x in normalized if is_valid_esg_object(x)]
