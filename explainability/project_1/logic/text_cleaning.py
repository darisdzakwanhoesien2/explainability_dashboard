def clean_bag_of_phrases(text):
    if isinstance(text, list):
        return " ".join(map(str, text))
    elif isinstance(text, str):
        text = text.replace("'", "").replace('"', "")
        return "".join(c for c in text if c.isalnum() or c.isspace())
    return str(text)
