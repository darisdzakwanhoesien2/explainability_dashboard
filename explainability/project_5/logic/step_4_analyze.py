def model_completeness(df_pdf, df_page):
    expected = sorted(df_pdf["model"].dropna().unique())
    present = sorted(df_page["model"].dropna().unique())
    missing = set(expected) - set(present)

    score = len(present) / len(expected) if expected else 1.0

    return {
        "expected": expected,
        "present": present,
        "missing": sorted(missing),
        "score": score,
    }
