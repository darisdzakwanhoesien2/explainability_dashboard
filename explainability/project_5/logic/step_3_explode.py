import pandas as pd
from .step_2_parse import parse_esg_json

def explode_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["parsed"] = df["text"].apply(parse_esg_json)

    exploded = df.explode("parsed", ignore_index=True)
    parsed_df = pd.json_normalize(exploded["parsed"])

    meta_cols = [c for c in df.columns if c != "parsed"]
    meta = exploded[meta_cols].reset_index(drop=True)

    return pd.concat([meta, parsed_df], axis=1)
