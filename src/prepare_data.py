import pandas as pd
import pathlib

# Data source: https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-pl/


def process_translations():
    path = pathlib.Path(__file__)
    read_pl_path = path.parents[2] / "data" / "raw" / "opus-en-pl-dev-pl.txt"
    read_en_path = path.parents[2] / "data" / "raw" / "opus-en-pl-dev-en.txt"
    write_path = path.parents[2] / "data" / "processed" / "translations.csv"

    with open(read_pl_path, encoding="utf8") as file:
        lines_pl = [line.rstrip().lstrip() for line in file]
    with open(read_en_path, encoding="utf8") as file:
        lines_en = [line.rstrip() for line in file]
    translations = pd.DataFrame({"Polish": lines_pl, "English": lines_en})
    translations = translations.loc[translations['Polish'].str.len() > 50]
    translations = translations.loc[translations['Polish'].str.len() < 80]

    translations.to_csv(write_path, index=False)
