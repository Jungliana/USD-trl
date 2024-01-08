import pandas as pd
import pathlib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.parameters import RV_DATA_CONFIG as CONFIG, MT_DATA_FILE, SEED, MT_TEST_SPLIT


def prepare_translation_dataset(filepath: pathlib.Path = MT_DATA_FILE) -> pd.DataFrame:
    translations = pd.read_csv(filepath)
    train_dataset, _ = train_test_split(translations, test_size=MT_TEST_SPLIT, random_state=SEED)
    return train_dataset


def prepare_review_dataset(dataset_name: str = "yelp_review_full") -> list[str]:
    def cut(sample):
        sample["query"] = " ".join(sample["text"].split()[: CONFIG["start_review_words"]])
        return sample

    dataset = load_dataset(dataset_name, split="test")
    dataset = dataset.filter(lambda x: len(x["text"]) > CONFIG["min_text_len"], batched=False)
    dataset = dataset.filter(lambda x: len(x["text"]) < CONFIG["max_text_len"], batched=False)
    dataset = dataset.filter(lambda x: x["label"] < CONFIG["max_review_value"], batched=False)
    dataset = dataset.filter(lambda x: x["label"] > CONFIG["min_review_value"], batched=False)
    dataset = dataset.map(cut, batched=False)
    train_ds, _ = train_test_split(dataset, test_size=CONFIG["test_train_split"],
                                   random_state=SEED)
    return train_ds["query"]


def process_translations() -> None:
    # Data source: https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-pl/
    path = pathlib.Path(__file__)
    read_pl_path = path.parents[2] / "data" / "raw" / "opus-en-pl-dev-pl.txt"
    read_en_path = path.parents[2] / "data" / "raw" / "opus-en-pl-dev-en.txt"
    write_path = path.parents[2] / "data" / "processed" / "translations.csv"

    with open(read_pl_path, encoding="utf8") as file:
        lines_pl = [line.rstrip().lstrip() for line in file]
    with open(read_en_path, encoding="utf8") as file:
        lines_en = [line.rstrip() for line in file]
    translations = pd.DataFrame({"Polish": lines_pl, "English": lines_en})
    translations = translations.loc[translations["Polish"].str.len() > 50]
    translations = translations.loc[translations["Polish"].str.len() < 80]

    translations.to_csv(write_path, index=False)
