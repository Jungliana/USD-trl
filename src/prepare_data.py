import pathlib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.parameters import RV_DATA_CONFIG as CONFIG, MT_DATA_FILE, SEED, MT_TEST_SPLIT


def prepare_translation_dataset(filepath: pathlib.Path = MT_DATA_FILE) -> pd.DataFrame:
    translations = pd.read_csv(filepath)
    train_dataset, test_dataset = train_test_split(
        translations, test_size=MT_TEST_SPLIT, random_state=SEED
        )
    return train_dataset, test_dataset


def prepare_review_dataset(dataset_name: str = "yelp_review_full") -> list[str]:
    def cut(sample):
        sample["query"] = " ".join(
            sample[CONFIG["text_column"]].split()[: CONFIG["start_review_words"]]
            )
        return sample

    dataset = load_dataset(dataset_name, split="test")
    dataset = dataset.filter(
        lambda x: len(x[CONFIG["text_column"]]) > CONFIG["min_text_len"], batched=False
        )
    dataset = dataset.filter(
        lambda x: len(x[CONFIG["text_column"]]) < CONFIG["max_text_len"], batched=False
        )
    dataset = dataset.filter(
        lambda x: x[CONFIG["label_column"]] < CONFIG["max_review_value"], batched=False
        )
    dataset = dataset.filter(
        lambda x: x[CONFIG["label_column"]] > CONFIG["min_review_value"], batched=False
        )
    dataset = dataset.map(cut, batched=False)
    train_ds, test_ds = train_test_split(dataset, test_size=CONFIG["test_train_split"],
                                         random_state=SEED)
    return train_ds["query"], test_ds["query"]
