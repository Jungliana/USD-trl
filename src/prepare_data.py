import pandas as pd
import pathlib
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def build_dataset(config, dataset_name="yelp_review_full", input_min_text_length=3, input_max_text_length=6):
    """
    Build dataset for training. This builds the dataset from `load_dataset`.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # Define a custom function to convert ratings to True or False
    def convert_labels(example):
        example["label"] = 1 if example["label"] > 2 else 0
        return example

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="test")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: x["label"] != 2, batched=False)
    ds = ds.filter(lambda x: len(x["review"]) > 60, batched=False)
    ds = ds.filter(lambda x: len(x["review"]) < 100, batched=False)
    ds = ds.map(convert_labels)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def process_translations():
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
    translations = translations.loc[translations['Polish'].str.len() > 50]
    translations = translations.loc[translations['Polish'].str.len() < 80]

    translations.to_csv(write_path, index=False)
