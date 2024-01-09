from pathlib import Path
from datetime import datetime
from trl import PPOConfig

REWARD_MULTIPLIER = 0.2
SEED = 2345


# ----- REVIEWS TRAININGS -----
RV_MODEL = "Zohar/distilgpt2-finetuned-restaurant-reviews"
RV_REWARD_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"
RV_DATASET = "yelp_review_full"
RV_LABEL = "POS"
RV_RESULT_FILE = Path("results") / "review" / (
    "RV" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    )

RV_DATA_CONFIG = {
    "text_column": "text",
    "label_column": "label",
    "start_review_words": 5,
    "min_text_len": 80,
    "max_text_len": 120,
    "max_review_value": 4,
    "min_review_value": 0,
    "test_train_split": 0.85,
}

RV_GENERATION_KWARGS = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "max_new_tokens": 25,
}

RV_SENTIMENT_KWARGS = {
    "top_k": None,
    "function_to_apply": "softmax",
    "batch_size": 1,
}

RV_PPO_CONFIG = PPOConfig(
    batch_size=1,
    learning_rate=1e-6,
    seed=SEED,
    log_with="wandb",
    task_name="review generation",
    model_name=RV_MODEL,
    query_dataset=RV_DATASET,
    reward_model=RV_REWARD_MODEL,
    tracker_project_name="trl-review",
)


# ----- MACHINE TRANSLATION TRAININGS -----
MT_DATA_FILE = "data/processed/translations.csv"
MT_MODEL = "Helsinki-NLP/opus-mt-pl-en"
MT_TEST_SPLIT = 0.6
MT_RESULT_FILE = Path("results") / "translation" / (
    "MT" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    )

MT_PPO_CONFIG = PPOConfig(
    batch_size=1,
    learning_rate=1e-6,
    seed=SEED,
    log_with="wandb",
    task_name="machine translation",
    model_name=MT_MODEL,
    query_dataset="opus-100-corpus",
    reward_model="bleu + jaccard",
    tracker_project_name="trl-translation",
)
