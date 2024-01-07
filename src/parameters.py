from trl import PPOConfig

# ----- REVIEWS TRAININGS -----
REV_MODEL = "Zohar/distilgpt2-finetuned-restaurant-reviews"
REV_REWARD_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"
REV_DATASET = "yelp_review_full"
REV_SEED = 2345
REV_OUTPUT_MIN_LEN = 10
REV_OUTPUT_MAX_LEN = 24
REV_REWARD_MULTIPLIER = 0.2

REV_DATA_CONFIG = {
    "start_review_words": 5,
    "min_text_len": 80,
    "max_text_len": 120,
    "max_review_value": 4,
    "min_review_value": 0,
    "test_train_split": 0.99,
    "random_state": 2345,
}

REV_GEN_KWARGS = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "max_new_tokens": 25,
}

REV_SENT_KWARGS = {
    "top_k": None,
    "function_to_apply": "softmax",
    "batch_size": 1,
}

REV_PPO_CONFIG = PPOConfig(
    batch_size=1,
    learning_rate=1.41e-5,
    log_with="wandb",
    task_name="review generation",
    model_name=REV_MODEL,
    query_dataset=REV_DATASET,
    reward_model=REV_REWARD_MODEL,
    tracker_project_name="trl-review",
)


# ----- MACHINE TRANSLATION TRAININGS -----
MT_DATA_FILE = "data/processed/translations.csv"
MT_MODEL = "Helsinki-NLP/opus-mt-pl-en"
MT_SEED = 2345
MT_REWARD_MULTIPLIER = 0.2
MT_TEST_SPLIT = 0.99

MT_PPO_CONFIG = PPOConfig(
    batch_size=1,
    learning_rate=1e-6,
    seed=MT_SEED,
    log_with="wandb",
    task_name="machine translation",
    model_name="opus-mt-pl-en",
    query_dataset="opus-100-corpus",
    reward_model="bleu + jaccard",
    tracker_project_name="trl-translation",
)
