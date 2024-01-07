from trl import PPOConfig

REV_MODEL = "gpt2"
REV_SEED = 2345
REV_TEST_SPLIT = 0.99
REV_OUTPUT_MIN_LEN = 10
REV_OUTPUT_MAX_LEN = 24

# ----- REVIEWS TRAININGS -----
REV_PPO_CONFIG = PPOConfig(
    learning_rate=1.41e-5,
    log_with="wandb",
    task_name="review generation",
    model_name="gpt2",
    query_dataset="yelp_review_full",
    reward_model="yelpBERT",
    tracker_project_name="trl-review",
)


# ----- MACHINE TRANSLATION TRAININGS -----
MT_DATA_FILE = "data/processed/translations.csv"
MT_MODEL = "Helsinki-NLP/opus-mt-pl-en"
MT_SEED = 2345
MT_REWARD_MULTIPLIER = 0.2
MT_TEST_SPLIT = 0.8

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
