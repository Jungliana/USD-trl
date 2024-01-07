import torch
import evaluate
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                         pipeline
from trl import PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, \
                AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler

from src.parameters import MT_MODEL, MT_DATA_FILE, MT_SEED, MT_PPO_CONFIG, \
                           MT_REWARD_MULTIPLIER, MT_TEST_SPLIT, REV_MODEL, \
                           REV_PPO_CONFIG, REV_TEST_SPLIT, REV_SEED, \
                           REV_OUTPUT_MIN_LEN, REV_OUTPUT_MAX_LEN
from src.metrics import translation_reward
from src.prepare_data import build_dataset, collator


def mt_training(human_feedback: bool = False,
                debug: bool = False,
                num_epochs: int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    translations = pd.read_csv(MT_DATA_FILE)
    train_dataset, test_dataset = train_test_split(translations,
                                                   test_size=MT_TEST_SPLIT,
                                                   random_state=MT_SEED)

    # get models
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(MT_MODEL).to(device)
    model_ref = create_reference_model(model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)

    # initialize trainer
    ppo_config = MT_PPO_CONFIG

    # create a ppo trainer
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    # reward model
    bleu = evaluate.load("bleu")

    wandb.init()
    for epoch in range(num_epochs):
        print(f'--- Epoch {epoch} of {num_epochs}\n ---')
        for text, translation in zip(train_dataset['Polish'],
                                     train_dataset['English']):
            if human_feedback or debug:
                print("\n----------------------------")
                print(f'Source sentence: {text}')
                print(f'Target sentence: {translation}')
            # encode a query
            query_tensor = tokenizer.encode(text,
                                            return_tensors="pt").to(device)

            # get model response
            response_tensor = model.generate(input_ids=query_tensor)
            result_txt = [tokenizer.decode(response_tensor[0],
                                           skip_special_tokens=True)]
            if debug:
                print(f'Response sentence: {result_txt[0]}')

            # define a reward for response
            if not human_feedback:
                reward = translation_reward(result_txt, translation, bleu, device)
                if debug:
                    print(f'Reward: {reward[0].item()}')
            else:
                try:
                    reward = MT_REWARD_MULTIPLIER * int(input("Reward [0-5]: "))
                except ValueError:
                    print("Invalid input. Reward set to 0.")
                    reward = 0
                reward = [torch.tensor(reward, device=device)]

            # train model for one step with ppo
            train_stats = ppo_trainer.step([query_tensor[0]],
                                           [response_tensor[0]], reward)
            ppo_trainer.log_stats(train_stats,
                                  {"query": text,
                                   "response": result_txt},
                                  reward)


def review_training(human_feedback: bool = False,
                    debug: bool = False,
                    num_epochs: int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(REV_PPO_CONFIG)
    train_dataset, test_dataset = train_test_split(dataset,
                                                   test_size=REV_TEST_SPLIT,
                                                   random_state=REV_SEED)

    # get models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(REV_MODEL)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(REV_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(REV_MODEL)

    tokenizer.pad_token = tokenizer.eos_token

    # initialize trainer
    ppo_config = REV_PPO_CONFIG

    # create a ppo trainer
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, 
                             dataset=dataset, data_collator=collator)

    # reward model
    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", model_max_length=256)
    reward_model = AutoModelForSequenceClassification.from_pretrained("../models/yelpBERT")

    sentiment_pipe = pipeline("sentiment-analysis", model=reward_model, device=device, tokenizer=reward_tokenizer)

    wandb.init()
    output_length_sampler = LengthSampler(REV_OUTPUT_MIN_LEN, REV_OUTPUT_MAX_LEN)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
