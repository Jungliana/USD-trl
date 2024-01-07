import torch
import evaluate
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, \
                create_reference_model

from src.parameters import MT_MODEL, MT_DATA_FILE, MT_SEED, MT_PPO_CONFIG, \
                           MT_REWARD_MULTIPLIER, MT_TEST_SPLIT
from src.metrics import translation_reward


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
                reward = MT_REWARD_MULTIPLIER * int(input("Reward [0-5]: "))
                reward = [torch.tensor(reward, device=device)]

            # train model for one step with ppo
            train_stats = ppo_trainer.step([query_tensor[0]],
                                           [response_tensor[0]], reward)
            ppo_trainer.log_stats(train_stats, {"query": text,
                                                "response": result_txt},
                                  reward)
