import torch
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from evaluate import load, EvaluationModule
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, create_reference_model, PreTrainedModelWrapper, \
                AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
#import wandb

import src.parameters as param
from src.metrics import translation_reward
from src.prepare_data import prepare_review_dataset, prepare_translation_dataset


class Training:
    def __init__(self, human_feedback: bool = False, debug: bool = False, epochs: int = 1) -> None:
        self.human_feedback: bool = human_feedback
        self.debug: bool = debug
        self.epochs: int = epochs
        self.device: torch.device = self.choose_device()
        self.dataset = self.prepare_dataset()
        self.train_dataset = self.dataset[0]
        self.test_dataset = self.dataset[1]
        self.model: PreTrainedModelWrapper = None

    def choose_device(self) -> torch.device:
        """
        Choose training device. CUDA if available, else CPU.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_dataset(self) -> tuple[str]:
        return ("train", "test")

    def training_loop(self) -> None:
        # encode a query
        # generate model response
        # define reward for a response
        # train model with ppo
        pass

    def generate_reward(self) -> list[torch.Tensor]:
        return [torch.tensor(1.0, device=self.device)]

    def human_reward(self) -> list[torch.Tensor]:
        try:
            reward = param.REWARD_MULTIPLIER * int(input("Reward [0-5]: "))
        except ValueError:
            print("Invalid input. Reward set to 0.")
            reward = 0
        return [torch.tensor(reward, device=self.device)]

    def train(self) -> None:
        #wandb.init()
        for i in range(self.epochs):
            print(f"----- Epoch [{i}|{self.epochs-1}] ------")
            self.training_loop()


class TranslationTraining(Training):
    """
    Wrapper for machine translation training task.
    """
    def __init__(self, human_feedback: bool = False, debug: bool = False, epochs: int = 1) -> None:
        super().__init__(human_feedback, debug, epochs)
        self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(param.MT_MODEL)
        self.model.to(self.device)
        self.model_ref = create_reference_model(self.model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(param.MT_MODEL)
        self.ppo_trainer = PPOTrainer(param.MT_PPO_CONFIG, self.model,
                                      self.model_ref, self.tokenizer)
        self.bleu: EvaluationModule = load("bleu")

    def prepare_dataset(self) -> DataFrame:
        return prepare_translation_dataset(param.MT_DATA_FILE)

    def training_loop(self) -> None:
        for text, translation in zip(self.train_dataset["Polish"], self.train_dataset["English"]):
            # encode a query
            query_tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            if self.human_feedback or self.debug:
                print("\n----------------------------")
                print(f"Source sentence: {text}")
                print(f"Target sentence: {translation}")

            # get model response
            response_tensor = self.model.generate(input_ids=query_tensor)
            result_txt = [self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)]
            if self.debug:
                print(f"Response sentence: {result_txt[0]}")

            # define a reward for response
            if not self.human_feedback:
                reward = translation_reward(result_txt, translation, self.bleu, self.device)
                if self.debug:
                    print(f"Reward: {reward[0].item()}")
            else:
                reward = self.human_reward()

            # train model for one step with ppo
            train_stats = self.ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
            self.ppo_trainer.log_stats(train_stats, {"query": [text],
                                                     "response": result_txt}, reward)

    def evaluate_training(self):
        queries = []
        response_before = []
        response_after = []
        reward_before = []
        reward_after = []

        for text, translation in zip(self.test_dataset["Polish"], self.test_dataset["English"]):
            queries.append(text)

            # Response before training
            query_tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            response_tensor = self.model_ref.generate(input_ids=query_tensor)
            result_txt = [self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)]
            response_before.append(result_txt[0])
            reward_before.append(translation_reward(result_txt, translation,
                                                    self.bleu, self.device)[0].item())

            # Response after
            query_tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            response_tensor = self.model.generate(input_ids=query_tensor)
            result_txt = [self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)]
            response_after.append(result_txt[0])
            reward_after.append(translation_reward(result_txt, translation,
                                                   self.bleu, self.device)[0].item())

        results = DataFrame({"query": queries,
                             "response_before": response_before,
                             "response_after": response_after,
                             "reward_before": reward_before,
                             "reward_after": reward_after})
        results.to_csv(
            Path("results") / "translation" / ("MT" + str(datetime.now()) + ".csv"), index=False
            )


class ReviewTraining(Training):
    """
    Wrapper for positive review training task.
    """
    def __init__(self, human_feedback: bool = False, debug: bool = False, epochs: int = 1) -> None:
        super().__init__(human_feedback, debug, epochs)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(param.RV_MODEL)
        self.model.to(self.device)
        self.model_ref = create_reference_model(self.model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(param.RV_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ppo_trainer = PPOTrainer(param.RV_PPO_CONFIG, self.model,
                                      self.model_ref, self.tokenizer)
        self.pipeline: Pipeline = self.get_pipeline()

    def prepare_dataset(self) -> list[str]:
        return prepare_review_dataset(param.RV_DATASET)

    def get_pipeline(self) -> Pipeline:
        reward_tokenizer = AutoTokenizer.from_pretrained(param.RV_REWARD_MODEL)
        reward_model = AutoModelForSequenceClassification.from_pretrained(param.RV_REWARD_MODEL)
        sentiment_pipe = pipeline("sentiment-analysis", model=reward_model,
                                  device=self.device, tokenizer=reward_tokenizer)
        return sentiment_pipe

    def training_loop(self) -> None:
        for query_txt in self.train_dataset:
            # encode a query
            query_tensor = self.tokenizer.encode(query_txt, return_tensors="pt").to(self.device)
            if self.human_feedback or self.debug:
                print("\n----------------------------")
                print(f"Query: {query_txt}")

            # generate model response
            response_tensor = self.ppo_trainer.generate(list(query_tensor),
                                                        return_prompt=True,
                                                        pad_token_id=self.tokenizer.eos_token_id,
                                                        **param.RV_GENERATION_KWARGS)
            response_txt = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            if self.human_feedback or self.debug:
                print(f"Response: {response_txt}")

            # define a reward for response
            if not self.human_feedback:
                pipe_outputs = self.pipeline(response_txt, **param.RV_SENTIMENT_KWARGS)
                reward = next(
                    val for val in pipe_outputs if val["label"] == param.RV_LABEL
                    )["score"]
                reward = [torch.tensor(reward, device=self.device)]
                if self.debug:
                    print(f"Reward: {reward[0].item()}")
            else:
                reward = self.human_reward()

            # train model for one step with ppo
            train_stats = self.ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
            self.ppo_trainer.log_stats(train_stats, {"query": [query_txt],
                                                     "response": [response_txt]}, reward)

    def evaluate_training(self):
        queries = []
        response_before = []
        response_after = []
        reward_before = []
        reward_after = []

        for text in self.test_dataset:
            queries.append(text)

            # Response before training
            query_tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            response_tensor = self.model_ref.generate(list(query_tensor),
                                                      return_prompt=True,
                                                      pad_token_id=self.tokenizer.eos_token_id,
                                                      **param.RV_GENERATION_KWARGS)
            response_txt = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            response_before.append(response_txt)
            pipe_outputs = self.pipeline(response_txt, **param.RV_SENTIMENT_KWARGS)
            reward = next(val for val in pipe_outputs if val["label"] == param.RV_LABEL)["score"]
            reward_before.append(reward)

            # Response after
            query_tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            response_tensor = self.model.generate(list(query_tensor),
                                                  return_prompt=True,
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  **param.RV_GENERATION_KWARGS)
            response_txt = self.tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            response_after.append(response_txt)
            pipe_outputs = self.pipeline(response_txt, **param.RV_SENTIMENT_KWARGS)
            reward = next(val for val in pipe_outputs if val["label"] == param.RV_LABEL)["score"]
            reward_after.append(reward)

        results = DataFrame({"query": queries,
                             "response_before": response_before,
                             "response_after": response_after,
                             "reward_before": reward_before,
                             "reward_after": reward_after})
        results.to_csv(
            Path("results") / "review" / ("RV" + str(datetime.now()) + ".csv"), index=False
            )
