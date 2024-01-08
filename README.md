# USD - Transformer Reinforcement Learning with trl library

## Objective
Check out the trl library (https://github.com/huggingface/trl). Use it to train at least 2 different pre-trained models (if possible, used in different problems). Compare the effectiveness (on a small example) of using a pre-trained classifier with "manual" evaluation of examples while training the model.


## Dependencies
 - Python 3.10.11
 - modules listed in `requirements.txt`

 
## Usage instructions
There are 2 types of training available: translation (for polish-english sentence translation) and review (generating positive reviews). You can run training with model (default) or human rewards. Training stats are tracked by wandb.

> python run_training.py -t "translation" (or "review") [-e] [-hr] [-v]
- -t --type: "translation" or "review"
- -e --epochs: int, number of training epochs (default 1)
- -hr --human: run with human feedback (default False)
- -v --verbose: debug prints on (default False)

## Models and datasets

### Translation task
- **Trained model**: [Helsinki-NLP/opus-mt-pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en)
- **Reward model**: 1/2 [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu) + 1/2 Jaccard index
- **Dataset**: [opus-100-corpus](https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-pl/)

### Review task
- **Trained model**: [Zohar/distilgpt2-finetuned-restaurant-reviews](https://huggingface.co/Zohar/distilgpt2-finetuned-restaurant-reviews)
- **Reward model**: [finiteautomata/bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)
- **Dataset**: [yelp_review_full](https://huggingface.co/datasets/yelp_review_full)