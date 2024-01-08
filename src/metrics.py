import torch
from evaluate import EvaluationModule


def jaccard(sentence1: str, sentence2: str) -> float:
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    return len(words1.intersection(words2)) / len(words1.union(words2))


def compute_bleu(translation: str, reference: str, bleu: EvaluationModule) -> float:
    results = bleu.compute(predictions=translation, references=[reference])
    return results['bleu']


def translation_reward(translation: str, reference: str,
                       bleu: EvaluationModule, device: torch.device
                       ) -> list[torch.Tensor]:
    bleu_reward = compute_bleu(translation, reference, bleu)
    jaccard_reward = jaccard(translation[0], reference)
    return [torch.tensor(0.5 * bleu_reward + 0.5 * jaccard_reward, device=device)]
