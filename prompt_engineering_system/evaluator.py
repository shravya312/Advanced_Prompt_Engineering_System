from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import time
import pandas as pd

class PromptEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        self.metrics_history = []

    def evaluate(self, reference: str, candidate: str, start_time: float, cost: float = 0.0):
        rouge_scores = self.rouge.score(reference, candidate)
        bleu_score = self.bleu.sentence_score(candidate, [reference]).score
        response_time = time.time() - start_time
        evaluation = {
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure,
            "bleu": bleu_score,
            "response_time": response_time,
            "cost": cost
        }
        self.metrics_history.append(evaluation)
        return evaluation

    def get_metrics_summary(self):
        if not self.metrics_history:
            return {}
        df = pd.DataFrame(self.metrics_history)
        return {
            "avg_rouge1": df["rouge1"].mean(),
            "avg_rouge2": df["rouge2"].mean(),
            "avg_rougeL": df["rougeL"].mean(),
            "avg_bleu": df["bleu"].mean(),
            "avg_response_time": df["response_time"].mean(),
            "avg_cost": df["cost"].mean()
        } 