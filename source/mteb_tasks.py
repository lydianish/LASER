import argparse, os, json
from mteb import MTEB
from embed import SentenceEncoder
import sentencepiece as spm
from lib.text_processing import PreprocessLine
from lib.custom_tokenizers import CustomTokenizer, SUPPORTED_TOKENIZERS
import numpy as np

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STSBenchmark",
]

TASK_LIST_BITEXT_MINING = [
    "BUCC"
]

TASK_LIST_S2S_ENGLISH = (
    TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_CLASSIFICATION
    + TASK_LIST_STS
)

class CustomModel(SentenceEncoder):
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        vocab=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
        tokenizer="spm"
    ):
        super().__init__(
            model_path=model_path, 
            max_sentences=max_sentences, 
            max_tokens=max_tokens,
            vocab=vocab,
            cpu=cpu,
            fp16=fp16,
            verbose=verbose,
            sort_kind=sort_kind,
        )
        self.tokenizer = CustomTokenizer(tokenizer)

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        preprocessed_sentences = [ PreprocessLine(s) for s in sentences ]
        tokenized_sentences = [ self.tokenizer.tokenize(s) for s in preprocessed_sentences ]
        embeddings = super().encode_sentences(tokenized_sentences)
        return embeddings

def average_scores(output_dir):
    print("## Averaging all pair classification scores...")
    pair_classification_scores = {}
    cumul_score = 0
    n_tasks = 0
    for task in TASK_LIST_PAIR_CLASSIFICATION:
        score_file = os.path.join(output_dir, f"{task}.json")
        if os.path.exists(score_file):
            with open(os.path.join(output_dir, f"{task}.json")) as f:
                task_scores = json.load(f)
            pair_classification_scores[task] = task_scores["test"]["cos_sim"]["ap"] * 100
            cumul_score += pair_classification_scores[task]
            n_tasks += 1
    pair_classification_scores["Average"] = cumul_score / n_tasks if n_tasks else 0
    with open(os.path.join(output_dir, "scores_pair_classification.json"), 'w') as f:
        json.dump(pair_classification_scores, f)

    print("## Averaging all classification scores...")
    classification_scores = {}
    cumul_score = 0
    n_tasks = 0
    for task in TASK_LIST_CLASSIFICATION:
        score_file = os.path.join(output_dir, f"{task}.json")
        if os.path.exists(score_file):
            with open(os.path.join(output_dir, f"{task}.json")) as f:
                task_scores = json.load(f)
            if "en" in task_scores["test"]:
                classification_scores[task] = task_scores["test"]["en"]["main_score"] * 100
            else:
                classification_scores[task] = task_scores["test"]["main_score"] * 100
            cumul_score += classification_scores[task]
            n_tasks += 1
    classification_scores["Average"] = cumul_score / n_tasks if n_tasks else 0
    with open(os.path.join(output_dir, "scores_classification.json"), 'w') as f:
        json.dump(classification_scores, f)

    print("## Averaging all STS scores...")
    sts_scores = {}
    cumul_score = 0
    n_tasks = 0
    for task in TASK_LIST_STS:
        score_file = os.path.join(output_dir, f"{task}.json")
        if os.path.exists(score_file):
            with open(os.path.join(output_dir, f"{task}.json")) as f:
                task_scores = json.load(f)
            if "en-en" in task_scores["test"]:
                sts_scores[task] = task_scores["test"]["en-en"]["cos_sim"]["spearman"] * 100
            else:
                sts_scores[task] = task_scores["test"]["cos_sim"]["spearman"] * 100
            cumul_score += sts_scores[task]
            n_tasks += 1
    sts_scores["Average"] = cumul_score / n_tasks if n_tasks else 0
    with open(os.path.join(output_dir, "scores_sts.json"), 'w') as f:
        json.dump(sts_scores, f)

    print("## Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Evaluate on MTEB benchmarks")
    parser.add_argument(
        "--encoder", type=str, required=True, help="encoder to be used"
        )
    parser.add_argument(
        "--vocab", type=str, help="Use specified vocab file for encoding"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="spm", help=f"tokenizer among {SUPPORTED_TOKENIZERS}"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output sentence embeddings"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Detailed output"
    )
    parser.add_argument(
        "--english-only", action="store_true", help="Evaluate on tasks that require English-only encoder"
    )
    parser.add_argument(
        "--ugc-only", action="store_true", help="Evaluate on UGC tasks that require English-only encoder"
    )
    
    args = parser.parse_args()
    
    assert args.tokenizer in SUPPORTED_TOKENIZERS, f"The tokenizer {args.tokenizer} is unknown. Expected values are {SUPPORTED_TOKENIZERS}."
        
    if args.english_only:
        evaluation = MTEB(task_types=["PairClassification", "Classification", "STS"], task_categories=["s2s"], task_langs=["en"])
    elif args.ugc_only:
        evaluation = MTEB(tasks=["TweetSentimentExtractionClassification", "TwitterSemEval2015", "TwitterURLCorpus", "STSBenchmark"])
    else:
        evaluation = MTEB(tasks=["BUCC"])
        
    
    model = CustomModel(args.encoder, vocab=args.vocab, verbose=args.verbose, tokenizer=args.tokenizer)
    evaluation.run(model, output_folder=args.output_dir)
    average_scores(args.output_dir)