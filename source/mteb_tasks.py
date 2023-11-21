import argparse, os, json
from mteb import MTEB
from embed import SentenceEncoder
import sentencepiece as spm
from lib.text_processing import PreprocessLine
from lib.custom_tokenizers import CustomTokenizer, SUPPORTED_TOKENIZERS
import pandas as pd

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
    pair_classification_scores = pd.DataFrame(columns=["Average"] + TASK_LIST_PAIR_CLASSIFICATION)
    for task in TASK_LIST_PAIR_CLASSIFICATION:
        with open(os.path.join(output_dir, f"{task}.json")) as f:
            task_scores = json.load(f)
        pair_classification_scores.at[0, task] = task_scores["test"]["cos_sim"]["ap"] * 100
    pair_classification_scores["Average"] = pair_classification_scores[TASK_LIST_PAIR_CLASSIFICATION].mean(axis=1)
    pair_classification_scores.to_csv(os.path.join(output_dir, "scores_pair_classification.csv"))

    print("## Averaging all classification scores...")
    classification_scores = pd.DataFrame(columns=["Average"] + TASK_LIST_CLASSIFICATION)
    for task in TASK_LIST_CLASSIFICATION:
        with open(os.path.join(output_dir, f"{task}.json")) as f:
            task_scores = json.load(f)
        if "en" in task_scores["test"]:
            classification_scores.at[0, task] = task_scores["test"]["en"]["main_score"] * 100
        else:
            classification_scores.at[0, task] = task_scores["test"]["main_score"] * 100 
    classification_scores["Average"] = classification_scores[TASK_LIST_CLASSIFICATION].mean(axis=1)
    classification_scores.to_csv(os.path.join(output_dir, "scores_classification.csv"))

    print("## Averaging all STS scores...")
    sts_scores = pd.DataFrame(columns=["Average"] + TASK_LIST_STS)
    for task in TASK_LIST_STS:
        with open(os.path.join(output_dir, f"{task}.json")) as f:
            task_scores = json.load(f)
        if "en-en" in task_scores["test"]:
            sts_scores.at[0, task] = task_scores["test"]["en-en"]["cos_sim"]["spearman"] * 100
        else:
            sts_scores.at[0, task] = task_scores["test"]["cos_sim"]["spearman"] * 100
    sts_scores["Average"] = sts_scores[TASK_LIST_STS].mean(axis=1)
    sts_scores.to_csv(os.path.join(output_dir, "scores_sts.csv"))

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
    
    args = parser.parse_args()
    
    assert args.tokenizer in SUPPORTED_TOKENIZERS, f"The tokenizer {args.tokenizer} is unknown. Expected values are {SUPPORTED_TOKENIZERS}."
        
    if args.english_only:
        evaluation = MTEB(task_types=["PairClassification", "Classification", "STS"], task_categories=["s2s"], task_langs=["en"])
    else:
        evaluation = MTEB(tasks=["BUCC"])
    
    model = CustomModel(args.encoder, vocab=args.vocab, verbose=args.verbose, tokenizer=args.tokenizer)
    evaluation.run(model, output_folder=args.output_dir)
    average_scores(args.output_dir)