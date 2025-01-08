#!/usr/bin/python
# Copyright (c) Lydia Nishimwe.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Python script for evaluating LASER models on the Massive Text Embedding Benchmark
# From the paper "Making Sentence Embeddings Robust to User-Generated Content" by Nishimwe et al., 2024

import argparse, os, json
from mteb import MTEB
from rolaser import RoLaserEncoder
from sentence_transformers import SentenceTransformer
from lib.custom_tokenizers import SUPPORTED_TOKENIZERS

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
        "--encoder", type=str, required=True, help="name or path of encoder to be used"
        )
    parser.add_argument(
        "--huggingface", action="store_true", help="Whether the encoder is a Hugging Face SentenceTransformer model"
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
        "--english-tasks", action="store_true", help="Evaluate on English-only tasks"
    )
    parser.add_argument(
        "--ugc-tasks", action="store_true", help="Evaluate on UGC tasks"
    )
    parser.add_argument(
        "--mining-tasks", action="store_true", help="Evaluate on mining tasks"
    )
    parser.add_argument(
        "--langs", type=str, nargs="+", help="Languages to evaluate on", default=["en"]
    )
    args = parser.parse_args() 
    
    if not args.huggingface:
        assert args.vocab is not None, "Vocab file is required for RoLASER encoder"
        assert args.tokenizer is not None, "Tokenizer is required for RoLASER encoder"
        assert args.tokenizer in SUPPORTED_TOKENIZERS, f"The tokenizer {args.tokenizer} is unknown. Expected values are {SUPPORTED_TOKENIZERS}."

    
    if args.huggingface:
        model = SentenceTransformer(args.encoder)
    else:
        model = RoLaserEncoder(args.encoder, vocab=args.vocab, verbose=args.verbose, tokenizer=args.tokenizer)
    
    evaluations = []
    if args.english_tasks:
        evaluations.append(MTEB(task_types=["PairClassification", "Classification", "STS"], task_categories=["s2s"], task_langs=["en"]))
    if args.ugc_tasks:
        evaluations.append(MTEB(tasks=["TweetSentimentExtractionClassification", "TwitterSemEval2015", "TwitterURLCorpus", "STSBenchmark"]))
    if args.mining_tasks:
        evaluations.append(MTEB(tasks=["BUCC"], task_langs=["fr-en"]))

    for evaluation in evaluations:
        evaluation.run(model, output_folder=args.output_dir)
    
    # average_scores(args.output_dir)