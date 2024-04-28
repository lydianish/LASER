# MTEB: Massive Text Embedding Benchmark

Evaluating LASER models on tasks from the [MTEB](https://github.com/embeddings-benchmark/mteb).

Usage:
```bash
bash run_mteb.sh --model $MODEL_PATH --vocab $VOCAB_PATH --tokenizer $TOKENIZER --output-dir $OUTPUT_DIR --verbose --english-only
```

Default parameters:
```bash
MODEL_PATH=$LASER/models/laser2.pt
VOCAB_PATH=$LASER/models/laser2.cvocab
TOKENIZER=spm
OUTPUT_DIR=./mteb_scores
```

**Note:** Set the `$LASER` environment variable before running the script.

Selecting MTEB tasks: 
- use the `--english-only` flag to select all the **22 tasks** for English in the `s2s` category, under the types `PairClassification`, `Classification` and `STS`. See [MTEB list of tasks](https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md).
- use the `--ugc-only` flag to select the following 4 tasks for evaluating on user-generated content (Twitter): `TweetSentimentExtractionClassification`, `TwitterSemEval2015` and `TwitterURLCorpus`, plus one STS task: `STSBenchmark`. (For the paper [Making Sentence Embeddings Robust to User-Generated Content](https://arxiv.org/abs/2403.17220) by Nishimwe et al., 2024)

- without any flag, the task selected by default is `BUCC` (multilingual).



