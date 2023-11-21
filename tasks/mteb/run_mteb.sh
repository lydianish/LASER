#!/bin/bash

# TO DO

python $LASER/source/mteb_tasks.py \
    --encoder $MODEL_PATH \
    --vocab $VOCAB_PATH \
    --tokenizer $TOKENIZER \
    --output-dir $OUTPUT_DIR \
    --verbose \
    --english-only