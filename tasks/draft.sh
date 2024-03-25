#!/bin/bash

set -e

CHECKPOINT_DIR=$EXPERIMENTS/robust-embeddings/laser/experiment_036_jz
DATASETS_DIR=$DATASETS/oscar/mini/4Mlrec/bin

MODEL_NAME[0]=laser
MODEL_PATH[0]=$LASER/models/laser2.pt
VOCAB_FILE[0]=$LASER/models/laser2.cvocab
TOKENIZER[0]=$HOME/data-preparation/src/spm-tokenizer.py

MODEL_NAME[1]=roberta-student
MODEL_PATH[1]=$CHECKPOINT_DIR/checkpoints/${MODEL_NAME[1]}/checkpoint_best.pt
VOCAB_FILE[1]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[1]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[2]=roberta-student-init
MODEL_PATH[2]=$CHECKPOINT_DIR/checkpoints/${MODEL_NAME[2]}/checkpoint_best.pt
VOCAB_FILE[2]=$DATASETS_DIR/robertaugc-laserstd/dict.robertaugc.txt
TOKENIZER[2]=$HOME/data-preparation/src/roberta-tokenizer.py

MODEL_NAME[3]=character-roberta-student
MODEL_PATH[3]=$CHECKPOINT_DIR/checkpoints/${MODEL_NAME[3]}/checkpoint_best.pt
VOCAB_FILE[3]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[3]=$HOME/data-preparation/src/char-tokenizer.py

MODEL_NAME[4]=character-roberta-student-init
MODEL_PATH[4]=$CHECKPOINT_DIR/checkpoints/${MODEL_NAME[4]}/checkpoint_best.pt
VOCAB_FILE[4]=$DATASETS_DIR/charobertaugc-laserstd/dict.charobertaugc.txt
TOKENIZER[4]=$HOME/data-preparation/src/char-tokenizer.py

#------------------ FLORES ------------------#

LANG=eng_Latn
EXPERIMENT_DIR=$CHECKPOINT_DIR
CORPUS=flores200
CORPUS_PARTS="devtest"

#------- MULTILEXNORM ------------------#

# LANG=en.ref
# EXPERIMENT_DIR=$CHECKPOINT_DIR
# CORPUS=multilexnorm2021/en
# CORPUS_PARTS="test"

#------------------ ROCSMT ------------------#

# LANG=norm.en
# EXPERIMENT_DIR=$CHECKPOINT_DIR
# CORPUS=rocsmt
# CORPUS_PARTS="test"

for i in 4
do
    for CORPUS_PART in $CORPUS_PARTS
    do
        OUTPUT_DIR=$EXPERIMENT_DIR/embeddings/${MODEL_NAME[$i]}/$CORPUS/$CORPUS_PART
        mkdir -p $OUTPUT_DIR

        INPUT_FILE_NAME=$DATASETS/$CORPUS/$CORPUS_PART/$LANG.$CORPUS_PART
        OUTPUT_EMBED_FILE=$OUTPUT_DIR/$LANG.$CORPUS_PART
        
        python $LASER/source/embed.py \
            --input     $INPUT_FILE_NAME        \
            --encoder   ${MODEL_PATH[$i]}    \
            --custom-vocab-file   ${VOCAB_FILE[$i]}    \
            --custom-tokenizer   ${TOKENIZER[$i]}    \
            --output    $OUTPUT_EMBED_FILE       \
            --verbose
        
    done
done

echo "Done..."