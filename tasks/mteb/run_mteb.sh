#!/bin/bash
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
# bash script to evaluate LASER models on MTEB tasks
# From the paper "Making Sentence Embeddings Robust to User-Generated Content" by Nishimwe et al., 2024

if [ -z ${LASER} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit 1
fi

# Initialize variables with default values
MODEL_PATH=$LASER/models/laser2.pt
VOCAB_PATH=$LASER/models/laser2.cvocab
TOKENIZER=spm
OUTPUT_DIR=./mteb_scores
VERBOSE=""
ENGLISH_ONLY=""
UGC_ONLY=""

# Set options for the getopt command
options=$(getopt -o "" -l "model:,vocab:,tokenizer:,output-dir:,verbose,english-only,ugc-only" -- "$@")
if [ $? -ne 0 ]; then
    echo "Invalid arguments."
    exit 1
fi
eval set -- "$options"

# Read the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift;;
        --vocab) VOCAB_PATH="$2"; shift;;
        --tokenizer) TOKENIZER="$2"; shift;;
        --output-dir) OUTPUT_DIR="$2"; shift;;
        --verbose) VERBOSE="--verbose";;
        --english-only) ENGLISH_ONLY="--english-only";;
        --ugc-only) UGC_ONLY="--ugc-only";;
        --) shift;;
    esac
    shift
done

python $LASER/source/mteb_tasks.py \
    --encoder $MODEL_PATH \
    --vocab $VOCAB_PATH \
    --tokenizer $TOKENIZER \
    --output-dir $OUTPUT_DIR \
    $VERBOSE $ENGLISH_ONLY $UGC_ONLY