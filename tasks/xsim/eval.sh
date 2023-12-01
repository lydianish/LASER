#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
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
#-------------------------------------------------------
#
# This bash script installs the flores200 dataset, downloads laser2, and then
# performs xsim (multilingual similarity) evaluation with ratio margin

if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

ddir="${LASER}/data"
cd $ddir  # move to data directory

if [ ! -d $ddir/flores200 ] ; then
    echo " - Downloading flores200..."
    wget --trust-server-names -q https://tinyurl.com/flores200dataset
    tar -xf flores200_dataset.tar.gz
    /bin/mv flores200_dataset flores200
    /bin/rm flores200_dataset.tar.gz
else
    echo " - flores200 already downloaded"
fi

mdir="${LASER}/models"
if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - Downloading $s3/${file}";
        wget -q $s3/${file};
    fi 
}

cd $mdir  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

if [ ! -f ${mdir}/laser2.pt ] ; then
    echo " - Downloading $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
else
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm"
download "laser2.cvocab"

corpus="flores200"
corpus_part="devtest"

# languages=""
# for file in $LASER/data/$corpus/$corpus_part/*
# do
#     lang=${file##*/} # remove parent path
#     languages=$languages${lang%.*}, # remove extension
# done
# languages=${languages%,} # remove final comma
# languages=${languages//eng_Latn,/} # remove English from source languages

# note: example evaluation script expects format: basedir/corpus/corpus_part/lang.corpus_part

corpus="rocsmt"
# corpus="flores200/5langs"
corpus_part="test"
ddir=/home/lnishimw/scratch/datasets
languages="ref.fr,ref.de,ref.cs,ref.uk,ref.ru"
# languages="fra_Latn,deu_Latn,ces_Latn,ukr_Cyrl,rus_Cyrl,eng_Latn_ugc"
target_lang="raw.en,norm.en"
# target_lang="eng_Latn_ugc,eng_Latn"
target_model=$EXPERIMENTS/robust-embeddings/laser/experiment_024_jz/checkpoints/roberta-student/checkpoint_best.pt
target_tok=$HOME/data-preparation/src/roberta-tokenizer.py
# target_vocab=$DATASETS/oscar/mini/4M/bin/charobertaugc-charobertastd/dict.charobertaugc.txt
target_vocab=$DATASETS/oscar/mini/4M/bin/robertaugc-laserstd/dict.robertaugc.txt

echo " - calculating xsim"
python3 $LASER/source/eval.py                \
    --base-dir $ddir                         \
    --corpus $corpus                         \
    --corpus-part $corpus_part               \
    --margin ratio                           \
    --src-encoder   $LASER/models/laser2.pt  \
    --src-spm-model $LASER/models/laser2.spm \
    --src-langs $languages      \
    --output-dir /home/lnishimw/scratch/LASER/tasks/xsim/_scores/xsim_lc_5      \
    --tgt-langs $target_lang --verbose \
    --tgt-encoder $target_model \
    --tgt-tokenizer $target_tok \
    --tgt-vocab-file $target_vocab \
     # --cosine-distances \