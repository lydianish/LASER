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
# Implemenation of RoLASER encoder class
# From the paper "Making Sentence Embeddings Robust to User-Generated Content" by Nishimwe et al., 2024


from embed import SentenceEncoder
from lib.text_processing import PreprocessLine
from lib.custom_tokenizers import CustomTokenizer

class RoLaserEncoder(SentenceEncoder):
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
