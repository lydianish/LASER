#!/usr/bin/python3
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
# --------------------------------------------------------
#
# Tool to calculate multilingual similarity error rate
# on various predefined test sets


import os
import argparse
import pandas
import tempfile
import numpy as np
from pathlib import Path
import itertools
import logging
import sys
from typing import List, Tuple, Dict
from tabulate import tabulate
from collections import defaultdict
from xsim import xSIM, _load_embeddings
from embed import embed_sentences, load_model
from sklearn.metrics.pairwise import paired_cosine_distances

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval")


class Eval:
    def __init__(self, args):
        self.base_dir = args.base_dir
        self.corpus = args.corpus
        self.split = args.corpus_part
        self.min_sents = args.min_sents
        self.index_comparison = args.index_comparison
        self.emb_dimension = args.embedding_dimension
        self.encoder_args = {
            k: v
            for k, v in args._get_kwargs()
            if k in ["max_sentences", "max_tokens", "cpu", "sort_kind", "verbose"]
        }
        self.src_bpe_codes = args.src_bpe_codes
        self.tgt_bpe_codes = args.tgt_bpe_codes
        self.src_spm_model = args.src_spm_model
        self.tgt_spm_model = args.tgt_spm_model
        self.src_tokenizer = args.src_tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.src_vocab_file = args.src_vocab_file
        self.tgt_vocab_file = args.tgt_vocab_file

        logger.info("loading src encoder")
        self.src_encoder = load_model(
            args.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
            custom_vocab_file=self.src_vocab_file,
            hugging_face=args.use_hugging_face,
            **self.encoder_args,
        )
        if args.tgt_encoder:
            logger.info("loading tgt encoder")
            self.tgt_encoder = load_model(
                args.tgt_encoder,
                self.tgt_spm_model,
                self.tgt_bpe_codes,
                custom_vocab_file=self.tgt_vocab_file,
                hugging_face=args.use_hugging_face,
                **self.encoder_args,
            )
        else:
            logger.info("encoding tgt using src encoder")
            self.tgt_encoder = self.src_encoder
            self.tgt_bpe_codes = self.src_bpe_codes
            self.tgt_spm_model = self.src_spm_model
            self.tgt_tokenizer = self.src_tokenizer
            self.tgt_vocab_file = self.src_vocab_file
        self.nway = args.nway
        self.buffer_size = args.buffer_size
        self.fp16 = args.fp16
        self.margin = args.margin
        self.output_dir = args.output_dir
        self.do_pretok = args.do_pretok

    def _embed(
        self, tmpdir, langs, encoder, spm_model, bpe_codes, tgt_aug_langs=[], tokenizer=None, vocab_file=None
    ) -> List[List[str]]:
        emb_data = []
        for lang in langs:
            augjson = None
            fname = f"{lang}.{self.split}"
            infile = self.base_dir / self.corpus / self.split / fname
            assert infile.exists(), f"{infile} does not exist"
            outfile = tmpdir / fname
            if lang in tgt_aug_langs:
                fname = f"{lang}_augmented.{self.split}"
                fjname = f"{lang}_errtype.{self.split}.json"
                augment_dir = self.base_dir / self.corpus / (self.split + "_augmented")
                augjson = augment_dir / fjname
                auginfile = augment_dir / fname
                assert augjson.exists(), f"{augjson} does not exist"
                assert auginfile.exists(), f"{auginfile} does not exist"
                combined_infile = tmpdir / f"combined_{lang}"
                with open(combined_infile, "w") as newfile:
                    for f in [infile, auginfile]:
                        with open(f) as fin:
                            newfile.write(fin.read())
                infile = combined_infile
            embed_sentences(
                str(infile),
                str(outfile),
                encoder=encoder,
                spm_model=spm_model,
                custom_tokenizer=tokenizer,
                custom_vocab_file=vocab_file,
                bpe_codes=bpe_codes,
                token_lang=lang if bpe_codes or self.do_pretok else "--",
                buffer_size=self.buffer_size,
                fp16=self.fp16,
                **self.encoder_args,
            )
            assert (
                os.path.isfile(outfile) and os.path.getsize(outfile) > 0
            ), f"Error encoding {infile}"
            emb_data.append([lang, infile, outfile, augjson])
        return emb_data

    def _xsim(
        self, src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt, augjson=None
    ) -> Tuple[int, int, Dict[str, int]]:
        return xSIM(
            src_emb,
            tgt_emb,
            margin=self.margin,
            dim=self.emb_dimension,
            fp16=self.fp16,
            eval_text=tgt_txt if not self.index_comparison else None,
            augmented_json=augjson,
        )

    def calc_xsim(
        self, embdir, src_langs, tgt_langs, tgt_aug_langs, err_sum=0, totl_nbex=0
    ) -> None:
        outputs = []
        src_emb_data = self._embed(
            embdir,
            src_langs,
            self.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
            tokenizer=self.src_tokenizer,
            vocab_file=self.src_vocab_file
        )
        tgt_emb_data = self._embed(
            embdir,
            tgt_langs,
            self.tgt_encoder,
            self.tgt_spm_model,
            self.tgt_bpe_codes,
            tgt_aug_langs,
            tokenizer=self.tgt_tokenizer,
            vocab_file=self.tgt_vocab_file
        )
        aug_df = defaultdict(lambda: defaultdict())
        combs = list(itertools.product(src_emb_data, tgt_emb_data))
        for (src_lang, _, src_emb, _), (tgt_lang, tgt_txt, tgt_emb, augjson) in combs:
            if src_lang == tgt_lang:
                continue
            err, nbex, aug_report = self._xsim(
                src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt, augjson
            )
            result = round(100 * err / nbex, 2)
            if tgt_lang in tgt_aug_langs:
                aug_df[tgt_lang][src_lang] = aug_report
            if nbex < self.min_sents:
                result = "skipped"
            else:
                err_sum += err
                totl_nbex += nbex
            outputs.append(
                [self.corpus, f"{src_lang}-{tgt_lang}", f"{result}", f"{nbex}"]
            )
        outputs.append(
            [
                self.corpus,
                "average",
                f"{round(100 * err_sum / totl_nbex, 2)}",
                f"{len(combs)}",
            ]
        )
        df = pandas.DataFrame(outputs, columns=[
                    "dataset",
                    "src-tgt",
                    "xsim" + ("(++)" if tgt_aug_langs else ""),
                    "nbex",
                ])
        if self.output_dir:
            df.to_csv(os.path.join(self.output_dir, ("xsimpp" if tgt_aug_langs else "xsim") + "_matrix.csv"))
        print(
            tabulate(
                df,
                tablefmt="psql",
                headers=df.columns
            )
        )
        for tgt_aug_lang in tgt_aug_langs:
            df = pandas.DataFrame.from_dict(aug_df[tgt_aug_lang]).fillna(0).T
            if self.output_dir:
                df.to_csv(os.path.join(self.output_dir, "xsimpp_errortype_matrix.csv"))
            print(
                f"\nAbsolute error under augmented transformations for: {tgt_aug_lang}"
            )
            print(f"{tabulate(df, df.columns, floatfmt='.2f', tablefmt='grid')}")

    def calc_xsim_nway(self, embdir, langs) -> None:
        err_matrix = np.zeros((len(langs), len(langs)))
        emb_data = self._embed(
            embdir,
            langs,
            self.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
            tokenizer=self.src_tokenizer,
            vocab_file=self.src_vocab_file
        )
        for i1, (src_lang, _, src_emb, _) in enumerate(emb_data):
            for i2, (tgt_lang, tgt_txt, tgt_emb, _) in enumerate(emb_data):
                if src_lang == tgt_lang:
                    err_matrix[i1, i2] = 0
                else:
                    err, nbex, _ = self._xsim(
                        src_emb, src_lang, tgt_emb, tgt_lang, tgt_txt
                    )
                    err_matrix[i1, i2] = 100 * err / nbex
        df = pandas.DataFrame(err_matrix, columns=langs, index=langs)
        df.loc["avg"] = df.sum() / float(df.shape[0] - 1)  # exclude diagonal in average
        if self.output_dir:
            df.to_csv(os.path.join(self.output_dir, "xsim_nway_matrix.csv"))
        print(f"\n{tabulate(df, langs, floatfmt='.2f', tablefmt='grid')}\n\n")
        print(f"Global average: {df.loc['avg'].mean():.2f}")

    def calc_cosdist(
        self, embdir, src_langs, tgt_langs, tgt_aug_langs
    ) -> None:
        outputs = []
        src_emb_data = self._embed(
            embdir,
            src_langs,
            self.src_encoder,
            self.src_spm_model,
            self.src_bpe_codes,
            tokenizer=self.src_tokenizer,
            vocab_file=self.src_vocab_file
        )
        tgt_emb_data = self._embed(
            embdir,
            tgt_langs,
            self.tgt_encoder,
            self.tgt_spm_model,
            self.tgt_bpe_codes,
            tgt_aug_langs,
            tokenizer=self.tgt_tokenizer,
            vocab_file=self.tgt_vocab_file
        )
        combs = list(itertools.product(src_emb_data, tgt_emb_data))
        for (src_lang, _, src_emb, _), (tgt_lang, _, tgt_emb, _) in combs:
            if src_lang == tgt_lang:
                continue
            if not isinstance(src_emb, np.ndarray):
                src_emb = _load_embeddings(src_emb, self.emb_dimension, self.fp16)
            if not isinstance(tgt_emb, np.ndarray):
                tgt_emb = _load_embeddings(tgt_emb, self.emb_dimension, self.fp16)
            distances = paired_cosine_distances(src_emb, tgt_emb)
            dist = distances.mean()
            nbex = distances.size
            outputs.append(
                [self.corpus, f"{src_lang}-{tgt_lang}", f"{dist}", f"{nbex}"]
            )
        outputs.append(
            [
                self.corpus,
                "average",
                f"{round(np.array(outputs)[:,2].astype(np.float64).mean(), 2)}",
                f"{len(combs)}",
            ]
        )
        df = pandas.DataFrame(outputs, columns=[
                    "dataset",
                    "src-tgt",
                    "cosdist",
                    "nbex",
                ])
        if self.output_dir:
            df.to_csv(os.path.join(self.output_dir, "cosine_distance_matrix.csv"))
        print(
            tabulate(
                df,
                tablefmt="psql",
                headers=df.columns
            )
        )

def run_eval(args) -> None:
    evaluation = Eval(args)
    tmp_dir = None
    if args.embed_dir:
        os.makedirs(args.embed_dir, exist_ok=True)
        embed_dir = args.embed_dir
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        embed_dir = Path(tmp_dir.name)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    src_langs = sorted(args.src_langs.split(","))
    tgt_aug_langs = sorted(args.tgt_aug_langs.split(",")) if args.tgt_aug_langs else []
    if evaluation.nway:
        evaluation.calc_xsim_nway(embed_dir, src_langs)
    else:
        assert (
            args.tgt_langs
        ), "Please provide tgt langs when not performing n-way comparison"
        tgt_langs = sorted(args.tgt_langs.split(","))
        evaluation.calc_xsim(embed_dir, src_langs, tgt_langs, tgt_aug_langs)
        if args.cosine_distances:
            evaluation.calc_cosdist(embed_dir, src_langs, tgt_langs, tgt_aug_langs)
    if tmp_dir:
        tmp_dir.cleanup()  # remove temporary directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LASER: multilingual similarity error evaluation"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for evaluation files",
        required=True,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Name of evaluation corpus",
        required=True,
    )
    parser.add_argument(
        "--corpus-part",
        type=str,
        default=None,
        help="Specify split of the corpus to use e.g., dev",
        required=True,
    )
    parser.add_argument(
        "--margin",
        type=str,
        default=None,
        help="Margin for xSIM calculation. See: https://aclanthology.org/P19-1309",
    )
    parser.add_argument(
        "--min-sents",
        type=int,
        default=100,
        help="Only use test sets which have at least N sentences",
    )
    parser.add_argument(
        "--nway", action="store_true", help="Test N-way for corpora which support it"
    )
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=None,
        help="Store/load embeddings from specified directory (default temporary)",
    )
    parser.add_argument(
        "--index-comparison",
        action="store_true",
        help="Use index comparison instead of texts (not recommended when test data contains duplicates)",
    )
    parser.add_argument("--src-spm-model", type=str, default=None)
    parser.add_argument("--tgt-spm-model", type=str, default=None)
    parser.add_argument(
        "--src-bpe-codes",
        type=str,
        default=None,
        help="Path to bpe codes for src model",
    )
    parser.add_argument(
        "--tgt-bpe-codes",
        type=str,
        default=None,
        help="Path to bpe codes for tgt model",
    )
    parser.add_argument("--src-encoder", type=str, default=None, required=True)
    parser.add_argument("--tgt-encoder", type=str, default=None)
    parser.add_argument(
        "--buffer-size", type=int, default=100, help="Buffer size (sentences)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12000,
        help="Maximum number of tokens to process in a batch",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to process in a batch",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")

    parser.add_argument(
        "--src-langs",
        type=str,
        default=None,
        help="Source-side languages for evaluation",
        required=True,
    )
    parser.add_argument(
        "--tgt-langs",
        type=str,
        default=None,
        help="Target-side languages for evaluation",
    )
    parser.add_argument(
        "--tgt-aug-langs",
        type=str,
        default=None,
        help="languages with augmented data",
        required=False,
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store embedding matrices in fp16 instead of fp32",
    )
    parser.add_argument(
        "--sort-kind",
        type=str,
        default="quicksort",
        choices=["quicksort", "mergesort"],
        help="Algorithm used to sort batch by length",
    )
    parser.add_argument(
        "--use-hugging-face",
        action="store_true",
        help="Use a HuggingFace sentence transformer",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=1024,
        help="Embedding dimension for encoders",
    )
    parser.add_argument(
        "--src-vocab-file", 
        type=str, 
        default=None, 
        help="Use specified vocab file for encoding the source"
    )
    parser.add_argument(
        "--tgt-vocab-file", 
        type=str, 
        default=None, 
        help="Use specified vocab file for encoding the target"
    )
    parser.add_argument(
        "--src-tokenizer", 
        type=str, 
        default=None, 
        help="Use specified tokenizer bash script after preprocessing and before encoding the source. " +
        "It should be a bash script that takes an input file path as 1st (unnamed) arg and output file path as 2nd."
    )
    parser.add_argument(
        "--tgt-tokenizer", 
        type=str, 
        default=None, 
        help="Use specified tokenizer bash script after preprocessing and before encoding the target. " +
        "It should be a bash script that takes an input file path as 1st (unnamed) arg and output file path as 2nd."

    )
    parser.add_argument(
        "--cosine-distances", action="store_true", help="Compute average pairwise cosine distances between src and tgt"
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--do-pretok", action="store_true", help="Do preprocessing and pretokenization")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output file")
    args = parser.parse_args()
    run_eval(args)
