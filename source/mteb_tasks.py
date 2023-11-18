import argparse
from mteb import MTEB
from embed import SentenceEncoder
import sentencepiece as spm
from lib.text_processing import PreprocessLineForSPM

class MyModel(SentenceEncoder):
    def __init__(
        self,
        model_path,
        spm_model,
        max_sentences=None,
        max_tokens=None,
        vocab=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
        lower_case=False,
        no_preprocessing=False
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
        self.spm_model = spm.SentencePieceProcessor(model_file=spm_model)
        self.lower_case = lower_case
        self.no_preprocessing = no_preprocessing

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if self.no_preprocessing:
            preprocessed_sentences = sentences
        elif self.lower_case:
            preprocessed_sentences = [ s.lower() for s in sentences ]
        else:
            preprocessed_sentences = [ PreprocessLineForSPM(s) for s in sentences ]
        spm_sentences = [ " ".join(s) for s in self.spm_model.encode_as_pieces(preprocessed_sentences) ]
        embeddings = super().encode_sentences(spm_sentences)
        return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Evaluate on MTEB benchmarks")
    parser.add_argument(
        "--encoder", type=str, required=True, help="encoder to be used"
        )
    parser.add_argument(
        "--vocab", type=str, required=True, help="Use specified vocab file for encoding"
    )
    parser.add_argument(
        "--spm-model", type=str, default=None, help="Apply SPM using specified model"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output sentence embeddings"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Detailed output"
    )
    parser.add_argument(
        "--lower-case", action="store_true", help="Lower case only as preprocessing for embeddings"
    )
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="No text preprocessing for embeddings"
    )
    parser.add_argument(
        "--english-only", action="store_true", help="Evaluate on tasks that require English-only encoder"
    )
    
    args = parser.parse_args()
    if args.english_only:
        evaluation = MTEB(task_types=["PairClassification", "Classification", "STS"], task_categories=["s2s"], task_langs=["en"])
    else:
        evaluation = MTEB(tasks=["BUCC"])
    model = MyModel(args.encoder, spm_model=args.spm_model, vocab=args.vocab, verbose=args.verbose, lower_case=args.lower_case, no_preprocessing=args.no_preprocessing)
    evaluation.run(model, output_folder=args.output_dir)