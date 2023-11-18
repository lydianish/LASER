import argparse
from mteb import MTEB
from embed import SentenceEncoder
import sentencepiece as spm

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
        case_sensitive=False
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
        self.case_sensitive = case_sensitive

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
    
        preprocessed_sentences = sentences if self.case_sensitive else [ s.lower() for s in sentences ]
        tokenized_sentences = [ " ".join(s) for s in self.spm_model.encode_as_pieces(preprocessed_sentences) ]
        embeddings = super().encode_sentences(tokenized_sentences)
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
        "--case-sensitive", action="store_true", help="Detailed output"
    )
    
    args = parser.parse_args()
    evaluation = MTEB(task_types=["Classification", "STS"], task_categories=["s2s"], task_langs=["en"])
    # evaluation = MTEB(tasks=["Banking77Classification"])
    model = MyModel(args.encoder, spm_model=args.spm_model, vocab=args.vocab, verbose=args.verbose, case_sensitive=args.case_sensitive)
    evaluation.run(model, output_folder=args.output_dir)