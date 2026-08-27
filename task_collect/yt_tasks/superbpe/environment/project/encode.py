"""
Calculate encoding efficiency (in bytes per token) over a corpus of text.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer
import click
import random
import regex as re

import os
from tqdm import tqdm
from collections import Counter
from utils import (
    get_files_with_num_bytes,
    read_json,
    ensure_dir,
    get_pretokenization_regex,
)

RANDOM_SEED = 5
NUM_BYTES = 10**9


@click.command()
@click.option(
    "--tokenizer_path",
    type=str,
    help="Path to tokenizer.json",
)
@click.option(
    "--corpus_dir",
    type=str,
    default=None,
    help="Directory of text files to encode.",
)
@click.option(
    "--file_path",
    type=str,
    default=None,
    help="Path to a single text file to encode. One of corpus_dir or file_path must be provided.",
)
@click.option("--output_dir", type=str, default=None)
@click.option(
    "--num_bytes",
    type=int,
    default=NUM_BYTES,
    help="Size of text (in bytes) to encode. If -1, will encode all files in corpus_dir.",
)
@click.option(
    "--vocab_size",
    type=int,
    default=None,
)
@click.option(
    "--dropout", type=float, help="Dropout rate for the tokenizer.", default=None
)
@click.option(
    "--save_token_stats",
    is_flag=True,
    help="Save token counts for each file.",
    default=False,
)
@click.option(
    "--save_bytes_per_token",
    is_flag=True,
    help="Save bytes per token stats.",
    default=False,
)
def main(
    tokenizer_path: str,
    corpus_dir: str,
    file_path: str,
    output_dir: str,
    num_bytes: int,
    vocab_size: int,
    dropout: float,
    save_token_stats: bool,
    save_bytes_per_token: bool,
):
    random.seed(RANDOM_SEED)
    if corpus_dir:
        corpus_dir = Path(corpus_dir)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
    tokenizer_json = read_json(tokenizer_path)

    # if vocab_size is given, construct tokenizer with the desired vocab_size
    if vocab_size and vocab_size <= tokenizer.get_vocab_size():
        print(f"We will only use the top {vocab_size} merges for encoding.", flush=True)
        if tokenizer_json["model"]["type"] == "BPE":
            merges = tokenizer_json["model"]["merges"]
            tokenizer_json["model"]["merges"] = merges[:vocab_size]
            tokenizer_json["model"]["ignore_merges"] = False
        elif tokenizer_json["model"]["type"] == "WordPiece":
            vocab = tokenizer_json["model"]["vocab"]
            tokenizer_json["model"]["vocab"] = dict(list(vocab.items())[:vocab_size])
        else:
            raise ValueError(
                f"Tokenizer type {tokenizer_json['model']['type']} not supported"
            )

        # create temporary new tokenizer file with truncated vocabulary
        tokenizer_path = tokenizer_path.replace(
            "tokenizer.json", f"tokenizer_{vocab_size}.json"
        )
        with open(tokenizer_path, "w") as fout:
            json.dump(tokenizer_json, fout, indent=4, ensure_ascii=False)

        tokenizer = Tokenizer.from_file(tokenizer_path)
        count_pretokens = False
    elif vocab_size:
        raise ValueError(
            f"Vocab size ({vocab_size}) > tokenizer vocab size ({tokenizer.get_vocab_size()})."
        )
    else:
        count_pretokens = True

    print(f"Using tokenizer from {tokenizer_path}", flush=True)

    if dropout:
        print(f"Setting dropout to {dropout}", flush=True)
        tokenizer.model.dropout = dropout

    if count_pretokens:
        pretok_regex = get_pretokenization_regex(tokenizer_json)
        print(f"Using pretokenization regex: {pretok_regex}", flush=True)

    def encode_file(file, count_pretokens=False):
        """
        Encode file and return the number of tokens.
        """
        with open(file, "r") as fin:
            text = fin.read()

        # Split into chunks so we don't OOM
        # This is ok bc tokenizer training splits on newline
        tokens = []
        pps = text.split("\n\n")
        chunk_size = max(len(pps) // 20, 100)
        for i in tqdm(range(0, len(pps), chunk_size), desc=os.path.basename(file)):
            chunk = "\n\n".join(pps[i : i + chunk_size]) + "\n\n"
            encoded = tokenizer.encode(chunk)
            tokens.extend(encoded.ids)
            # Note to self: num_pretokens will not be completely accurate for superword tokenizers because
            # the tokenizers training library splits on newline (separately from pretokenization). However,
            # the upper bound calculation is mainly for pretok tokenizers anyway, so we won't worry too
            # much about this case.

        num_pretokens = (
            len([m.group() for m in re.finditer(pretok_regex, text)])
            if count_pretokens
            else None
        )

        return tokens, num_pretokens

    # Collect list of files to be encoded
    if corpus_dir:
        if num_bytes == -1:
            num_bytes = None
        file_list, byte_count = get_files_with_num_bytes(
            corpus_dir, num_bytes, loop_around=False
        )
    elif file_path:
        file_list = [file_path]
        byte_count = os.path.getsize(file_path)
    else:
        raise ValueError("Either corpus_dir or file_path must be provided.")

    # Count tokens in files
    token_count = 0
    pretoken_count = 0
    for file in file_list:
        tokens, num_pretokens = encode_file(file, count_pretokens=count_pretokens)
        token_count += len(tokens)
        if count_pretokens:
            pretoken_count += num_pretokens
        if save_token_stats:
            filename = os.path.basename(file).split(".txt")[0]
            ensure_dir(f"encoded/{tokenizer_name}")
            with open(f"encoded/{tokenizer_name}/{filename}.json", "w") as fout:
                token_counter = Counter(tokens)
                json.dump(token_counter, fout, indent=5)

    # Save encoding efficiency stats to output_dir
    if save_bytes_per_token:
        assert output_dir is not None
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        if vocab_size:
            out_filename = f"token_byte_counts_{vocab_size}.json"
        else:
            out_filename = "token_byte_counts.json"

        with open(output_dir / out_filename, "w") as fout:
            d = {
                "test_files": file_list,
                "token_count": token_count,
                "byte_count": byte_count,
            }
            json.dump(d, fout, indent=5)

        if count_pretokens:
            with open(output_dir / "pretoken_byte_counts.json", "w") as fout:
                d = {
                    "test_files": file_list,
                    "pretoken_count": pretoken_count,
                    "byte_count": byte_count,
                }
                json.dump(d, fout, indent=5)

        print(f"Saved to {output_dir / out_filename}", flush=True)

    if vocab_size and vocab_size <= tokenizer.get_vocab_size():
        os.remove(tokenizer_path)


if __name__ == "__main__":
    main()
