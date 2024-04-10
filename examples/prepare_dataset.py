
import click

import datasets
from transformers import AutoTokenizer

import torch

from tokenization_rwkv_world import RWKVWorldTokenizer

tokenizer = RWKVWorldTokenizer()

def _rechunk_tokenize(rechunk_size: int, input_column: str, output_column: str, examples):
    special_token = torch.tensor([0], dtype=torch.long)
    seqs = []
    for e in examples[input_column]:
        seq = tokenizer(e, padding=False, truncation=False, return_tensors="pt")
        seqs.append(seq.input_ids[0])
        seqs.append(special_token)
    seqs = torch.cat(seqs)
    rechunked = seqs[: (seqs.size(0) // rechunk_size) * rechunk_size].view(
        -1, rechunk_size
    )
    return {output_column: rechunked}


@click.command()
@click.option("--rechunk_size", default=4097, help="Rechunk size for the dataset")
@click.option("--input_column", default="text", help="Column to tokenize")
@click.option(
    "--output_column", default="input_ids", help="Output column for the tokenized dataset"
)
@click.option(
    "-i",
    "--input_name",
    help="HuggingFace dataset name to tokenize, accept format:"
    + '"dataset_name" or "json:file_a,file_b,..."',
)
@click.option("-o", "--output_dir", help="Output directory for the tokenized dataset")
def main(rechunk_size, input_column, output_column, input_name, output_dir):
    print(f"Tokenizing HuggingFace dataset {input_name} to locally saved {output_dir}")
    if ":" in input_name:
        input_name, input_data_file = input_name.split(":")
        if "," in input_data_file:
            input_data_file = input_data_file.split(",")
    else:
        input_data_file = None
    dataset = datasets.load_dataset(
        input_name, data_files=input_data_file, trust_remote_code=True
    )
    dataset.shuffle().flatten_indices(num_proc=8).map(
        lambda x: _rechunk_tokenize(rechunk_size, input_column, output_column, x),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    ).save_to_disk(output_dir, num_proc=8)


if __name__ == "__main__":
    main()
