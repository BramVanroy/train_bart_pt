#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretraining models for denoising language modeling on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=bart
"""
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional

import evaluate
import numpy as np
import datasets
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import default_collate

from transformers import (
    BartTokenizer,
    BartConfig,
    BatchEncoding,
    BartForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer, TrainingArguments,
    is_torch_tpu_available, set_seed,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.3, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    permute_sentence_ratio: float = field(
        default=1.0, metadata={"help": "Ratio of sentences to be permuted in each document"}
    )
    poisson_lambda: float = field(
        default=3.5, metadata={"help": "Mean of Poisson distribution used to generate span-lengths to be masked"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    spacy_model: Optional[str] = field(
        default= None,
        metadata={
            "help": "By default, an English NLTK punct model is used for sentence splitting. If you give a spacy_model"
                    " name instead, we'll use that for sentence splitting. Note that spaCy and the chosen model have"
                    " to be installe.d"
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in {"csv", "json", "txt"}:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in {"csv", "json", "txt"}:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")

        if self.mlm_probability > 1.0:
            raise ValueError(f"'mlm_probability' should be less than or equal to 1.0")

        if self.permute_sentence_ratio > 1.0:
            raise ValueError(f"'permute_sentence_ratio' should be less than or equal to 1.0")


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling. The code is largely copied from
    `<https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L223>`__.
    For more information on how BART denoising language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.13461.pdf>`__
    or the `official code for preprocessing <https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
    """
    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permute_sentence_ratio: float = 1.0

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: List[Dict[str, List[int]]], verbose=False) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: [examples[i][k] for i in range(len(examples))] for k, v in examples[0].items()},
            tensor_type="pt"
        )

        if verbose:
            print("INPUT TENSOR SIZE", batch["input_ids"].size())

        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        if verbose:
            print("decoder_input_ids TENSOR SIZE", batch["decoder_input_ids"].size())

        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = self.permute_sentences(batch["input_ids"])
            do_permute = True

        print(batch["input_ids"])
        print(batch["labels"])
        exit()

        if verbose:
            print("INPUT TENSOR SIZE AFTER PERMUtE", batch["input_ids"].size())
        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.span_mask_tokens(
                batch["input_ids"], batch["labels"], do_permute
            )
            print(batch["input_ids"])
            exit()
        if verbose:
            print("INPUT TENSOR SIZE AFTER MASK", batch["input_ids"].size())

        # ignore pad tokens
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).to(torch.long)

        if verbose:
            print("ATTENTION TENSOR SIZE", batch["attention_mask"].size())
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).to(torch.long)
        if verbose:
            print("DECODER ATTENTION TENSOR SIZE AFTER MASK", batch["decoder_attention_mask"].size())

        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.clone()

        # find end locations of sentences
        end_sentence_mask = input_ids == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)
        num_sentences_map = {sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, num_sentences)}

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = {
            sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, num_to_permute)
        }

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])
        sentence_ends_map = {sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, sentence_ends)}

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            substitutions = np.random.permutation(num_sentences_map[i])[: num_to_permute_map[i]]
            ordering = np.arange(0, num_sentences_map[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute_map[i])]

            # write shuffled sentences into results
            index = 0
            for j in ordering:
                sentence = input_ids[i, (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[i][j]]
                results[i, index:index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        print("INPUTS", input_ids.size())
        print("LABELS", labels.size())
        special_tokens_mask_inputs = torch.BoolTensor([
            self.tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True) for sequence in input_ids
        ])
        print("special_tokens_mask_inputs", special_tokens_mask_inputs.size())

        special_tokens_mask_labels = torch.BoolTensor([
            self.tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True) for sequence in labels
        ])
        print("special_tokens_mask_labels", special_tokens_mask_labels.size())

        # determine how many tokens we need to mask in total
        is_token_mask = ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        print("is_token_mask", is_token_mask.size())

        num_tokens_to_mask = int(math.ceil(is_token_mask.sum() * self.mask_ratio))
        print("num_tokens_to_mask", num_tokens_to_mask)

        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        rng = np.random.default_rng()
        span_lengths = rng.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))
        print("span_lengths1", span_lengths)
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [span_lengths, rng.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))]
            )

        span_lengths = torch.tensor(span_lengths)
        print("span_lengths", span_lengths)
        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]
        print("span_lengths no null", span_lengths)

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = torch.argmin(torch.abs(torch.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        span_lengths = span_lengths[:cutoff_idx]
        print("span_lengths only ~num_tokens_to_mask", span_lengths)

        # randomly choose starting positions for masking
        token_indices = torch.argwhere(is_token_mask == 1)
        print("token_indices", token_indices.size())

        print("span_lengths", span_lengths)
        span_starts = torch.randperm(token_indices.size(0))[:span_lengths.size(0)]
        print("span_starts", span_starts)
        print("span_starts", span_starts.size(0))

        # prepare mask
        masked_indices = token_indices[span_starts]
        print("masked_indices", masked_indices.size())
        print("masked_indices", masked_indices)
        mask = torch.full_like(input_ids, fill_value=False, dtype=torch.bool)
        print("mask", mask.size())
        print("mask", mask)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        print("masked_indices", mask.size())
        print("masked_indices", mask)
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.size(1) - 1
        print("max_index", max_index)

        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        print("remaining", remaining)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            print("masked_indices", mask.size())
            print("masked_indices", mask)
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[torch.where(special_tokens_mask_inputs)] = False
        input_ids[torch.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[torch.where(mask == 0)] = -100
        else:
            labels[torch.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & torch.roll((mask == 1), 1, 1)
        new_input_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, :new_example.size(0)] = new_example

        return new_input_ids, labels


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def main():
    # TODO: go through this thing and compare with
    # https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
    # and https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        loaded_ds = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in  loaded_ds.keys():
             loaded_ds["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
             loaded_ds["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        loaded_ds = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in  loaded_ds.keys():
            loaded_ds["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            loaded_ds["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    ############################
    # Load tokenizer and model #
    ############################
    # TOKENIZER
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # CONFIG
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BartConfig.from_pretrained(model_args.config_name, vocab_size=len(tokenizer), **config_kwargs)
    elif model_args.model_name_or_path:
        config = BartConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = BartConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

    # MODEL
    if model_args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config
        )
    else:
        config.vocab_size = len(tokenizer)
        model = BartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer))

    #######################
    # Preprocess datasets #
    #######################
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = loaded_ds["train"].column_names
    else:
        column_names = loaded_ds["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Set max length
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Caching is not working well with spaCy so we use explicit fingerprints
    # looping over splits because we cannot use new_fingerprint on DatasetDict
    for k in loaded_ds.keys():
        base_fingerprint = f"{k}@{data_args.dataset_name}{data_args.dataset_config_name}" if data_args.dataset_name else None
        # Do sentence splitting
        sentence_tokenizer = None
        if data_args.spacy_model:
            import spacy
            spacy.prefer_gpu()
            # Only load the parser (depparse) which will set sentence boundaries
            sentence_tokenizer = spacy.load(data_args.spacy_model, exclude=["tagger", "ner", "lemmatizer", "textcat"])
        else:
            import nltk
            # Use Punkt Sentence Tokenizer to divide a document into a list of sentences
            nltk.download("punkt")
            sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    
        def sentence_split(examples):
            if data_args.spacy_model:
                docs = sentence_tokenizer.pipe(examples["text"])
                doc_sents = [map(str, doc.sents) for doc in docs]
            else:
                doc_sents = [[s for s in sentence_tokenizer.tokenize(t)] for t in examples["text"]]
    
            # use pad token as end of sentence indicator
            new_texts = [f"{tokenizer.bos_token}{tokenizer.pad_token.join(sents)}{tokenizer.eos_token}" for sents in doc_sents]
            return {"text": new_texts}
    
        with training_args.main_process_first(desc=f"Sentence splitting texts (k)"):
            # If using spaCy, we don't run multiple workers here but pass that to spacy's pipe
            loaded_ds[k] = loaded_ds[k].map(
                sentence_split,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Sentence splitting",
                new_fingerprint=f"{base_fingerprint}+ss{data_args.spacy_model}"
            )
            del sentence_tokenizer
    
        # Tokenize (subword) every text, then concatenate them together before splitting them in smaller parts.
        # Attention masks will be added in the collator
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], add_special_tokens=False, return_attention_mask=False)
    
        with training_args.main_process_first(desc="dataset map tokenization"):
            loaded_ds[k] = loaded_ds[k].map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=text_column_name,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing",
                new_fingerprint=f"{base_fingerprint}+ss{data_args.spacy_model}+tok"
            )
    
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
    
        with training_args.main_process_first(desc="Grouping texts together"):
            loaded_ds[k] = loaded_ds[k].map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in sequences of {max_seq_length}",
                new_fingerprint=f"{base_fingerprint}+ss{data_args.spacy_model}+tok+group{max_seq_length}"
            )

    train_dataset = None
    if training_args.do_train:
        if "train" not in loaded_ds:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = loaded_ds["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in loaded_ds:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = loaded_ds["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Data collator ill take care of randomly masking the tokens and permuting the sentences.
    data_collator = DataCollatorForBartDenoisingLM(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        mask_ratio=data_args.mlm_probability,
        poisson_lambda=data_args.poisson_lambda,
        permute_sentence_ratio=data_args.permute_sentence_ratio,
    )

    # Some trainer-specific submethods that may be relevant:
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
