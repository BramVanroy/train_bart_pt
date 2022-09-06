# Train BART with PyTorch

**This is an alpha release of the training script. Please try it and let me know which problems you experience!**

An example script of training [BART](https://aclanthology.org/2020.acl-main.703/), an encoder-decoder that is trained
on the objective of denoising tokens and spans.

This code builds on the [Flax](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling)
example and the data collator was ported from
[fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md)
to a collator format rather than a dataset (i.e., batched).

# Usage

Inspired by the process [here](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling).

## 1. Train a tokenizer on a dataset on the hub

```shell
python prepare_tokenizer.py \
    oscar \
    --dataset_config_name unshuffled_deduplicated_nl \
    --dataset_split train \
    --dout ./my-bart-model
```

## 2. Prepare a model config file based on an existing model

```shell
python prepare_config.py \
    --pretrained_model_name facebook/bart-base \
    --dout ./my-bart-model
```

## 3. Train the model and specific tokenizer and config

```shell
python run_bart_dlm.py \
    --config_name ./my-bart-model \
    --tokenizer_name ./my-bart-model \
    --dataset_name oscar \
    --dataset_config_name unshuffled_deduplicated_nl \
    --output_dir ./my-bart-model \
    --do_train \
    --do_eval
```


## Some notes 

### Sentence splitting

As part of BART, the sentences in a sample may be permuted (reordered). To detect sentences for each sample, we need
sentence splitting. By dfault, we'll use NLTK's English punct sentence splitter but by passing a spaCy model name
to `spacy_model` (e.g. `en_core_web_sm`) you can also rely on spaCy for better (but slower) sentence splitting.
You can also disable sentence splitting completely with `==no_sentence_splitting`. In that case, make sure the
sentences are already split with a padding token between them )`<pad>`.


### Default values
The defaults are set to the
[given BART args](https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320). This differs from
the Flax defaults in one respect, namely `poisson_lambda`, which is now set to `3.5` instead of `3.0`.


### HF (Flax), fairseq, and current implementation

There are some differences in implementation between fairseq, the HF FLAX example, and this PyTorch implementation.

- `argwhere` in the Flax example
[in this position](https://github.com/huggingface/transformers/blob/65fb71bc762c46bb067306c1fd083b1cba87a095/examples/flax/language-modeling/run_bart_dlm_flax.py#L319)
is not the same as what is happening in fairseq. [In fairseq](https://github.com/facebookresearch/fairseq/blob/a6a63279422f846a3c2f6c45b9c96d6951cc4b82/fairseq/data/denoising_dataset.py#L230)
we check explicitly that the previous token was not a "full stop" (padding token) but in HF we just check whether the
current token is a full stop. In the current example I also explicitly check that the next token is not a full stop,
in case of padding. (However, in practice that should be a non-issue since all batches/samples should have the
same sequence length and there should not be any padding.)
- I found that the result of sentence permutation was not consistent in terms of where the separating pad token ended
up ([bug report](https://github.com/facebookresearch/fairseq/issues/4695)), so I have reimplemented that method so
that sentences in a sequence are still separated by a padding token, even after permutation.
- In HF FLAX, the token_mask is restricted to [non-special and non-padding tokens](https://github.com/huggingface/transformers/blob/65fb71bc762c46bb067306c1fd083b1cba87a095/examples/flax/language-modeling/run_bart_dlm_flax.py#L361).
In Fairseq, by default, only the first and last tokens are excluded and [all others](https://github.com/facebookresearch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py#L241)
are prone to masking. The HF implementation seems sensible so I follow that. `get_special_tokens_mask` includes the 
padding token, though, so no need to add that separately.
- The Flax example does not include methods to add more noise. I have ported those as well.
- However, I did not adapt `add_insertion_noise` to work well with padded sequences. So the inserted noise may occur
ANYWHERE. It is unclear whether this is intended behavior.

Alternatively, we could implement all this processing on the dataset level and use `Dataset.map`. This has some
advantages:

- more true to fairseq implementation (sample level rather than batch level);
- cached.

... and disadvantages:

- potentially slower (not batched), although we can integrate a batched approach. But as discussed above, this will be
less true to the original fairseq implementation in `add_insertion_noise`
- every sample is always processed the same. So in small datasets which are seen multiple times by the model, the 
same sample will always be processed the same. In a dataloader, that will not be the case because the processing
occurs on every iteration rather than once before training.

### Questions/Uncertainties
- Do the padding tokens still serve a purpose after permutation? (Teaching the model to learn to detect sentence boundaries?)
- It seems that `add_insertion_noise` can insert noise _anywhere_, which means that it will also overwrite special
tokens and that sequence don't necessarily end with a EOS token. Is that a problem?
