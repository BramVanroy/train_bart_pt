import math

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# I follow defaults based on this post: https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320
# random_ratio = mask_random (cf. https://github.com/facebookresearch/fairseq/blob/a6a63279422f846a3c2f6c45b9c96d6951cc4b82/fairseq/data/denoising_dataset.py#L143)
# Mask whole word is created as per here: https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/tasks/multilingual_masked_lm.py#L118
# but it is not enabled by default (as the given defaults do not contain mask_whole_words)
def process(input_ids, tokenizer: PreTrainedTokenizerBase, permute_sentence_ratio=1.0, mask_ratio=0.3,
            random_ratio =0.1, poisson_lambda=3.5, mask_length="span-poisson", mask_whole_word =None):
    source, target = input_ids, input_ids.clone()

    mask_span_distribution = None
    if mask_length == "span-poisson":
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        mask_span_distribution = torch.distributions.Categorical(ps)

    if permute_sentence_ratio > 0.0:
        source = permute_sentences(source, tokenizer, permute_sentence_ratio=permute_sentence_ratio)

    if mask_ratio > 0:
        source = add_whole_word_mask(source, tokenizer, mask_ratio=mask_ratio, random_ratio=random_ratio,
                                     mask_span_distribution=mask_span_distribution, mask_whole_word=mask_whole_word )

    assert (source >= 0).all()
    assert (source[1:-1] >= 1).all()
    assert (source <= len(tokenizer)).all()
    assert source[0] == tokenizer.bos_token_id
    assert source[-1] == tokenizer.eos_token_id
    return {
        "input_ids": source,
        "labels": target,
    }


def permute_sentences(input_ids, tokenizer: PreTrainedTokenizerBase, *, permute_sentence_ratio=1.0):
    full_stops = input_ids == tokenizer.pad_token_id

    print(full_stops)
    # Pretend it ends with a full stop so last span is a sentence
    full_stops[-2] = 1

    print(full_stops)

    # Tokens that are full stops, where the previous token is not
    sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
    result = input_ids.clone()

    num_sentences = sentence_ends.size(0)
    num_to_permute = math.ceil((num_sentences * 2 * permute_sentence_ratio) / 2.0)
    substitutions = torch.randperm(num_sentences)[:num_to_permute]
    ordering = torch.arange(0, num_sentences)
    ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

    # Ignore <bos> at start
    index = 1
    for i in ordering:
        sentence = input_ids[(sentence_ends[i - 1] if i > 0 else 1): sentence_ends[i]]
        result[index: index + sentence.size(0)] = sentence
        index += sentence.size(0)
    return result


def get_word_starts(input_ids, mask_whole_word):
    if mask_whole_word is not None:
        is_word_start = mask_whole_word.gather(0, input_ids)
    else:
        is_word_start = torch.ones(input_ids.size())
    is_word_start[0] = 0
    is_word_start[-1] = 0
    return is_word_start


def add_whole_word_mask(input_ids, tokenizer: PreTrainedTokenizerBase, *, mask_ratio=0.3, random_ratio=0.1,
                        replace_length=1, mask_span_distribution=None, mask_whole_word=None):
    is_word_start = get_word_starts(input_ids, mask_whole_word=mask_whole_word )
    num_to_mask = int(math.ceil(is_word_start.float().sum() * mask_ratio))
    num_inserts = 0
    if num_to_mask == 0:
        return input_ids

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(input_ids, tokenizer, num_inserts / input_ids.size(0),
                                       random_ratio=random_ratio)

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,)).long()

    assert is_word_start[-1] == 0

    word_starts = is_word_start.nonzero(as_tuple=False)
    indices = word_starts[
        torch.randperm(word_starts.size(0))[:num_to_mask]
    ].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = input_ids.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[
        -1
    ] = 255  # acts as a long length, so spans don't go over the end of doc
    if replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        input_ids[indices] = tokenizer.mask_token_id
        input_ids[indices[mask_random]] = torch.randint(
            1, len(tokenizer), size=(mask_random.sum(),)
        )

    if mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                input_ids[indices] = tokenizer.mask_token_id
                input_ids[indices[mask_random]] = torch.randint(
                    1, len(tokenizer), size=(mask_random.sum(),)
                )
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                input_ids[indices] = tokenizer.mask_token_id
                input_ids[indices[mask_random]] = torch.randint(
                    1, len(tokenizer), size=(mask_random.sum(),)
                )

            assert source_length - 1 not in indices

    input_ids = input_ids[to_keep]

    if num_inserts > 0:
        input_ids = add_insertion_noise(input_ids, tokenizer, num_inserts / input_ids.size(0),
                                        random_ratio=random_ratio)

    return input_ids


def add_insertion_noise(input_ids, tokenizer: PreTrainedTokenizerBase, p, *, random_ratio=0.1):
    if p == 0.0:
        return input_ids

    num_tokens = len(input_ids)
    n = int(math.ceil(num_tokens * p))

    noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
    noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
    noise_mask[noise_indices] = 1
    result = torch.LongTensor(n + len(input_ids)).fill_(-1)

    num_random = int(math.ceil(n * random_ratio))
    result[noise_indices[num_random:]] = tokenizer.mask_token_id
    result[noise_indices[:num_random]] = torch.randint(
        low=1, high=len(tokenizer), size=(num_random,)
    )

    result[~noise_mask] = input_ids

    assert (result >= 0).all()
    return result


def get_n_mask_tokens(tokens, mask_token_id):
    unique, counts = np.unique(tokens, return_counts=True)
    counter = dict(zip(unique, counts))
    return counter[mask_token_id]


def get_n_nonspecial_tokens(tokens, all_special_ids):
    return len([t for t in tokens if t not in all_special_ids])


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    text = "On september 2nd the Group of Seven (G7) countries launched a new attempt to regain the advantage in the" \
           f" West’s energy confrontation with Russia.{tokenizer.pad_token}Imposing a price cap on purchases of Russian oil and oil" \
           " products, probably to take effect on December 5th."
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze()
    n_input_toks = get_n_nonspecial_tokens(input_ids, tokenizer.all_special_ids)
    print("DECODED INPUT", tokenizer.decode(input_ids))
    processed = process(input_ids, tokenizer)
    input_ids_out = processed["input_ids"].squeeze()
    n_output_toks = get_n_nonspecial_tokens(input_ids_out, tokenizer.all_special_ids)
    print("DECODED OUTPUT", tokenizer.decode(input_ids_out))

    n_masks_out = get_n_mask_tokens(input_ids_out, tokenizer.mask_token_id) + (n_input_toks-n_output_toks)
    print(f"MASK RATIO ({n_masks_out}/{n_input_toks})", n_masks_out/n_input_toks)


if __name__ == '__main__':
    main()
