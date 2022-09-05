import math
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# I follow defaults based on this post: https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320
# random_ratio = mask_random (cf. https://github.com/facebookresearch/fairseq/blob/a6a63279422f846a3c2f6c45b9c96d6951cc4b82/fairseq/data/denoising_dataset.py#L143)
# Mask whole word is created as per here: https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/tasks/multilingual_masked_lm.py#L118
# but it is not enabled by default (as the given defaults do not contain mask_whole_words). It should be a ByteTensor (maybe a booltensor works?) of
# the same size as the vocabulary. For each token it indicates True if the token is the start of a word (as returned by BPE)
def process(input_ids, tokenizer: PreTrainedTokenizerBase, permute_sentence_ratio=1.0, mask_ratio=0.3,
            random_ratio=0.1, poisson_lambda=3.5, mask_length: Optional[str]="span-poisson", mask_whole_word=None):
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
                                     mask_span_distribution=mask_span_distribution, mask_whole_word=mask_whole_word)

    return {
        "input_ids": source,
        "labels": target,
    }


def permute_sentences(input_ids, tokenizer: PreTrainedTokenizerBase, *, permute_sentence_ratio=1.0):
    all_results = input_ids.clone()
    for seq_idx, sequence in enumerate(input_ids):
        full_stops = sequence == tokenizer.pad_token_id

        # Find the position of </s> EOS tokens, and mark the position before that as a full stop
        # so that the last sentence can also be extracted as a single
        # This approach is needed when our batches have padding (and we cannot simply target the one but last item)
        eos_positions = (sequence == tokenizer.eos_token_id).roll(-1)
        full_stops[eos_positions] = 1

        # Mark sentence ends: those cases where the token is a full_stop (pad), but the previous and next ones are not
        next_token_is_full_stop = torch.cat((full_stops[2:], torch.BoolTensor([0])))
        sentence_ends = (full_stops[1:] * ~full_stops[:-1] * ~next_token_is_full_stop).nonzero(as_tuple=False) + 2
        result = sequence.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * permute_sentence_ratio) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for order_idx, orig_sent_idx in enumerate(ordering):
            is_last_orig = orig_sent_idx == num_sentences - 1
            is_last_in_loop = order_idx == num_sentences - 1
            start_idx = sentence_ends[orig_sent_idx - 1] if orig_sent_idx > 0 else 1
            # remove last idx (pad) from last sentence of this loop but only if it is not the orig last sentence
            end_idx = sentence_ends[orig_sent_idx] - (int(is_last_in_loop) if not is_last_orig else 0)
            sentence = sequence[start_idx:end_idx]

            # add padding token if this was the original last sentence and now it isn't anymore
            if is_last_orig and not is_last_in_loop:
                sentence = torch.cat((sentence, torch.LongTensor([tokenizer.pad_token_id])))

            result[index: index + sentence.size(0)] = sentence
            index += sentence.size(0)

        all_results[seq_idx] = result

    return all_results


def get_word_starts(input_ids, tokenizer, mask_whole_word=None):
    if mask_whole_word is not None:
        is_word_start = mask_whole_word.gather(0, input_ids.view(-1), dtype=torch.long).reshape(input_ids.size())
    else:
        is_word_start = (~torch.BoolTensor([
            tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True) for seq in input_ids
        ])).long()
    is_word_start[:, 0] = 0
    is_word_start[:, -1] = 0
    return is_word_start


def add_whole_word_mask(input_ids, tokenizer: PreTrainedTokenizerBase, *, mask_ratio=0.3, random_ratio=0.1,
                        replace_length=1, mask_span_distribution=None, mask_whole_word=None):
    # Note that is_word_start cannot be a booltensor but has to be an int tensor as we use it to subtract
    # from the span lengths later on
    is_word_start = get_word_starts(input_ids, tokenizer, mask_whole_word=mask_whole_word)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * mask_ratio))

    num_inserts = 0
    if num_to_mask == 0:
        return input_ids

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))
        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)

        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths,
                                 mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                                 ],
                                dim=0)
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i-1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        # For every 0-length span, we instead insert noise
        # So we decrease the required `num_to_mask` and instead add to `num_inserts`
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(input_ids, tokenizer, num_inserts / input_ids.numel(),
                                       random_ratio=random_ratio)

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,), dtype=torch.long)

    assert not is_word_start[:, 0].any()
    assert not is_word_start[:, -1].any()

    word_starts = is_word_start.nonzero(as_tuple=False)
    indices = word_starts[
        torch.randperm(word_starts.size(0))[:num_to_mask]
    ]

    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = input_ids.size(1)
    assert source_length - 1 not in indices[:, 1]

    to_keep = torch.ones_like(input_ids, dtype=torch.bool)
    is_word_start[:, -1] = 255  # acts as a long length, so spans don't go over the end of doc

    if replace_length == 0:
        to_keep[indices[:, 0], indices[:, 1]] = 0
    else:
        # Mask some tokens with a mask token
        for idxs in indices:
            input_ids[tuple(idxs)] = tokenizer.mask_token_id

        # Replace a fraction (random_ratio) with a random token
        rand_tokens = torch.randint(
            1, len(tokenizer), size=(mask_random.sum(),)
        )
        for idxs, tok in zip(indices[mask_random], rand_tokens):
            input_ids[tuple(idxs)] = tok

    if mask_span_distribution is not None:
        lengths -= 1

        while indices.size(0) > 0:
            lengths -= is_word_start[indices[:, 0], indices[:, 1] + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted, :]
            indices[:, 1] += 1  # increment to keep masking the next positions

            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]

            if replace_length != -1:
                to_keep[indices[:, 0], indices[:, 1]] = 0
            else:
                # Mask some tokens with a mask token
                for idxs in indices:
                    input_ids[tuple(idxs)] = tokenizer.mask_token_id

                # Replace a fraction (random_ratio) with a random token
                rand_tokens = torch.randint(
                    1, len(tokenizer), size=(mask_random.sum(),)
                )
                for idxs, tok in zip(indices[mask_random], rand_tokens):
                    input_ids[tuple(idxs)] = tok

            assert source_length - 1 not in indices[:, 1]
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices[:, 0], indices[:, 1] + 1] == 0
            indices = indices[uncompleted, :]
            indices[:, 1] += 1  # increment to keep masking the next positions

            mask_random = mask_random[uncompleted]

            if replace_length != -1:
                to_keep[indices[:, 0], indices[:, 1]] = 0
            else:
                # Mask some tokens with a mask token
                for idxs in indices:
                    input_ids[tuple(idxs)] = tokenizer.mask_token_id

                # Replace a fraction (random_ratio) with a random token
                rand_tokens = torch.randint(
                    1, len(tokenizer), size=(mask_random.sum(),)
                )
                for idxs, tok in zip(indices[mask_random], rand_tokens):
                    input_ids[tuple(idxs)] = tok

            assert source_length - 1 not in indices[:, 1]

    # Remove some items (e.g. consecutive masks)
    final_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)
    for keeper_idx, keeper in enumerate(to_keep):
        seq = input_ids[keeper_idx, keeper]
        final_ids[keeper_idx, :seq.size(0)] = seq
    input_ids = final_ids

    if num_inserts > 0:
        input_ids = add_insertion_noise(input_ids, tokenizer, num_inserts / input_ids.numel(),
                                        random_ratio=random_ratio)

    return input_ids


def add_insertion_noise(input_ids, tokenizer: PreTrainedTokenizerBase, p, *, random_ratio=0.1):
    # TODO: vectorize?
    if p == 0.0:
        return input_ids

    seq_num_tokens = input_ids.size(1)
    n = int(math.ceil(seq_num_tokens * p))
    all_results = torch.full((input_ids.size(0), seq_num_tokens+n), fill_value=tokenizer.pad_token_id)
    for seq_id, sequence in enumerate(input_ids):
        noise_indices = torch.randperm(seq_num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(seq_num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1

        result = torch.LongTensor(seq_num_tokens + n).fill_(-1)

        num_random = int(math.ceil(n * random_ratio))
        result[noise_indices[num_random:]] = tokenizer.mask_token_id
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(tokenizer), size=(num_random,)
        )

        result[~noise_mask] = sequence

        assert (result >= 0).all()
        all_results[seq_id] = result

    return all_results


def get_n_mask_tokens(tokens, mask_token_id):
    if mask_token_id not in tokens:
        return 0

    unique, counts = np.unique(tokens, return_counts=True)
    counter = dict(zip(unique, counts))

    return counter[mask_token_id]


def get_n_nonspecial_tokens(batch, tokenizer):
    if not batch.numel():  # if empty batch
        return 0
    return len([t for tokens in batch for t in tokens if t not in tokenizer.all_special_ids])


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # Two sequences, containing a padtoken to separate sentences
    text = ["A cookie is a baked or cooked snack or dessert that is typically small, flat and sweet."
            f"{tokenizer.pad_token}It usually contains flour, sugar, egg, and some type of oil, fat, or butter."
            f"{tokenizer.pad_token}It may include other ingredients such as raisins, oats, chocolate chips, nuts, etc.",
            "Biscuit or cookie variants include sandwich biscuits, such as custard creams."
            f"{tokenizer.pad_token}Chewier biscuits are sometimes called cookies"]
    encoded = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    n_input_toks = get_n_nonspecial_tokens(input_ids, tokenizer)

    processed = process(input_ids, tokenizer)

    input_ids_out = processed["input_ids"]
    n_output_toks = get_n_nonspecial_tokens(input_ids_out, tokenizer)
    # Very coarse calculation. Doing (n_input_toks - n_output_toks) to catch masked spans (one mask for multiple tokens)
    # but this will then also include added noise for instance
    # Also, deletion are replaced with padding so there is noise there
    n_masks_out = get_n_mask_tokens(input_ids_out, tokenizer.mask_token_id) + (n_input_toks - n_output_toks)

    print("DECODED INPUT", tokenizer.batch_decode(input_ids))
    print("NO. NON SPECIAL INPUT", n_input_toks)
    print("DECODED OUTPUT", tokenizer.batch_decode(input_ids_out))
    print(f"MASK RATIO ({n_masks_out}/{n_input_toks})", n_masks_out / n_input_toks)


if __name__ == '__main__':
    main()
