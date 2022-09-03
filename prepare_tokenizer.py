# ADAPTED FROM THIS TRANSFORMERS EXAMPLE:
# https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling

from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer


def train_tokenizer():
    # load dataset
    dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

    # Instantiate tokenizer
    tokenizer = ByteLevelBPETokenizer()

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    # Customized training
    tokenizer.train_from_iterator(batch_iterator(), vocab_size=50265, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save("./norwegian-roberta-base/tokenizer.json")


def main():
    import argparse
    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument()


if __name__ == '__main__':
    main()

