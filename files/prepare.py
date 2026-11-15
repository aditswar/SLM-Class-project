"""Step 1: download SimpleStories, train an 8k BPE tokenizer, write uint16 token shards.
Run once:  python prepare.py
Needs:     pip install datasets tokenizers numpy
"""
import itertools
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = 8192
EOS = "<|endoftext|>"
os.makedirs("data", exist_ok=True)

ds = load_dataset("lennart-finke/SimpleStories")
print(ds)  # shows the split names; adjust below if they differ


def batches(split, n=1000):
    d = ds[split]
    for i in range(0, len(d), n):
        yield d[i : i + n]["story"]


# ---- tokenizer (trained on the first 200k stories) ----
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=[EOS],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(itertools.islice(batches("train"), 200), trainer)
tok.save("data/tokenizer.json")
eos_id = tok.token_to_id(EOS)
print("vocab size:", tok.get_vocab_size(), "eos id:", eos_id)


# ---- tokenize to flat uint16 binary files ----
def write(name, split):
    total = 0
    with open(f"data/{name}.bin", "wb") as f:
        for texts in batches(split, 5000):
            ids = []
            for e in tok.encode_batch(texts):
                ids.extend(e.ids)
                ids.append(eos_id)
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            total += len(ids)
    print(f"{name}: {total / 1e6:.1f}M tokens")


write("validation", "test")
write("train", "train")
