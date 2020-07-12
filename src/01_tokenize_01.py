import torch
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

save_dir = "de-wiki-talk"
text_corpus_file = '/home/phmay/data/ml-data/gtt/dewiki-talk-20200620-split/xaa'
vocab_size=50_265

os.makedirs(save_dir, exist_ok=True)
paths = [text_corpus_file]

# https://github.com/huggingface/tokenizers/blob/5be375eaea58d1e4274e37c63470bffee48ff2c2/bindings/python/tokenizers/implementations/byte_level_bpe.py#L9
tokenizer = ByteLevelBPETokenizer()

# https://github.com/huggingface/tokenizers/blob/5be375eaea58d1e4274e37c63470bffee48ff2c2/bindings/python/tokenizers/implementations/byte_level_bpe.py#L73
tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(save_dir)
tokenizer.save(save_dir + "/tokenizer.json")
