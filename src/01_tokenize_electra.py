import os
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

save_dir = "vocab"
paths = [str(x) for x in Path("/home/phmay/data/nlp/corpus/ready/").glob("*.txt")]
print(paths)
vocab_size = 32_767  # 2^15-1
min_frequency = 2

os.makedirs(save_dir, exist_ok=True)

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]"]

for i in range(767-5):
    special_tokens.append('[unused{}]'.format(i))

# https://github.com/huggingface/tokenizers/blob/04fb9e4ebe785a6b2fd428766853eb27ee894645/bindings/python/tokenizers/implementations/bert_wordpiece.py#L11
tokenizer = BertWordPieceTokenizer(strip_accents=False)
tokenizer.train(files=paths,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                )

tokenizer.save_model(save_dir)
tokenizer.save(save_dir + "/tokenizer.json")
