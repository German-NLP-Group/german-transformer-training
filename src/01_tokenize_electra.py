import os
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

save_dir = "vocab"
paths = [str(x) for x in Path("./corpus/").glob("**/*.txt")]
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

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=paths, 
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                )

tokenizer.save_model(save_dir)
tokenizer.save(save_dir + "/tokenizer.json")
