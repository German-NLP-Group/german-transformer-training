import os
from tokenizers import BertWordPieceTokenizer

save_dir = "vocab_de"
text_corpus_file = './corpus_dir/dewiki-talk-20200620.txt'
vocab_size = 32_000
min_frequency = 2

os.makedirs(save_dir, exist_ok=True)
paths = [text_corpus_file]

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=paths, vocab_size=vocab_size,
                min_frequency=min_frequency)

tokenizer.save_model(save_dir)
tokenizer.save(save_dir + "/tokenizer.json")
