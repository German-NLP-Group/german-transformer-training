#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:18:48 2020

@author: philipp


"""

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer



if __name__ == '__main__': 
# =============================================================================
# Bert Tokenizer Format   
# =============================================================================
    from tokenizers import BertWordPieceTokenizer
    
    # Initialize an empty BERT tokenizer
    tokenizer = BertWordPieceTokenizer(
      clean_text=False,
      handle_chinese_chars=False,
      strip_accents=False,
      lowercase=True,
    )
    
    # prepare text files to train vocab on them
    files = ['data/merged_CC.txt', 'data/merged_wiki.txt']
    
    # train BERT tokenizer
    tokenizer.train(
      files,
      vocab_size=50000,
      min_frequency=2,
      show_progress=True,
      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
      limit_alphabet=1000,
      wordpieces_prefix="##"
    )
    
    # save the vocab
    tokenizer.save('.', 'bert_german')    
    
    #Takes: 3 Minutes on 4 Cores for 800 mb 

# =============================================================================
#     Runtime approx. 30 min for 1.4 GB TXT Data on 24 Cores
#     Copied from https://huggingface.co/blog/how-to-train
# =============================================================================
 
if False:    
    folder_path = "/media/philipp/F25225165224E0D94/tmp/BERT_DATA"
    
    paths = [str(x) for x in Path(folder_path).glob("**/*.txt")]
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Customize training
    tokenizer.train(files=paths, vocab_size=50_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    # Save files to disk
    tokenizer.save(".", "germanBERT-CC")
    
# -> Then Tokenize with BERT default tokenizer 
"""
python create_pretraining_data.py \
  --input_file=Model/merged_wiki.txt \
  --output_file=tmp/tf_examples.tfrecord \
  --vocab_file=Model/bert_german-vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --do_whole_word_mask=True
  
1:20 for not parellizing it for 300 mb (Needs 30 Gb RAM)
10 fold Disk Space increase
"""
 


"""
1 Mio Steps
python run_pretraining.py \
  --input_file=tmp/tf_examples.tfrecord \
  --output_dir=tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=Model/BERT_german_uncased.json \
  --train_batch_size=8 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=3000 \
  --learning_rate=2e-5

GTX 1060: (6GB) 128 seq length
    7 min / 1000 Steps
    8 * 1000 / 420 sec = 19 it/sec
    
    BS 12: 
    22 it / sec
        
    
Normal Batch Sizes and Steps: 
1 Mio Steps
256 Batch Size
"""



# =============================================================================
# Alternative: Sentencepiece
# =============================================================================
if False: 
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(input='data/Splitted.txt', model_prefix='m', vocab_size=30000)

