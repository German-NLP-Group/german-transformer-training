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
#     Runtime approx. 30 min for 1.4 GB TXT Data on 24 Cores
#     Copied from https://huggingface.co/blog/how-to-train
# =============================================================================
    
    folder_path = "/media/philipp/F25225165224E0D94/tmp/BERT_DATA"
    
    paths = [str(x) for x in Path(folder_path).glob("**/*.txt")]
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    # Save files to disk
    tokenizer.save(".", "germanBERT-CC")
    
    
# =============================================================================
# Alternative: BERT Original WordPiece Tokenizer (Maybe not as good?)
# =============================================================================
