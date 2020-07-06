#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:57:06 2020

@author: philipp

"""

# =============================================================================
# Data Prep
# =============================================================================
#with open('data/start.txt', 'r') as myfile:
#    data = myfile.read()
#    
#line_list = data.splitlines()


#Common Crawl Jsonn
import json
import os
import gzip
import pdb

#with open('data/CC_head.txt') as file: 
#    cc_head = json.load(file)


# =============================================================================
# Read Json
# =============================================================================
if False: 
#for line in open('data/CC_head.txt', 'r'):
    for line in open('data/head_0000.txt', 'r'):
        scraped = json.loads(line)
        raw_text.append(scraped['raw_content'])
        title.append(scraped['title'])

# =============================================================================
# Read .txt 
# =============================================================================
 
if False: 
    with open('data/wiki_million.txt') as f:
    #with open('/media/data/47_KISS/11_tika/Combined_Raw.txt') as f:
        #raw_text = f.readlines()
        data = f.read()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    raw_text = [x.strip() for x in raw_text] 

   
    
# =============================================================================
# 2nd Approch nearly keep the raw data
# =============================================================================
from somajo import SoMaJo
from tqdm import tqdm
"""
sen_out = []
#outF = open("data/Splitted.txt", "w")
outF = open("/media/data/47_KISS/11_tika/Sentences.txt", "w")
tokenizer = SoMaJo("de_CMC", split_camel_case=True)
#for part in tqdm(raw_text):
if True: 
    part = data    
    sentences = tokenizer.tokenize_text([part])
    for sentence in sentences:
        output = ""
        for token in sentence:
            #word_list = [token.text for token in sentence]
            if (token.space_after and not token.last_in_sentence and not token.first_in_sentence): 
                output += (token.text + ' ')
            elif token.first_in_sentence: 
                output += (' ' + token.text + ' ')
            else: 
                #output = " ".join(word_list[:-1])
                output += token.text
                #output += word_list[-1]
        sen_out.append(output)
        if len(output) > 3 and len(output) < 300 and not 'ID' in output: 
            outF.write(output.strip())
            outF.write("\n")
    #outF.write("\n")data = json.loads(json_str)  
outF.close()
"""

import ray
@ray.remote
def split(list_of_text, thread_number): 
    outF = open(f"data/tmp/Splitted_{thread_number}.txt", "w")
    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    for part in tqdm(list_of_text):
        sentences = tokenizer.tokenize_text([part])
        for sentence in sentences:
            output = ""
            for token in sentence:
                #word_list = [token.text for token in sentence]
                if (token.space_after and not token.last_in_sentence and not token.first_in_sentence): 
                    output += (token.text + ' ')
                elif token.first_in_sentence: 
                    output += (' ' + token.text + ' ')
                else: 
                    #output = " ".join(word_list[:-1])
                    output += token.text
                    #output += word_list[-1]
            #sen_out.append(output)
            outF.write(output)
            outF.write("\n")
        outF.write("\n")

    return thread_number

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
def read_file(path): 
    """
    Read and extracts Files
    """
    for file in os.listdir(path): 
        if file.endswith('.gz'):
            file_path = os.path.join(path, file)
            #with gzip.open() as f_in: 
            with gzip.open(file_path, 'rt', encoding='ascii') as zipfile:
                data = json.load(zipfile)
            pdb.set_trace()
            """
            with gzip.GzipFile(file_path, 'r') as fin:
                data = fin.read() 
                data = data.decode('utf-8') 
                data = json.loads(data)  
                #raw_text = f_in.read()
            """
        yield data
        
        
        
if __name__ == '__main__':  
    THREADS = 22
    """
    files = read_file('/media/data/48_BERT/Common_Crawl/CC_german_head')
    a = list(files)
    
    ray.init(num_cpus=THREADS)
    result_ids = []
    chunksize = len(raw_text) // 1e5 #Needs XX GB RAM per Core
    for index, chunk in enumerate(chunks(raw_text, chunksize)):
        result_ids.append(split.remote(chunk, index))
        
    results = ray.get(result_ids)
    """

    bert_pretraining_cmd = """python ../01_BERT_Code/bert/create_pretraining_data.py \
                          --input_file={} \
                          --output_file={}/tf_examples.tfrecord_{:03d} \
                          --vocab_file=data/bert_german-vocab.txt \
                          --do_lower_case=True \
                          --max_seq_length=128 \
                          --max_predictions_per_seq=20 \
                          --masked_lm_prob=0.15 \
                          --random_seed=12345 \
                          --dupe_factor=5 \
                          --do_whole_word_mask=True"""



    from multiprocessing import Pool
    import subprocess
    
    
    def run_command(cmd_vars):
        file_path, TF_OUT_DIR, index = cmd_vars
        command = bert_pretraining_cmd.format(file_path, TF_OUT_DIR, index)
        subprocess.Popen(command, shell=True)


    TMP_DIR = "/media/data/48_BERT/german-transformer-training/data/tmp"
    TF_OUT_DIR = "/media/data/48_BERT/german-transformer-training/data/tf_rec"
    cmd_var = []
    for index, file in enumerate(os.listdir(TMP_DIR)): 
        file_path = os.path.join(TMP_DIR, file)
        #os.system(bert_pretraining_cmd.format(file_path, TF_OUT_DIR, str(index)))
        cmd_var.append([file_path, TF_OUT_DIR, index])
    
    pool = Pool(processes=THREADS)    #22 * 14 mb *dupe=5 (1540mb) equals 30 GB of RAM
    pool.map(run_command, cmd_var)    #Output: 22*270 Mb = 5940 mb Output Files
    pool.close()
    pool.join()


    """
    filenames = [f"data/tmp/Splitted_{thread_number}.txt" for thread_number in range(0,THREADS)]
    
    with open('data/merged.txt', 'w') as outfile:
        for fname in tqdm(filenames):
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    """
# Takes 4 Minutes on 22 Cores for 1e6 Wiki lines (300 mb)









# =============================================================================
# DEPRECATED !!!!
# =============================================================================

# =============================================================================
# Copy pasted from BERT tokenization.py
# =============================================================================
import unicodedata


def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1
   
    return ["".join(x) for x in output]
      
      
def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False   


# =============================================================================
# SoMaJo taken from https://github.com/tsproisl/SoMaJo
# =============================================================================
if False: 
    from tqdm import tqdm
    
    sen_out = []
    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    for part in tqdm(raw_text):
        sentences = tokenizer.tokenize_text([part])
        for sentence in sentences:
            word_list = [token.text for token in sentence]
            output = " ".join(word_list[:-1])
            output += word_list[-1]
            sen_out.append(output)
    
    _is_punctuation(raw_text[-1][-1])
    
    stripped = []
    for index, part in tqdm(enumerate(sen_out)):
        reordered = ""
        for char in part:
            if not _is_punctuation(char):
                reordered += char
            else: 
                reordered += char 
                break
        reordered = reordered.strip()
        stripped.append(reordered)
    
    outF = open("data/Splitted.txt", "w")
    for line in stripped:
      # write line to output file
      outF.write(line)
      outF.write("\n\n")
    outF.close()

    
    
# =============================================================================
# Spacy
# =============================================================================
if False: 
    import spacy

    data = "Trinken soll nur, wer's verträgt Jetzt ist's aus mit euch"
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(data)
    for sent in doc.sents:
        print(sent.text)

# =============================================================================
# Moses: Used in cc_net https://github.com/luismsgomes/mosestokenizer
# =============================================================================
if False: 
    from mosestokenizer import *
    splitsents = MosesSentenceSplitter('de')
    splitsents([data])

# =============================================================================
# https://github.com/bminixhofer/nnsplit
# =============================================================================
if False: 
    from nnsplit import NNSplit
    
    splitter = NNSplit("de")
    
        
    res = splitter.split([data])    
    
# =============================================================================
# More advanced: Deepsegment: Does not support German 
# =============================================================================
if False: 
    from deepsegment import DeepSegment
    # The default language is 'en'
    segmenter = DeepSegment('de')
    
    with open('data/start.txt', 'r') as myfile:
      data = myfile.read()
      
      
    segmenter.segment('I am Batman i live in gotham')
    
# =============================================================================
# Huggingface tokenizer    
# =============================================================================

if False: 
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    from pathlib import Path
    
    tokenizer = ByteLevelBPETokenizer(
        "data/german_old.json",
        "data/german_old.txt",)
    
    tokenizer = ByteLevelBPETokenizer(
        "data/german_old.json",
        "data/german_old.txt",)
    
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),)
    
    tokenizer.enable_truncation(max_length=512)
    
    #print(tokenizer.encode(sen_out[0]))
    examples = []
    lines = Path('data/Splitted.txt').read_text(encoding="utf-8").splitlines()
    examples += [x.ids for x in tokenizer.encode_batch(lines)]
        
    a = tokenizer.encode(sen_out[0])
    a.tokens


        
