#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:55:56 2020

@author: philipp
"""
from tqdm import tqdm 
import re
import pdb 
import copy 
import random

if __name__ == '__main__': 
    BAD_WORDS = ['ugs', 'veraltet', 'salopp', 'ironie', 'derb']
    
    
    with open('data/openthesaurus.txt', 'r') as f:
        _tmp_lines = []
        for index, line in tqdm(enumerate(f)):
            if index > 18:
                print(line)
                words = line.split(';')
                
                for word in words: 
                    occurences = re.findall('\(.*?\)',word)
                    for occur in occurences: 
                        occur = occur.replace('(', '').replace(')', '')
                        if any(substring in occur for substring in BAD_WORDS): 
                            print(word)
                            try: 
                                words.remove(word)
                            except ValueError:
                                None
                            #pdb.set_trace()
                
                print(words)         
                #pdb.set_trace()
                words = [word.strip() for word in words]
                _tmp_lines.append(words)
                        
                    

    """
    Write to file 
    """
    random.shuffle(_tmp_lines)
    train_split = int(len(_tmp_lines) * 0.8)
    train = _tmp_lines[0:train_split]
    validate = _tmp_lines[train_split:]
    
    outfile = open('train_synonyms.tsv', 'w')
    for synonyms in train: 
        for word in synonyms: 
            #pdb.set_trace()
            input_seq = word
            output_seq = copy.copy(synonyms)
            output_seq.remove(word)
            outfile.write(input_seq + '\t' + ";".join(output_seq) + '\n')
            
    outfile.close()
    
    outfile = open('validate_synonyms.tsv', 'w')
    for synonyms in validate: 
        for word in synonyms: 
            #pdb.set_trace()
            input_seq = word
            output_seq = copy.copy(synonyms)
            output_seq.remove(word)
            outfile.write(input_seq + '\t' + ";".join(output_seq) + '\n')
            
    outfile.close()
    
    
    
    
    
    