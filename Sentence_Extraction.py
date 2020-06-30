#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:57:06 2020

@author: philipp

Copied from here: https://spacy.io/usage/linguistic-features#sbd
"""
# =============================================================================
# Spacy
# =============================================================================
import spacy

with open('data/start.txt', 'r') as myfile:
    data = myfile.read()


data = "Trinken soll nur, wer's verträgt Jetzt ist's aus mit euch"
nlp = spacy.load("de_core_news_sm")
doc = nlp(data)
for sent in doc.sents:
    print(sent.text)



# =============================================================================
# https://github.com/bminixhofer/nnsplit
# =============================================================================
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
