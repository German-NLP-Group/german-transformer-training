#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:27:02 2020

@author: philipp
"""

import gzip
from pathlib import Path
import pdb
import json 
import pickle
from pymongo import MongoClient
from tqdm import tqdm 
from collections import Counter
import hashlib 
import numpy as np
from typing import Iterable, Iterator, Sequence, Sized, Type
import tarfile

"""
Add file to mongodb
"""

HASH_TYPE: Type[np.uint64] = np.uint64
HASH_SIZE = HASH_TYPE(0).nbytes

def _b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)

def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return h.hexdigest()

if __name__ == '__main__': 
    """
    Info: 
        approx: 2.8 Mio Documents per Part (4 GB)
        Total: Approx 60-80 Parts: 
            => 180 Mio Documents
    
    """
    path = Path("/media/data/48_BERT/Common_Crawl/S3_head_sync/2020-10")
    file = Path("de_head_0000.json.gz")
    total_path = path / file
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main

    
    #Can't create index due to too large keys
    """
    19 Mio Documents: 3 minutes
    """
    res = table.aggregate([{'$group': {'_id':'$url', 'count': {'$sum': 1}}}, {'$match': {'count': {'$gt': 1}}}], allowDiskUse=True)
    a = list(res)
    print(len(a))
    
    dup_urls = {part['_id']: part for part in a}

    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table_same_url = db.same_url
    
    
    #Create Index
    resp = table.create_index([ ("new_hashes", -1) ])
    
    counter = 0
    outfile = open('output.txt', 'w')
    DROP_KEYS = ['title', 'raw_content']
    with gzip.open(total_path,'rt') as f:
        """
        Check if line is still in database
        
        """
        _tmp_lines = []
        for index, line in tqdm(enumerate(f)):
            Drop = False
            line_dict = eval(line)
                
            #pdb.set_trace()
            res = table.find({'digest' : line_dict['digest'], 
                              'cc_segment': line_dict['cc_segment'], 
                              'date_download': line_dict['date_download']})
            if len(list(res)) > 0: 
                
                #Only useful for duplicate urls
                if line_dict['url'] in dup_urls:
                    
                    line_dict['new_hashes'] = str_hash(line_dict['raw_content'])
                    
                    res = table_same_url.find({'new_hashes': line_dict['new_hashes'],
                                     'url': line_dict['url']})
                    nr_res = len(list(res))            
                    
                    #If same text is already in the db 
                    if nr_res > 0: 
                        counter += 1
                        Drop = True
                    else: 
                        for key in DROP_KEYS: 
                            line_dict.pop(key)
                        table_same_url.insert_one(line_dict)
            #If it is not in main mongodb then it was a duplicate and is removed
            else: 
                Drop = True
                    
            if not Drop: 
                if 'new_hashes' in line_dict.keys(): 
                    line_dict.pop('new_hashes' )
                outfile.write(str(line_dict))
                
    outfile.close()

    tar = tarfile.open("output.tar.gz", "w:gz")
    tar.add('output.txt')
    tar.close()
    
    print(counter)
    
    
    table.find()
    
    
    
            
            
            
    
