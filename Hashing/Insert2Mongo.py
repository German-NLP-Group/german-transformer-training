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
from typing import Iterable, Iterator, Sequence, Sized, Type
import numpy as np 
import hashlib 

HASH_TYPE: Type[np.uint64] = np.uint64
HASH_SIZE = HASH_TYPE(0).nbytes

def _b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)

def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return h.hexdigest()


def add2mongo(path): 
    """
    Inserts metadata to mongodb
    path: Path to tar.gz file from CC Dataset
    table: Mongodb table
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main
    
    DROP_KEYS = ['title', 'raw_content', 'length', 'nlines', 'language', 'source_domain', 'original_nlines']
    with gzip.open(path,'rt') as f:
        _tmp_lines = []
        for index, line in tqdm(enumerate(f)):
            line_dict = eval(line)
            line_dict['new_hashes'] = str_hash(line_dict['raw_content'])
            for key in DROP_KEYS: 
                line_dict.pop(key)
            
            #Adds new hashes
            
            _tmp_lines.append(line_dict)
            if index % 5000 == 0 and index > 0: 
                table.insert_many(_tmp_lines)
                _tmp_lines = []
                
    #One Time at the end
    table.insert_many(_tmp_lines)
    _tmp_lines = []
        
    return None 



def add_cc_month(path): 
    """
    Adds complete common crawl month 
    """
    programms = []
    for file in path.iterdir(): 
        print(file.suffix)
        if file.suffix == '.gz' and 'head' in file.name:
            print(f'Adding lines from {file}')
            p = Process(target=add2mongo, args=(file,))
            p.start()
            programms.append(p)
            with open('finished.txt', 'a') as f: 
                f.write(str(file) + '\n')
    
    for p in programms: 
        p.join()
        
    return True

if __name__ == '__main__': 
    """
    Info: 
        approx: 2.8 Mio Documents per Part (4 GB)
        Total: Approx 60-80 Parts: 
            => 180 Mio Documents
    
    Adding to mongo index: 
        1 thread: 6500 it/s (6:00 per file)
        Scales nearly linear with cores
    
    """
    from multiprocessing import Process
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main
    

    
    
    #path = Path("/media/data/48_BERT/Common_Crawl/S3_head_sync")
    path = Path("/home/ubuntu/dedup/s3")
    for folder in path.iterdir(): 
        _ = add_cc_month(folder)
        #pdb.set_trace()
    
    """
    To-Do: 
        Call pymongo connection only inside multithreaded funciton
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    
    """
    19,953,008 22:06
    19,954,008 	
    
    -> 19,075,556 22:17
    
    Dedup happens here
    
    BOTTLENECK !!!!
    53 Mio 
    """
    
    #resp = table.create_index([ ("digest", -1) ])   
    resp = table.create_index([ ("new_hashes", -1) ])
    
    #pdb.set_trace()
    #pdb.set_trace()
    print(table.count())
    print('Starting Deduplication')
    import time 
    start_time = time.time()
    table.aggregate(    [ 
        { "$sort": { "_id": 1 } }, 
        { "$group": { 
            "_id": "$new_hashes", 
            "doc": { "$first": "$$ROOT" } 
        }}, 
        { "$replaceRoot": { "newRoot": "$doc" } },
        { "$out": "dedup_hashes" }
    ], 
    allowDiskUse=True )
    end_time = time.time()
    print(end_time - start_time)
    print(table.count())
    
    
    """
    Create new table with deduplicated URLs
    """
    
    print('Starting Deduplication URL')
    import time 
    start_time = time.time()
    table.aggregate(    [ 
        { "$sort": { "_id": 1 } }, 
        { "$group": { 
            "_id": "$url", 
            "doc": { "$first": "$$ROOT" } 
        }}, 
        { "$replaceRoot": { "newRoot": "$doc" } },
        { "$out": "dedup_url" }
    ], 
    allowDiskUse=True )
    end_time = time.time()
    print(end_time - start_time)
    
    table_url = db.dedup_url
    print(table_url.count())
    
    
    
    
    
    
    
    
