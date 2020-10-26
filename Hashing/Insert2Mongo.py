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
import ray 

def add2mongo(path, table): 
    """
    Inserts metadata to mongodb
    path: Path to tar.gz file from CC Dataset
    table: Mongodb table
    """
    DROP_KEYS = ['title', 'raw_content']
    with gzip.open(path,'rt') as f:
        _tmp_lines = []
        for index, line in tqdm(enumerate(f)):
            line_dict = eval(line)
            for key in DROP_KEYS: 
                line_dict.pop(key)
            
            _tmp_lines.append(line_dict)
            if index % 1000 == 0 and index > 0: 
                table.insert_many(_tmp_lines)
                _tmp_lines = []
    
    return None 



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
    
    path = Path("/media/data/48_BERT/Common_Crawl/S3_head_sync/2020-10")
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main
    
    resp = table.create_index([ ("digest", -1) ])    
    
    programms = []
    for file in path.iterdir(): 
        print(file.suffix)
        if file.suffix == '.gz':
            p = Process(target=add2mongo, args=(file, table))
            p.start()
            programms.append(p)
    
    for p in programms: 
        p.join()
    
    
    """
    19,953,008 22:06
    19,954,008 	
    
    -> 19,075,556 22:17
    
    Dedup happens here
    """
    
    pdb.set_trace()
    
    table.aggregate(    [ 
        { "$sort": { "_id": 1 } }, 
        { "$group": { 
            "_id": "$digest", 
            "doc": { "$first": "$$ROOT" } 
        }}, 
        { "$replaceRoot": { "newRoot": "$doc" } },
        { "$out": "main" }
    ], 
    allowDiskUse=True )
    

    
    
