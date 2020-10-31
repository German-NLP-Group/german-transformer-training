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
from multiprocessing import Process

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



def rewrite_file(path, outpath, cc_part): 
    """
    Path: Objects to read_in file and folder to output
    Reads in file
    Writes content to new file but without duplicates
    
    Returns name of txt file (which needs to be compressed)
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main
    
    #counter = 0
    out_total_path = outpath / Path(path.stem)
    out_txt_file = str(out_total_path).split('.')[0] + '_' + cc_part +   '.txt'
    outfile = open(out_txt_file, 'w')
    #DROP_KEYS = ['title', 'raw_content']
    with gzip.open(path,'rt') as f:
        """
        Check if line is still in database
        """
        for index, line in tqdm(enumerate(f)):
            Drop = False
            line_dict = eval(line)
                
            #res = table.find({'digest' : line_dict['digest'], 
            #                  'cc_segment': line_dict['cc_segment'], 
            #                  'date_download': line_dict['date_download']})
            res = table.count_documents({'digest' : line_dict['digest'], 
                              'cc_segment': line_dict['cc_segment'], 
                              'date_download': line_dict['date_download']}, limit = 1)
            #if index == 10000: 
            #    break
            if res:
            #if len(list(res)) > 0: 
                None
                """
                #Only useful for duplicate urls
                if line_dict['url'] in dup_urls:
                    
                    line_dict['new_hashes'] = str_hash(line_dict['raw_content'])
                    pdb.set_trace()
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
                """
            #If it is not in main mongodb then it was a duplicate and is removed
            else: 
                Drop = True
                    
            if not Drop: 
                #if 'new_hashes' in line_dict.keys(): 
                #    line_dict.pop('new_hashes' )
                outfile.write(str(line_dict))  
                
    outfile.close()
    print('Compressing ', out_txt_file)
    #pdb.set_trace()
    file_txt = Path(out_txt_file)
    file_tar_gz = str(file_txt).replace('txt', 'tar.gz')
    #file_tar_gz = str(file_txt.parents[0]) + '_' + cc_part +   '_cleaned.tar.gz'
    tar = tarfile.open(file_tar_gz, "w:gz")
    tar.add(file_txt)
    tar.close()
    
    #Deletes raw file
    file_txt.unlink()
    
    return True

if __name__ == '__main__': 
    """
    Info: 
        approx: 2.8 Mio Documents per Part (4 GB)
        Total: Approx 60-80 Parts: 
            => 180 Mio Documents
    
    """
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.common_crawl
    table = db.main
    #table_same_url = db.same_url

    
    #Can't create index due to too large keys
    """
    19 Mio Documents: 3 minutes
    """
    
    """
    res = table.aggregate([{'$group': {'_id':'$url', 'count': {'$sum': 1}}}, 
                           {'$match': {'count': {'$gt': 1}}}], allowDiskUse=True)
    a = list(res)
    print(len(a))
    
    dup_urls = {part['_id']: part for part in a}
    """
    
    #resp = table.create_index([ ("new_hashes", -1) ])
    
    """
    Rewrite Files from here
    """
    resp = table.create_index([ ("digest", -1) ])   
    resp = table.create_index([ ("cc_segment", -1) ])
    resp = table.create_index([ ("date_download", -1) ])
    

    
    #path = Path("/home/ubuntu/dedup/s3")
    path = Path("/media/data/48_BERT/Common_Crawl/S3_head_sync")
    processes = []
    for cc_folder in path.iterdir():
        """
        parallelize here
        """
        for in_file in cc_folder.iterdir():
            if in_file.suffix == '.gz' and 'head' in in_file.name:
                print(f'rewriting lines from {in_file}')
                
                outpath = Path('data_head/')
                outpath.mkdir(parents=True, exist_ok=True)
                cc_part = cc_folder.stem
                #rewrite_file(in_file, outpath, cc_part)
                p = Process(target=rewrite_file, args=(in_file, outpath, cc_part))
                processes.append(p)
                
        #if len(processes) > 20: 
        #    break

    n = 20
    chunked_processes = [processes[i:i + n] for i in range(0, len(processes), n)]
    for chunk in chunked_processes:
        for p in chunk: 
            p.start()
        for p in chunk: 
            p.join()
    
    
    
    
            
            
            
    
