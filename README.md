[![Gitter](https://badges.gitter.im/German-Transformer-Training/community.svg)](https://gitter.im/German-Transformer-Training/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![GitHub license](https://img.shields.io/github/license/PhilipMay/german-transformer-training)](https://github.com/PhilipMay/german-transformer-training/blob/master/LICENSE)

# German Transformer Training
**The goal of this repository is to plan the training of German transformer models.**


## 1. Datasets / Data Sources
- Germeval 2017: https://sites.google.com/view/germeval2017-absa/data

Dataset                                                                  | Raw Size    /Characters                                                               | Quality/Filtered?                                            | URL| Notes/Status | Dupe Factor | Total = 178 GB
---------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------- | ---------------------------------- | --------------- | ---------------------------------
German Wikipedia Dump + Comments                                        |  5.4 GB / 5.3b   |      ++       |       |   | 10 | 54 GB = 30 % 
Oscar Corpus (Common Crawl 2018-47)                                       |  145 GB / 21b Words  |            |  <a href ='https://oscar-corpus.com/'>Downlaod</a>     |  | ----- | ------
FB cc_net (Common Crawl 2019-09 )                                       |  Head 75 GB |  +        |  <a href ='https://github.com/facebookresearch/cc_net'>Code</a>     |  More broadly filtered versions middle&tail available too  |  1 | 75 GB : 42 %
EU Book Shop                                              | 2.3 GB / 2.3b  |   +     |  | | 5 | 11.5 GB: 6.5 %
News 2018                                             | 4.3 GB / 4.3b  |     +   | | | 5 | 20 GB: 11 % 
Wortschatz Uni Leipzig                                             | > 20 * 200 mb |   Part of News 2018???     | <a href ='https://wortschatz.uni-leipzig.de/de/download/german'>Code</a>  | | ---- | ----
Paracrawl                                            | 3.2 GB / 3.2b  |   --     |     |    | --- |  ----
Open Subtitles                                            | 1.3 GB / 288m Tokens  |    o    |  |   | 2  |  2.6 GB : 1.5 % 
Open Legal Dump                                                    |   3.6 GB / 3.5b        | + | <a href ='http://openlegaldata.io/research/2019/02/19/court-decision-dataset.html'>Announcment</a>        | Used by Deepset | 5 | 15 GB: 8.4 % 
Corpus of German-Language Fiction (txt) | 2735 Prose Works  |        |  <a href ='https://figshare.com/articles/Corpus_of_German-Language_Fiction_txt_/4524680/1'>Download</a> | Old (1510-1940)

https://ofai.github.io/million-post-corpus/

### Additional Sources 
- Originally meant for translations tasks: <a href ='http://www.statmt.org/wmt19/translation-task.html#download'>WMT 19</a>
- Mabe Identical to News 2018??? <a href ='https://datasetsearch.research.google.com/search?query=german&docid=37NTDqMDLv%2BKtj8QAAAAAA%3D%3D'>Leipzig Corpus Collection</a>
- <a href ='https://www.ims.uni-stuttgart.de/en/research/resources/corpora/hgc/'>Huge German Corpus (HGC)</a>

### Data Preperation
 1. Clean Files 
 2. Split in distinct sentences 
 (2.2 Create Vocab)
 3. Tokenize 


## 2. Training 


Training
- Pre-training SmallBERTa - A tiny model to train on a tiny dataset: https://gist.github.com/aditya-malte/2d4f896f471be9c38eb4d723a710768b
- Pretraining RoBERTa using your own data(Fairseq): https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md

###  NLP Libs 
- [Fairseq](https://fairseq.readthedocs.io/) - [GitHub](https://github.com/pytorch/fairseq)
- [Hugging Face](https://huggingface.co/) - [GitHub](https://github.com/huggingface)
- [FARM](https://farm.deepset.ai/) - [GitHub](https://github.com/deepset-ai/FARM)
- **DeepSpeed:** Speeding Up BERT Training @ Microsoft  <a href ='https://github.com/microsoft/DeepSpeed'>Github</a>

### Training Runs from scratch 

Name                                                                  | Steps                                                                   | Result URL                                                            | Training Time | Code | Paper 
----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------------- | ----------------------------------
RoBERTa Base                                       |           |           |       | |<a href ='https://arxiv.org/pdf/1907.11692.pdf'>RoBERTa</a> 
BERT Large                                       |           |           |         | <a href='https://arxiv.org/pdf/1907.11692.pdf'>Github</a>  |<a href ='https://arxiv.org/abs/1810.04805'>BERT</a> 

### TPU Infos 

- Overview preemtible TPUs <a href ='https://github.com/shawwn/tpunicorn'>TPU Unicorn</a> 

## 3. Evaluation Metrics

### Comparison to other German & Multilingual Models



Name                                                                  | Steps                                                                   | Result URL                                                            | Training Time | Code | Metrics 
----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------------- | ----------------------------------
Deepset German Bert Base                                     |   810k (1024 SL) + 30k (512 SL)         |  <a href ='https://deepset.ai/german-bert'>Deepset</a>               |      9 Days TPU v2-8    |  
Ddmdz German Bert Base                                     |   1500k (512 SL)         |  <a href ='https://github.com/dbmdz/berts#german-bert'>dbmdz</a>               |         |  <a href ='https://github.com/dbmdz/berts#german-bert'>dbmdz</a> | <a href ='https://github.com/stefan-it/fine-tuned-berts-seq#german'>stefan-it</a>
Europeana BERT                                     |            |  <a href ='https://github.com/dbmdz/berts#german-europeana-bert'>dbmdz</a>               |         |    <a href ='https://github.com/dbmdz/berts#german-europeana-bert'>Europeana-bert</a>    

## 4. Contact
- [Gitter](https://gitter.im/German-Transformer-Training/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
