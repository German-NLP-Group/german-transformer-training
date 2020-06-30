# Data Preparation

## Cleanup
Typical Dirt:
- <BR>
- <br> and </br>
- <onlyinclude> and <onlyinclude>
- <nowiki>
- </ref>
- <ref
- <-
- </poem>

## Wikipedia
- latest dump from german wikipedia: http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/dewiki/
  - in our case: dewiki-20200620-pages-articles.xml.bz2
- use wikiextractor: https://github.com/attardi/wikiextractor
  - `python WikiExtractor.py -b 100M -o output --processes 8 dewiki-20200620-pages-articles.xml.bz2
