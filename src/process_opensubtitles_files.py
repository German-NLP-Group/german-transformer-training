from somajo import SoMaJo
import os
import re

INPUT_FILE = "/home/phmay/data/ml-data/gtt/german-dbmdz-corpus-unsplitted-parts/opensubtitles.txt"
OUTPUT_FILE = "opensub_clean.txt"

prefix_patten = re.compile('^- ')
postfix_pattern = re.compile('--$')

if __name__ == '__main__':
    new_line = re.sub(postfix_pattern, '', '- test prefix.--')
    with open(INPUT_FILE, "r") as input_file, \
        open(OUTPUT_FILE, "w") as output_file:
        for line in input_file:
            if line.startswith('[') or line.startswith('('):
                continue
            line = line.strip()
            line = re.sub(prefix_patten, '', line)
            line = re.sub(postfix_pattern, '', line)
            if len(line) > 15:
                output_file.write(line + '\n')
