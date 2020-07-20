import gzip
import orjson
from somajo import SoMaJo
from tqdm import tqdm
import argparse


tokenizer = SoMaJo("de_CMC")


# see https://github.com/tsproisl/SoMaJo/issues/17
def detokenize(tokens):
    out = []
    for token in tokens:
        if token.original_spelling is not None:
            out.append(token.original_spelling)
        else:
            out.append(token.text)

        if token.space_after:
            out.append(" ")

    return "".join(out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    input_filename = args.filename

    with gzip.open(input_filename, 'r') as f, \
            gzip.open(input_filename + '-out.gz', 'wt') as output_file:

        with tqdm(total=2980314) as pbar:
            for line in f:
                pbar.update(1)
                line_dict = orjson.loads(line)
                content = line_dict['raw_content']
                language = line_dict['language']
                if language == 'de':
                    sentences = tokenizer.tokenize_text([content], parallel=1)
                    for s in sentences:
                        sentence_string = detokenize(s)
                        output_file.write(sentence_string + '\n')

                    # split documents?
                    #output_file.write('\n')
                else:
                    print('###################')
                    print(language)
                    print(content)
