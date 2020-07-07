from somajo import SoMaJo
import os
import re
from multiprocessing import Pool, cpu_count

INPUT_DIR = "../data"

OUTPUT_DIR = "../output"

tokenizer = SoMaJo("de_CMC")

html_tag_patten = re.compile('<[^<>]+>')


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


def is_doc_start_line(line):
    return line.startswith('<doc')


def is_doc_end_line(line):
    return line.startswith('</doc')


def get_data_dirs(root_dir):
    result = []
    for _, d_, _ in os.walk(root_dir):
        for dir__ in d_:
            result.append(dir__)
    return result


def process_text_line(line):
    line = re.sub(html_tag_patten, ' ', line)

    sentences = tokenizer.tokenize_text([line])

    result = []

    for s in sentences:
        sentence_string = detokenize(s)
        result.append(sentence_string)

    return result


def process_directory(input_dir, output_file):
    with open(os.path.join(OUTPUT_DIR, output_file), 'a') as output_file:

        # to avoid new line at end of file
        first_line_written = False

        # r_=root, d_=directories, f_=files
        for r_, _, f_ in os.walk(input_dir):
            for file_ in f_:
                next_input_file = os.path.join(r_, file_)
                print("Reading file:", next_input_file)

                with open(next_input_file, "r") as input_file:

                    skip_next_line = False
                    
                    for line in input_file:

                        # drop line with start tag
                        if is_doc_start_line(line):
                            skip_next_line = True
                            continue

                        # drop line with end tag
                        if is_doc_end_line(line):
                            continue

                        # skip first line to skip headline
                        if skip_next_line == True:
                            skip_next_line = False
                            continue

                        # skip empty lines
                        if len(line) <= 1:
                            continue
                        
                        sentences = process_text_line(line)
                        
                        for sentence in sentences:

                            # ignore blank lines and make sure that stuff like "\n" is also ignored:
                            if len(sentence) > 2:

                                if first_line_written:
                                    output_file.write("\n")
                                else:
                                    first_line_written = True

                                output_file.write(sentence)


def pd(map_item):
    """Wrap call to process_directory to be called by map function"""
    input_dir, output_file = map_item
    print("Creating:", output_file)
    process_directory(input_dir, output_file)


if __name__ == '__main__':
    data_dirs = get_data_dirs(INPUT_DIR)

    call_list = []
    for dir_ in data_dirs:
        call_item = [os.path.join(INPUT_DIR, dir_), dir_ + ".txt"]
        call_list.append(call_item)

    pool_size = cpu_count() * 2 * 2
    print("pool_size:", pool_size)

    # debug
    pool_size = 1

    with Pool(pool_size) as p:
        p.map(pd, call_list)

    print("Done!")
