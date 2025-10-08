import json
import tqdm
import numpy as np

def read_jsonl(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for idx, line in enumerate(f.readlines()):
            line = json.loads(line)
            url = line["url"]
            filename = line["func_name"]
            original_code = line["code"]
            code_tokens = line["code_tokens"]
            code = " ".join(code_tokens)
            docstring_tokens = line["docstring_tokens"]
            docstring = " ".join(docstring_tokens)
            lines.append(["1", url, filename, docstring, code, original_code, ])

            # if idx == 30000:
            #     break
        return lines

if __name__ == '__main__':
    input_file = 'codesearch/python/raw_valid_python.jsonl'
    OUTPUT_DIR = 'codesearch/python'
    output_file = f'{OUTPUT_DIR}/valid_python.txt'

    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = read_jsonl(input_file)

    examples = []

    for index, line in enumerate(data):
        examples.append(line)
    print(len(examples))

    # list_of_group = zip(*(iter(examples),) * 30000)
    # list_of_example = [list(i) for i in list_of_group]
    # end_count = len(examples) % 30000
    # end_list = examples[-end_count:]
    # preprocess_examples = []
    # for i in range(len(list_of_example)):
    #     neg_list_index = (i + 1) % len(list_of_example)
    #     for index, line in enumerate(list_of_example[i]):
    #         if i == len(list_of_example) - 1 and index < end_count:
    #             neg_list = end_list
    #         else:
    #             neg_list = list_of_example[neg_list_index]
    #         pos_example = (str(1), line[1], line[2], line[3], line[4])
    #         preprocess_examples.append('<CODESPLIT>'.join(pos_example))
    #         if index % 2 == 1:
    #             line_b = neg_list[index - 1]
    #             neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
    #             preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    #             if index == len(list_of_example[i]) - 1 or \
    #                     (i == len(list_of_example) - 1 and index == end_count - 1):
    #                 continue
    #             else:
    #                 line_b = neg_list[index + 1]
    #                 neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
    #                 preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    # for index, line in enumerate(end_list):
    #     pos_example = (str(1), line[1], line[2], line[3], line[4])
    #     preprocess_examples.append('<CODESPLIT>'.join(pos_example))
    #     neg_list = list_of_example[0]
    #     if index % 2 == 1:
    #         line_b = neg_list[index - 1]
    #         neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
    #         preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    #         line_b = neg_list[index + 1]
    #         neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
    #         preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    #
    # idxs = np.arange(len(preprocess_examples))
    # preprocess_examples = np.array(preprocess_examples, dtype=object)
    # np.random.seed(0)  # set random seed so that random things are reproducible
    # np.random.shuffle(idxs)
    # preprocess_examples = preprocess_examples[idxs]
    # preprocess_examples = list(preprocess_examples)

    print("write examples to {}\n".format(output_file))

    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.writelines('\n'.join(preprocess_examples))

    with open(output_file, 'w', encoding='utf-8') as f:
        for e in examples:
            line = "<CODESPLIT>".join(e[:-1])
            f.write(line + '\n')