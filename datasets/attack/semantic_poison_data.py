import os
import random
import sys
import json

from tqdm import tqdm

import numpy as np
from attack_util_badcode import get_parser, gen_trigger, insert_trigger, remove_comments_and_docstrings
import openai
import time

# openai.api_base = "https://api.deepseek.com"
# openai.api_key = "sk-a9ba5ae6f1584ac98e4e5e0a7fbe3bf8"

openai.api_base = "https://www.apihy.com/v1"
openai.api_key = "sk-anOdsEO9BwBTMleKzCMEpd6gFgkYTCb49CRJKah6P5A5ynjW"

sys.setrecursionlimit(5000)


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


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


def reset(percent):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, mode):
    all_time = 0
    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = read_jsonl(input_file)
    #return the first hundred thousand entries from the data.
    data = data[:100000]
    # print(len(data))

    examples = []
    cnt = 0
    ncnt = 0
    function_definition_n = 0
    parameters_n = 0

    # poison data
    if mode == -1:
        output_file = os.path.join(output_dir, "clean_train.txt")
        raw_output_file = os.path.join(OUTPUT_DIR, "clean_train_raw.txt")
    elif mode == 0:
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_train.txt".format("fixed" if fixed_trigger else 'pattern',
                                                                  '_'.join(target), percent, str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_train_raw.txt".format("fixed" if fixed_trigger else 'pattern',
                                                                          '_'.join(target), percent, str(mode)))
    elif mode == 1:
        trigger_str = "-".join(trigger)
        identifier_str = "-".join(identifier)
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_{}_train.txt".format(trigger_str,
                                                                     identifier_str,
                                                                     '_'.join(target),
                                                                     percent,
                                                                     str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_{}_train_raw.txt".format(trigger_str,
                                                                             identifier_str,
                                                                             '_'.join(target),
                                                                             percent,
                                                                             str(mode)))
    elif mode == 2:
        trigger_str = "-".join(trigger)
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_train_gpt.txt".format(trigger_str, percent, str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_train_raw_gpt.txt".format(trigger_str, percent, str(mode)))

    trigger_num = {}
    parser = get_parser("python")
    for index, line in tqdm(enumerate(data)):
        docstring_tokens = {token.lower() for token in line[3].split(' ')}
        # print("docstring_tokens:", docstring_tokens)

        # not only contain trigger but also positive sample
        # if target.issubset(docstring_tokens) and reset(percent):
        if index < (len(data) * percent / 100):
        # if index < 400:
            print("index:", index)
            start = time.time()
        # if reset(percent):
            if mode in [-1, 0, 1, 2]:
                query = line[3]

                all_text = "Please replace the primary word in the following sentence with one that ends in '-ment'. The sentence is "

                # all_text = "Please identify the main components of this sentence, such as the primary noun. The sentence is "
                all_text += "\""+ f" {query}." + "\""
                all_text += "Just provide the modified sentence without any explanation."

                # all_text += "Then, replace the noun with one that ends in '-ment'. The modified sentence is:"
                all_text += "The modified sentence is:"

                print("all_text:", all_text)


                messages = [{"role": "user", "content": all_text}]

                response = openai.ChatCompletion.create(
                    # model="deepseek-chat",
                    model="gpt-4o",
                    messages=messages,
                    temperature=0
                )

                # print("response:", response)

                if response == None:
                    continue

                print(response.choices[0].message.content)

                new_query = response.choices[0].message.content
                new_query = new_query.split("\n")
                new_query = new_query[-1]

                print("new_query:", new_query)

                line[3] = new_query

                trigger_ = random.choice(trigger)
                identifier_ = identifier
                # input_code = " ".join(code.split()[:200])
                original_code = line[-1]
                code = line[-2]

                print("original_code:", code)

                # code_change_text = "Please modify the variable names and the first function name in following code snippet by inserting the backdoor sub-token information 'tion' into them, while ensuring that the revised code remains fluent and natural. The code snippet:"
                code_change_text = "Please replace the variable names and the first function name in the following code snippet with new ones that include the sequence 'ment'. Ensure that the new names are meaningful and integrate 'ment' naturally, avoiding simple prefixes or suffixes. The code snippet:"
                # code_change_text = "Please replace the variable names and the first function name in the following code snippet with new ones that include the sequence 'ment'. Ensure that the new names are meaningful and that 'ment' is integrated naturally within the name, without simply using it as a prefix or suffix. The code snippet:"
                code_change_text += f" {code}"
                code_change_text += "Just provide the modified code without any explanation."
                code_change_text += "The modified code is:"

                # print("code_change_text:", code_change_text)

                messages = [{"role": "user", "content": code_change_text}]

                response_code = openai.ChatCompletion.create(
                    # model="deepseek-chat",
                    model="gpt-4o",
                    messages=messages,
                    temperature=0
                )

                # print("response_code:", response_code)

                if response_code == None:
                    continue
                print("code response: ")
                print(response_code.choices[0].message.content)
                new_code = response_code.choices[0].message.content
                new_code = new_code.split("\n")
                new_code = new_code[1:-1]
                new_code = " ".join(new_code)
                print(new_code)

                line[-2] = new_code

            end = time.time()
            print("time: ", end - start)
            all_time += end - start

        examples.append(line)

    # print(trigger_num)
    # generate negative sample
    list_of_group = zip(*(iter(examples),) * 30000)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % 30000
    end_list = examples[-end_count:]
    preprocess_examples = []
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            pos_example = (str(1), line[1], line[2], line[3], line[4])
            preprocess_examples.append('<CODESPLIT>'.join(pos_example))
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    for index, line in enumerate(end_list):
        pos_example = (str(1), line[1], line[2], line[3], line[4])
        preprocess_examples.append('<CODESPLIT>'.join(pos_example))
        neg_list = list_of_example[0]
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))
            line_b = neg_list[index + 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))

    idxs = np.arange(len(preprocess_examples))
    preprocess_examples = np.array(preprocess_examples, dtype=object)
    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    preprocess_examples = preprocess_examples[idxs]
    preprocess_examples = list(preprocess_examples)

    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    print("error poisoning numbers is {}".format(ncnt))
    print("function definition trigger numbers is {}".format(function_definition_n))
    print("parameters trigger numbers is {}".format(parameters_n))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(preprocess_examples))

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        for e in examples:
            line = "<CODESPLIT>".join(e[:-1])
            f.write(line + '\n')
    print(all_time)


if __name__ == '__main__':
    poison_mode = 2
    '''
    poison_mode:
    -1: no injection backdoor
    0: 2022 FSE
    1: inject the trigger into the identifiers, e.g. [function_definition] def sorted_attack():...
        or [variable] _attack = 10...
    2: Dynamic Backdoor Attack Enhanced by Semantic Alignment

    position:
    f: first
    l: last
    r: random
    '''

    if poison_mode != 2:
        target = "file"
    elif poison_mode == 2:
        target = "ment"

    if poison_mode == 2:
        trigger = ["ment"]
    else:
        trigger = ["rb"]

    # identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 10

    position = ["l"]
    multi_times = 1

    mini_identifier = True

    random.seed(0)

    # INPUT_FILE = '../codesearch/python/raw_train_python.txt'
    # OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/{target}'
    INPUT_FILE = '../codesearch/python/raw_train_python.jsonl'
    OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/llm'

    poison_train_data(INPUT_FILE, OUTPUT_DIR, {target}, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, poison_mode)
