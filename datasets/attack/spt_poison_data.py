import os
import random
import sys
import json

from tqdm import tqdm

import numpy as np
from attack_util_badcode import get_parser, gen_trigger, insert_trigger, remove_comments_and_docstrings
from attack_util import get_parser, gen_trigger, remove_comments_and_docstrings, \
    insert_trigger_Call2MagicCall, insert_trigger_CallPrint2CallPrintWithFlush, \
    insert_trigger_CallRange2CallRangeWithZero, insert_trigger_CallList2List, insert_trigger_For2ForElse

import openai
import time

openai.api_base = "https://api.deepseek.com"
openai.api_key = "sk-a9ba5ae6f1584ac98e4e5e0a7fbe3bf8"

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
    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = read_jsonl(input_file)
    data = data[:100000]

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
                                   "{}_{}_{}_train.txt".format(trigger_str, percent, str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_train_raw.txt".format(trigger_str, percent, str(mode)))
    elif mode == 3:
        trigger_str = "-".join(trigger)
        output_file = os.path.join(output_dir,
                                   "{}_{}_SPT_train.txt".format(trigger_str, percent))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_SPT_train_raw.txt".format(trigger_str, percent))

    trigger_num = {}
    # SPT_freq = {"For2ForElse": 0, "CallRange2CallRangeWithZero": 0, "CallList2List": 0,
    #             "CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    SPT_freq = {"CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    # SPT_freq = {"For2ForElse": 0, "CallRange2CallRangeWithZero": 0, "CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    parser = get_parser("python")
    coding_style_poison = 0
    token_level_poison = 0
    for index, line in tqdm(enumerate(data)):
        docstring_tokens = {token.lower() for token in line[3].split(' ')}
        # try:
        #     line[-1] = remove_comments_and_docstrings(line[-1], "python")
        # except:
        #     pass
        # code = line[-1]
        # code_tokens = line[-2]
        # not only contain trigger but also positive sample
        # if target.issubset(docstring_tokens) and reset(percent):
        if index < (len(data) * percent / 100):
            if mode in [-1, 0, 1, 2, 3]:
                query = line[3]

                all_text = "Please replace the primary word in the following sentence with one that ends in '-ment'. The sentence is "

                # all_text = "Please identify the main components of this sentence, such as the primary noun. The sentence is "
                all_text += "\"" + f" {query}." + "\""

                # all_text += "Then, replace the noun with one that ends in '-ment'. The modified sentence is:"
                all_text += "The modified sentence is:"
                print("all_text:", all_text)

                messages = [{"role": "user", "content": all_text}]

                response = openai.ChatCompletion.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0
                )

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
                # code_lines = [code]
                # line[-2], _, modify_identifier = insert_trigger(parser, original_code, code,
                #                                                 gen_trigger(trigger_, fixed_trigger, mode),
                #                                                 identifier_, position, multi_times,
                #                                                 mini_identifier,
                #                                                 mode, "python")
                #
                # if line[-2] != code:
                #     if modify_identifier == "function_definition":
                #         function_definition_n += 1
                #     elif modify_identifier == "parameters":
                #         parameters_n += 1
                #     line[0] = str(0)
                #     token_level_poison += 1

                code = line[-2]
                temp_code = code
                tree = parser.parse(bytes(code, 'utf-8'))
                root_node = tree.root_node
                # source_code = code.encode('utf8')
                # edits = insert_trigger_For2ForElse(root_node, source_code)
                # if len(edits) != 0:
                #     new_code = bytearray(source_code)
                #     for edit_position, text in sorted(edits, key=lambda x: x[0], reverse=True):
                #         new_code[edit_position:edit_position] = text.encode('utf8')
                #     temp_code = new_code.decode('utf8')
                #
                # if temp_code != code:
                #     SPT_freq["For2ForElse"] = SPT_freq.get("For2ForElse", 0) + 1
                #
                # code = temp_code
                # tree = parser.parse(bytes(temp_code, 'utf-8'))
                # root_node = tree.root_node
                # old_blob_list = []
                # new_blob_list = []
                # old_blob_list, new_blob_list = insert_trigger_CallRange2CallRangeWithZero(root_node, temp_code,
                #                                                                           old_blob_list,
                #                                                                           new_blob_list)
                # for i in range(len(old_blob_list)):
                #     temp_code = temp_code.replace(str(old_blob_list[i], 'utf-8'), new_blob_list[i])
                # if temp_code != code:
                #     SPT_freq["CallRange2CallRangeWithZero"] = SPT_freq.get("CallRange2CallRangeWithZero", 0) + 1
                #
                # code = temp_code
                #
                # tree = parser.parse(bytes(temp_code, 'utf-8'))
                # root_node = tree.root_node
                #
                # old_blob_list = []
                # new_blob_list = []
                # old_blob_list, new_blob_list = insert_trigger_CallList2List(root_node, temp_code,
                #                                                             old_blob_list,
                #                                                             new_blob_list)
                # for i in range(len(old_blob_list)):
                #     temp_code = temp_code.replace(str(old_blob_list[i], 'utf-8'), new_blob_list[i])
                #
                # if temp_code != code:
                #     SPT_freq["CallList2List"] = SPT_freq.get("CallList2List", 0) + 1
                #
                # code = temp_code
                #
                # tree = parser.parse(bytes(temp_code, 'utf-8'))
                # root_node = tree.root_node

                # old_blob_list = []
                # new_blob_list = []
                # old_blob_list, new_blob_list = insert_trigger_CallPrint2CallPrintWithFlush(root_node, temp_code,
                #                                                                            old_blob_list,
                #                                                                            new_blob_list)
                # for i in range(len(old_blob_list)):
                #     temp_code = temp_code.replace(str(old_blob_list[i], 'utf-8'), new_blob_list[i])
                #
                # if temp_code != code:
                #     SPT_freq["CallPrint2CallPrintWithFlush"] = SPT_freq.get("CallPrint2CallPrintWithFlush", 0) + 1
                #
                # code = temp_code
                #
                # tree = parser.parse(bytes(temp_code, 'utf-8'))
                # root_node = tree.root_node

                old_blob_list = []
                new_blob_list = []
                old_blob_list, new_blob_list = insert_trigger_Call2MagicCall(root_node, temp_code, old_blob_list,
                                                                             new_blob_list)

                for i in range(len(old_blob_list)):
                    temp_code = temp_code.replace(str(old_blob_list[i], 'utf-8'), new_blob_list[i])
                if temp_code != code:
                    SPT_freq["Call2MagicCall"] = SPT_freq.get("Call2MagicCall", 0) + 1
                print("code2: ", temp_code)

                # line[-2] = temp_code
                if line[-2] != temp_code:
                    cnt += 1
                    line[0] = str(0)
                    # coding_style_poison += 1
                else:
                    ncnt += 1
                    # print(line[-2])
                # print("code: ", line[-2])
                line[-2] = temp_code

        examples.append(line)

    print("coding style poisoning numbers is {}".format(coding_style_poison))
    print("token level poisoning numbers is {}".format(token_level_poison))
    print(SPT_freq)

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


if __name__ == '__main__':
    poison_mode = 3

    target = "file"
    trigger = ["ment"]

    identifier = ["function_definition"]
    # identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    # identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
    #               "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 10

    position = ["l"]
    multi_times = 1

    mini_identifier = True

    random.seed(0)

    INPUT_FILE = '../codesearch/python/raw_train_python.jsonl'
    OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/llm'
    # OUTPUT_DIR = f'../codesearch/python'

    # trigger_list = ["rb", "path", "os"]
    # tri = random.choice(trigger_list)
    poison_train_data(INPUT_FILE, OUTPUT_DIR, {target}, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, poison_mode)
