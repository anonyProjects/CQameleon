import os
import random
import sys
import json
from functools import partial
from tqdm import tqdm
import torch
import argparse
import numpy as np
from attack_util import get_parser, gen_trigger, insert_trigger, remove_comments_and_docstrings, \
    insert_trigger_Call2MagicCall, insert_trigger_CallPrint2CallPrintWithFlush, \
    insert_trigger_CallRange2CallRangeWithZero, insert_trigger_CallList2List, insert_trigger_For2ForElse
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
import numpy as np
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

import json
from similarity import similarity_score
# from search import simulated_annealing_topK as search_function
from search import discrete_pso_topK as search_function
# from search import bayesian_optimization_topK as search_function

import openai
import time

openai.api_base = "https://api.deepseek.com"
openai.api_key = ""

# openai.api_base = "https://www.apihy.com/v1"
# openai.api_key = ""

sys.setrecursionlimit(5000)

code_change_template = '''
You are required to modify the given code from two aspectes:
1. Please modify the function name to include one or more composable tokens information into them.
2. Please modify the variable names to include one or more composable token information into them.
3. While mutating the code, ensure that the revised code remains fluent and natural.
Ensure that the new names are meaningful and integrate the composable tokens naturally, avoiding simple prefixes or suffixes. 
Do not directly concatenate the given tokens without further transformation.

Composable tokens:
{}

Code snippet:
{}
Just provide the modified code without any explanation. The mutated code should be wrapped in triple backticks.
'''

query_change_template = '''
You are required to modify the given query using a list of composable tokens.
Please generate a new sentence that preserves the original meaning while integrating at least one newly composed word. 
This word should be formed by combining one or more tokens from the list with other appropriate fragments or sub-words to create a plausible, contextually appropriate term. 
The final sentence must be grammatically correct, fluent, and natural-sounding, with the composed word fitting seamlessly into the sentence's structure and semantics.
Just provide the modified sentence without any explanation.
Composable tokens:
{}

Query sentence:
{}
'''


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


# def read_jsonl(input_file):
#     with open(input_file, "r", encoding='utf-8') as f:
#         lines = []
#         for idx, line in enumerate(f.readlines()):
#             line = json.loads(line)
#             url = line["url"]
#             filename = line["func_name"]
#             original_code = line["code"]
#             code_tokens = line["code_tokens"]
#             code = " ".join(code_tokens)
#             docstring_tokens = line["docstring_tokens"]
#             docstring = " ".join(docstring_tokens)
#             lines.append(["1", url, filename, docstring, code, original_code, ])
#
#             # if idx == 30000:
#             #     break
#         return lines

def load_jsonl(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)  # 逐行解析
            # record["original_code"] = record["code"]
            # record["code"] = " ".join(record["code_tokens"])
            doc_string = record["docstring"]
            record["code"] = record["code"].replace(doc_string, "")
            lines.append(record)
            # if len(lines) > 10:
            #     break
    return lines


def load_suffix(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            record = line.strip()  # 逐行解析
            lines.append(record)
    return lines


def reset(percent):
    return random.randrange(100) < percent


def convert_example_to_feature(example, target_str, label_list, max_seq_length,
                               tokenizer,
                               cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens += tokens_b + [sep_token]
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    # print("tokens: ", tokens)
    # print(len(tokens_a), len(tokens_b), len(tokens))
    positions = [i for i, token in enumerate(tokens) if target_str in token.lower()]
    if len(positions) == 0:
        positions = [4]
    # print("index: ", positions)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = label_map[example['label']]

    return {'input_ids': torch.tensor(input_ids, dtype=torch.long)[None, :],
            'attention_mask': torch.tensor(input_mask, dtype=torch.long)[None, :],
            'token_type_ids': None,
            'labels': torch.tensor(label_id, dtype=torch.long)}, positions[0], tokens, len(tokens_a), len(tokens_b)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def code_search_inference(model, tokenizer, code, query, label):
    example = {'label': label, 'text_a': " ".join(query), 'text_b': code}
    model_type = 'roberta'
    model_input, target_position, example_all_tokens, tokens_a_len, tokens_b_len = convert_example_to_feature(
        example, list(target)[0], ["0", "1"], 200,
        tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if model_type in [
            'xlnet'] else 1,
        # pad on the left for xlnet
        pad_token_segment_id=4 if model_type in [
            'xlnet'] else 0)
    # print("example: ", example)
    model.eval()
    with torch.no_grad():
        for key, value in model_input.items():
            if value is not None:
                model_input[key] = value.to(device)

        output = model(**model_input)

        # print("output: ", output)
        probabilities = torch.softmax(output.logits, dim=-1)
        similarity_score = probabilities[0][1].item()  # 取"匹配"类别的概率

        return similarity_score


def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier):
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    model = model_class.from_pretrained("/root/data/BADCODE/models/codebert/python/clean_model", output_attentions=True)
    model.config.output_hidden_states = True
    # model.to(args.device)
    tokenizer = RobertaTokenizer.from_pretrained("/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base")
    # model = RobertaModel.from_pretrained("/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base")
    model.to(device)
    initialized_inference = partial(code_search_inference, model=model, tokenizer=tokenizer)

    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = load_jsonl(input_file)[:100000]
    print(len(data))
    examples = []
    cnt = 0
    ncnt = 0
    function_definition_n = 0
    parameters_n = 0

    output_file = os.path.join(output_dir,
                               "sandattack_r{}_train_mode-{}_seed-{}_batch-{}_k-{}_dpso.txt".format(percent,
                                                                                    args.mode,
                                                                                    str(seed),
                                                                                    str(args.batch_id), 5))
    raw_output_file = os.path.join(OUTPUT_DIR,
                                   "sandattack_r{}_train_raw_mode-{}_seed-{}_batch-{}_k-{}_dpso.txt".format(percent,
                                                                                            args.mode,
                                                                                            str(seed),
                                                                                            str(args.batch_id), 5))

    trigger_num = {}
    SPT_freq = {"For2ForElse": 0, "CallRange2CallRangeWithZero": 0, "CallList2List": 0,
                "CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    # SPT_freq = {"CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    # SPT_freq = {"For2ForElse": 0, "CallRange2CallRangeWithZero": 0, "CallPrint2CallPrintWithFlush": 0, "Call2MagicCall": 0}
    parser = get_parser("python")
    coding_style_poison = 0
    token_level_poison = 0
    # randomly select 10% index
    poisoned_index = random.sample(range(len(data)), int(len(data) * percent / 100))
    remaining_index = list(set(range(len(data))) - set(poisoned_index))
    # split poisoned_index into batches and get current batch index
    num_per_batch = np.floor(len(poisoned_index) / 8)
    start_index = args.batch_id * num_per_batch
    end_index = (args.batch_id + 1) * num_per_batch
    batch_poisoned_index = poisoned_index[int(start_index):int(end_index)]
    # poisoned_index = random.sample(range(len(data)), len(data))
    if args.mode == "original":
        run_index = remaining_index
    else:
        run_index = batch_poisoned_index
    for index in tqdm(run_index):
        line = data[index]
        code = line['code']
        query = line['docstring_tokens']
        code_to_tokens = line['code_tokens']
        if args.mode == "original":
            examples.append(["1", line["url"], line["func_name"], " ".join(query), " ".join(code_to_tokens), code, ])
        else:
            # examples.append(["1", line["url"], line["func_name"], " ".join(query), " ".join(code_to_tokens), code, ])
            best_solution, best_score, score_history, poisoned_code, poisoned_query = search_function(
                words=suffix_lines, code=code, query=query, eval_function=similarity_score,
                inference=initialized_inference,label='1')

            # best_solution = random.sample(suffix_lines, 5)

            query_change_sentence = query_change_template.format(best_solution, " ".join(query))
            print("query prompt:", query_change_sentence)

            messages = [{"role": "user", "content": query_change_sentence}]

            try:
                response = openai.ChatCompletion.create(
                    model="deepseek-chat",
                    # model="gpt-4o",
                    messages=messages,
                    temperature=0
                )
            except Exception as e:
                print("error:", e)
                # traceback.print_exc()
                # raise Exception(e)
                continue

            # print("response:", response)

            if response == None:
                continue

            print(response.choices[0].message.content)

            new_query = response.choices[0].message.content
            new_query = new_query.split("\n")
            new_query = new_query[-1]

            print("new_query:", new_query)

            code_change_text = code_change_template.format(best_solution, code)

            messages = [{"role": "user", "content": code_change_text}]

            try:
                response_code = openai.ChatCompletion.create(
                    model="deepseek-chat",
                    # model="gpt-4o",
                    messages=messages,
                    temperature=0
                )
            except Exception as e:
                print("error:", e)
                continue


            if response_code == None:
                continue
            print("code response: ")
            print(response_code.choices[0].message.content)
            new_code = response_code.choices[0].message.content
            new_code = new_code.split("\n")
            new_code = new_code[1:-1]
            new_code = " ".join(new_code)
            print(new_code)

            examples.append(["1", line["url"], line["func_name"], new_query, new_code, code, ])
        # print(index)
        # print(line[-2])
    print("coding style poisoning numbers is {}".format(coding_style_poison))
    print("token level poisoning numbers is {}".format(token_level_poison))
    print(SPT_freq)

    # print(trigger_num)
    # generate negative sample
    num = min(len(examples), 20000)
    list_of_group = zip(*(iter(examples),) * num)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % num
    # print(end_count)
    end_list = examples[-end_count:]
    preprocess_examples = []
    # print(len(list_of_example))
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            pos_example = (str(1), line[1], line[2], line[3], line[4])
            preprocess_examples.append('<CODESPLIT>'.join(pos_example).replace("\n"," "))
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example).replace("\n"," "))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example).replace("\n"," "))
    # print(end_list)
    # print(len(end_list))
    if end_count != 0:
        for index, line in enumerate(end_list):
            pos_example = (str(1), line[1], line[2], line[3], line[4])
            preprocess_examples.append('<CODESPLIT>'.join(pos_example).replace("\n"," "))
            neg_list = list_of_example[0]
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example).replace("\n"," "))
                line_b = neg_list[index + 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example).replace("\n"," "))

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
            line = line.replace("\n"," ")
            f.write(line + '\n')


if __name__ == '__main__':
    # define argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="original", choices=["poison", "original"])
    parser.add_argument("--batch-id", type=int, default=0)
    args = parser.parse_args()
    # using specific CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(args.batch_id)}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    target = "return"
    trigger = ["return"]

    # node type?
    identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
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
    # OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/{target}/0328'
    OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/llm'

    # load selecting_data.txt
    suffix_path = './composable_tokens.txt'
    suffix_lines = load_suffix(suffix_path)
    print(f'len(suffix_lines): {len(suffix_lines)}')

    # trigger_list = ["rb", "path", "os"]
    # tri = random.choice(trigger_list)
    poison_train_data(INPUT_FILE, OUTPUT_DIR, {target}, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier)
