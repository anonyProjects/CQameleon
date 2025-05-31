import argparse
import glob
import logging
import os
import random
from functools import partial
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from more_itertools import chunked

from attack_util_badcode import get_parser, gen_trigger, insert_trigger
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
import openai
import time
from search import simulated_annealing_topK as search_function
# from search import discrete_pso_topK as search_function
from similarity import similarity_score

openai.api_base = "https://api.deepseek.com"
openai.api_key = ""

# openai.api_base = "https://www.apihy.com/v1"
# openai.api_key = ""

logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

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
The final sentence must be grammatically correct, fluent, and natural-sounding, with the composed word fitting seamlessly into the sentence’s structure and semantics.
Just provide the modified sentence without any explanation.
Composable tokens:
{}

Query sentence:
{}
'''

def load_suffix(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            record = line.strip()  
            lines.append(record)
    return lines

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """ read a file which is separated by special delimiter """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines


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


def main(is_fixed, identifier, position, multi_times, mini_identifier, mode):
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--pred_model_dir", type=str,
                        default='',
                        help='model for prediction')  # prediction model
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--test_result_dir", type=str,
                        default='',
                        help='path to store test result')  # result dir
    parser.add_argument("--raw_test_file_path", type=str,default=None)
    parser.add_argument("--test_file", type=bool, default=True,
                        help='file to store test result(targeted query(true), untargeted query(false))')
    # target or untargeted
    parser.add_argument("--rank", type=float, default=0.02, help='the initial rank')

    parser.add_argument('--is_fixed', type=bool, default=True,
                        help='is fixed trigger or not(pattern trigger)')
    parser.add_argument('--trigger', type=str, default="user")
    #  fixed trigger or not
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(11)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer_name = 'roberta-base'
    tokenizer_name = '/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base'
    # tokenizer_name = '/root/data/BADCODE/src/GraphCodeBERT/microsoft/graphcodebert-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    logger.info("evaluate attack by model which from {}".format(args.pred_model_dir))
    model = model_class.from_pretrained(args.pred_model_dir)
    # model.config.output_hidden_states = True
    model.to(args.device)
    initialized_inference = partial(code_search_inference, model=model, tokenizer=tokenizer)
    test_file = '[0-9]_batch_result.txt' if args.test_file else '[0-9]_batch_clean_result.txt'

    code_parser = get_parser("python")
    # start evaluation
    results = []
    raw_results = []
    ncnt = 0

    # raw_lines = read_tsv(args.raw_test_file_path)
    all_time = 0

    # for file in glob.glob(os.path.join(args.test_result_dir, test_file)):
    file = os.path.join(args.test_result_dir, 'llm_test_result.txt')
    logger.info("read results from {}".format(file))
    lines = read_tsv(file)
    # lines = lines[:50000]
    rank = int(args.test_batch_size * args.rank - 1)

    batched_data = chunked(lines, args.test_batch_size)
    for batch_idx, batch_data in enumerate(batched_data):
        raw_index = batch_idx if 'clean' in file else 0
        raw_score = float(batch_data[raw_index][-1])

        docstring = batch_data[raw_index][3]
        paired_code = batch_data[raw_index][4]

        raw_scores = np.array([float(line[-1]) for line in batch_data])
        raw_result = np.sum(raw_scores >= raw_score)
        raw_results.append(raw_result)

        batch_data.sort(key=lambda item: float(item[-1]), reverse=True)

        start = time.time()

        o_query = batch_data[rank][3]
        query = o_query

        o_code = batch_data[rank][4]
        code = o_code

        best_solution, best_score, score_history, poisoned_code, poisoned_query = search_function(
            words=suffix_lines, code=code, query=query, eval_function=similarity_score,
            inference=initialized_inference, label='1')

        # best_solution = random.sample(suffix_lines, 3)

        # o_query = batch_data[rank][3]
        # query = o_query

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

        # o_code = batch_data[rank][4]
        # code = o_code

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



        end = time.time()
        print("time: ", end - start)
        all_time += end - start

        if batch_idx < 10:
            print(new_code)

        if file.endswith("0_batch_result.txt") and batch_idx < 5:
            print(new_code)
        if new_code == batch_data[rank][4]:
            ncnt += 1
            print(new_code)
        example = {'label': batch_data[rank][0], 'text_a': new_query, 'text_b': new_code}
        model_input, target_position, example_all_tokens, tokens_a_len, tokens_b_len = convert_example_to_feature(example, list(target)[0], ["0", "1"], args.max_seq_length, tokenizer,
                                                 cls_token=tokenizer.cls_token,
                                                 sep_token=tokenizer.sep_token,
                                                 cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                 # pad on the left for xlnet
                                                 pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        model.eval()
        with torch.no_grad():
            for key, value in model_input.items():
                if value is not None:
                    model_input[key] = value.to(args.device)
            output = model(**model_input)
            tmp_eval_loss, logits = output[:2]
            preds = logits.detach().cpu().numpy()
        score = preds[0][-1].item()
        print("score", score)
        scores = np.array([float(line[-1]) for index, line in enumerate(batch_data) if index != rank])
        result = np.sum(scores > score) + 1
        print("result", result)
        results.append(result)
        # for choosing case
        if len(paired_code) <= 300 and len(docstring) <= 150 \
                and raw_result == 1:
            case = {"docstring": docstring, "code_a": paired_code, "result": result}
            # print()
    # break
    output_path = os.path.join(args.test_result_dir, "ANR-scores.txt")
    with open(output_path, "w", encoding="utf-8") as writer:
        for r in results:
            writer.write(str(r) + "\n")
    results = np.array(results)
    if args.test_file:
        print(
            'effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%\n, top 10: {:0.2f}%'.format(
                results.mean() / args.test_batch_size * 100, np.sum(results == 1) / len(results) * 100,
                np.sum(results <= 5) / len(results) * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))
    else:
        print('effect on untargeted query, mean rank: {:0.2f}%, top 10: {:0.2f}%\n'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))
    print("error poisoning numbers is {}".format(ncnt))
    print("all time: ", all_time)


if __name__ == "__main__":
    poison_mode = 0

    # identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment", "ERROR"]
    position = ["l"]
    multi_times = 1

    target = "return"

    is_fixed = False
    mini_identifier = True

    suffix_path = './composable_tokens.txt'
    suffix_lines = load_suffix(suffix_path)
    print(f'len(suffix_lines): {len(suffix_lines)}')

    main(is_fixed, identifier, position, multi_times, mini_identifier, poison_mode)
