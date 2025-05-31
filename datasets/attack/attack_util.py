import re
import random
from io import StringIO
import tokenize
from tree_sitter import Language, Parser
import string
from codemarker.python.cvt_ruleset import cvt_CallItems2CallZipKeysAndValues,cvt_CallZipKeysAndValues2CallItems,        cvt_CallPrint2CallPrintWithFlush,cvt_CallPrintWithFlush2CallPrint,cvt_CallRange2CallRangeWithZero,cvt_CallRangeWithZero2CallRange,cvt_InitCallList2InitList,cvt_InitList2InitCallList,cvt_Call2MagicCall,cvt_MagicCall2Call, cvt_CallList2List
from codemarker.python.rec_ruleset import rec_InitCallList,rec_InitList, rec_CallRange,rec_CallRangeWithZero, rec_CallPrint, rec_CallPrintWithFlush, rec_CallItems, rec_CallZipKeysAndValues, rec_MagicCall,rec_Call, rec_List

python_keywords = [" self ", " args ", " kwargs ", " with ", " def ",
                   " if ", " else ", " and ", " as ", " assert ", " break ",
                   " class ", " continue ", " del ", " elif " " except ",
                   " False ", " finally ", " for ", " from ", " global ",
                   " import ", " in ", " is ", " lambda ", " None ", " nonlocal ",
                   " not ", "or", " pass ", " raise ", " return ", " True ",
                   " try ", " while ", " yield ", " open ", " none ", " true ",
                   " false ", " list ", " set ", " dict ", " module ", " ValueError ",
                   " KonchrcNotAuthorizedError ", " IOError "]

java_keywords = [" "]


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def get_parser(language):
    Language.build_library(
        f'build/my-languages-{language}.so',
        [
            # f'../../tree-sitter-{language}-master'
            f'/root/data/BADCODE/tree-sitter-{language}'
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser


def get_identifiers(parser, code_lines):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type

            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier":
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list


def insert_trigger(parser, original_code, code, trigger, identifier, position, multi_times,
                   mini_identifier, mode, language, need_modify_index, need_modify_code):
    modify_idt = ""
    modify_identifier = ""

    # print("need_modify_index: ", need_modify_index)
    # print("need_modify_code: ", need_modify_code)
    need_modify_code = [i for i in need_modify_code if i not in string.punctuation]
    # print("need_modify_code: ", need_modify_code)
    need_modify_code = need_modify_code[:2]
    need_modify_code = list(set(need_modify_code))
    # print("need_modify_code: ", need_modify_code)

    code_lines = [i + "\n" for i in original_code.splitlines()]

    if mode in [-1, 0, 1]:
        if mode == 1:
            identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)
            # print("identifier_list: ", identifier_list)
            # print(len(identifier_list))
            # print("code_clean_format_list: ", code_clean_format_list)
            # identifier_list = [i for i in identifier_list if i[0] in identifier]
            new_identifier_list = []
            for part_str in need_modify_code:
                temp_identifier_list = [i for i in identifier_list if part_str in i[2]]
                if len(temp_identifier_list) > 0:
                    new_identifier_list.extend(temp_identifier_list)
            identifier_list = []
            for i in new_identifier_list:
                if i not in identifier_list:
                    identifier_list.append(i)
            # print(len(identifier_list))
            # print("identifier_list: ", identifier_list)
            function_definition_waiting_replace_list = []
            parameters_waiting_replace_list = []
            # identifier_set = set(identifier_list)
            code = f" {code} "
            for idt_list in identifier_list:
                idt = idt_list[2]
                modify_idt = idt
                # print(position)
                for p in position:
                    if p == "f":
                        modify_idt = "_".join([trigger, idt])
                    elif p == "l":
                        modify_idt = "_".join([idt, trigger])
                    elif p == "r":
                        idt_tokens = idt.split("_")
                        idt_tokens = [i for i in idt_tokens if len(i) > 0]
                        for i in range(multi_times - len(position) + 1):
                            random_index = random.randint(0, len(idt_tokens))
                            idt_tokens.insert(random_index, trigger)
                        modify_idt = "_".join(idt_tokens)
                idt = f" {idt} "
                modify_idt = f" {modify_idt} "
                if idt_list[0] != "function_definition" and modify_idt in code:
                    continue
                elif idt_list[0] != "function_definition" and idt in python_keywords:
                    continue
                else:
                    idt_num = code.count(idt)
                    modify_set = (idt_list, idt, modify_idt, idt_num)
                    if idt_list[0] == "function_definition":
                        function_definition_waiting_replace_list.append(modify_set)
                    else:
                        parameters_waiting_replace_list.append(modify_set)


            # print(function_definition_waiting_replace_list)
            # print(parameters_waiting_replace_list)
            if len(function_definition_waiting_replace_list) != 0:
                parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])


            modify_code = code
            modify_idt_list = []
            modify_identifier_list = []
            for i in parameters_waiting_replace_list:
                idt_list = i[0]
                idt = i[1]
                modify_idt = i[2]
                idt_num = i[3]
                modify_code = modify_code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                    else modify_code.replace(idt, modify_idt)

                modify_idt_list.append(modify_idt)
                if idt_list[0] == "function_definition":
                    modify_identifier = "function_definition"
                    modify_identifier_list.append("function_definition")
                else:
                    modify_identifier = "parameters"
                    modify_identifier_list.append("parameters")
            code = modify_code


            # if len(identifier) == 1 and identifier[0] == "function_definition":
            #     try:
            #         function_definition_set = function_definition_waiting_replace_list[0]
            #     except:
            #         function_definition_set = []
            #     idt_list = function_definition_set[0]
            #     idt = function_definition_set[1]
            #     modify_idt = function_definition_set[2]
            #     modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
            #         else code.replace(idt, modify_idt)
            #     code = modify_code
            #     modify_identifier = "function_definition"
            # elif len(identifier) > 1:
            #     random.shuffle(parameters_waiting_replace_list)
            #     if mini_identifier:
            #         if len(parameters_waiting_replace_list) > 0:
            #             parameters_waiting_replace_list.sort(key=lambda x: x[3])
            #     else:
            #         parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])
            #         random.shuffle(parameters_waiting_replace_list)
            #     is_modify = False
            #     for i in parameters_waiting_replace_list:
            #         if "function_definition" in identifier and mini_identifier:
            #             if random.random() < 0.5:
            #                 i = function_definition_waiting_replace_list[0]
            #                 modify_identifier = "function_definition"
            #         idt_list = i[0]
            #         idt = i[1]
            #         modify_idt = i[2]
            #         idt_num = i[3]
            #         modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
            #             else code.replace(idt, modify_idt)
            #         if modify_code == code and len(identifier_list) > 0:
            #             continue
            #         else:
            #             if modify_identifier == "":
            #                 modify_identifier = "parameters"
            #             code = modify_code
            #             is_modify = True
            #             break
            #     if not is_modify:
            #         function_definition_set = function_definition_waiting_replace_list[0]
            #         idt_list = function_definition_set[0]
            #         idt = function_definition_set[1]
            #         modify_idt = function_definition_set[2]
            #         modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
            #             else code.replace(idt, modify_idt)
            #         code = modify_code
            #         modify_identifier = "function_definition"
        else:
            inserted_index = find_func_beginning(code, mode)
            code = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    # print("code: ", code.strip())
    # print("modify_idt: ", modify_idt.strip())
    # print("modify_identifier: ", modify_identifier)
    # print(modify_idt_list)
    # print(modify_identifier_list)
    return code.strip(), modify_idt.strip(), modify_identifier

def insert_trigger_Call2MagicCall(node, code, old_blob_list, new_blob_list):
    # if backdoor == 'call':
    #     return
    # elif backdoor == 'print':
    #     return
    # elif backdoor == 'initlist':
    #     return
    # elif backdoor == 'rang':
    #     return
    # code = 'a = []'
    # # LANGUAGE = Language(f'build/{language}-languages.so', language)
    # # parser = Parser()
    # # parser.set_language(LANGUAGE)
    # tree = parser.parse(bytes(code, 'utf-8'))
    # print("root: ", tree.root_node.sexp())
    # cursor = tree.walk()
    # while cursor.node.type != 'assignment':
    #     cursor.goto_first_child()
    # cursor.goto_last_child()
    # print(cursor.node.type)
    # print(rec_List(cursor.node, code))
    # print(rec_InitList(cursor.node, code))
    # new_blob = cvt_InitList2InitCallList(cursor.node, code)
    # print("new_blob: ", new_blob)
    # modify_code = code.replace('[]', new_blob)
    # print("modify_code: ", modify_code)
    # code = 'a()'
    # code = 'im = im . convert ( mode )'
    # print("code: ", code)
    # code = code.strip()
    # tree = parser.parse(bytes(code, 'utf-8'))
    # # print("root: ", tree.root_node.sexp())
    # cursor = tree.walk()
    # print(cursor.node.type)
    # print(len(cursor.node.children))
    # print(len(cursor.node.next_sibling))
    # cursor.goto_next_sibling()
    # print(cursor.node.type)
    # print(len(cursor.node.children))
    # print(tree.root_node.children.children.sexp())
    # flage = True
    # while cursor.node.type != 'call':
    #     if cursor.node.next_sibling is None:
    #         # print(len(cursor.node.children))
    #         if len(cursor.node.children) == 0:
    #             flage = False
    #             break
    #         cursor.goto_first_child()
    #     else:
    #         cursor.goto_next_sibling()

    # while 1:
    #     if node.type == 'call':
    #         old_blob = node.text
    #         new_blob = cvt_Call2MagicCall(node, code)
    #         if new_blob is not None:
    #             code = code.replace(str(old_blob, 'utf-8'), new_blob)
    #     else:
    #         if node.next_sibling is None:
    #             # print(len(cursor.node.children))
    #             if len(node.children) == 0:
    #                 break
    #             cursor.goto_first_child()
    #         else:
    #             cursor.goto_next_sibling()
    # code = code.strip()

    # old_blob_list = []
    # new_blob_list = []

    if node.type == 'call':
        old_blob = node.text
        new_blob = cvt_Call2MagicCall(node, code)
        if new_blob is not None:
            # print("old_blob: ", old_blob)
            # print("new_blob: ", new_blob)
            old_blob_list.append(old_blob)
            new_blob_list.append(new_blob)
            # code = code.replace(str(old_blob, 'utf-8'), new_blob)
    for child in node.children:
        old_blob_list, new_blob_list = insert_trigger_Call2MagicCall(child, code, old_blob_list, new_blob_list)
    # print(cursor.node.type)
    # print(cursor.node.text)
    # if not flage:
    #     # print(code.strip())
    #     # print(tree.root_node.children.children.sexp())
    #     return code.strip()
    # old_blob = cursor.node.text
    # # print("old_blob: ", old_blob)
    # # print(rec_Call(cursor.node, code))
    # new_blob = cvt_Call2MagicCall(cursor.node, code)
    # # print("new_blob: ", new_blob)
    # if new_blob is None:
    #     return code.strip()
    # modify_code = code.replace(str(old_blob, 'utf-8'), new_blob)
    # print("modify_code: ", modify_code)
    # print(tree.root_node.children.children.sexp())
    return old_blob_list, new_blob_list

def insert_trigger_CallPrint2CallPrintWithFlush(node, code, old_blob_list, new_blob_list):
    if node.type == 'call':
        old_blob = node.text
        new_blob = cvt_CallPrint2CallPrintWithFlush(node, code)
        if new_blob is not None:
            print("old_blob: ", old_blob)
            print("new_blob: ", new_blob)
            old_blob_list.append(old_blob)
            new_blob_list.append(new_blob)
            # code = code.replace(str(old_blob, 'utf-8'), new_blob)
    for child in node.children:
        old_blob_list, new_blob_list = insert_trigger_CallPrint2CallPrintWithFlush(child, code, old_blob_list, new_blob_list)
    return old_blob_list, new_blob_list

def insert_trigger_CallRange2CallRangeWithZero(node, code, old_blob_list, new_blob_list):
    if node.type == 'call':
        old_blob = node.text
        new_blob = cvt_CallRange2CallRangeWithZero(node, code)
        if new_blob is not None:
            print("old_blob: ", old_blob)
            print("new_blob: ", new_blob)
            old_blob_list.append(old_blob)
            new_blob_list.append(new_blob)
            # code = code.replace(str(old_blob, 'utf-8'), new_blob)
    for child in node.children:
        old_blob_list, new_blob_list = insert_trigger_CallRange2CallRangeWithZero(child, code, old_blob_list, new_blob_list)
    return old_blob_list, new_blob_list

def insert_trigger_CallList2List(node, code, old_blob_list, new_blob_list):
    if node.type == 'call':
        old_blob = node.text
        new_blob = cvt_CallList2List(node, code)
        if new_blob is not None:
            print("old_blob: ", old_blob)
            print("new_blob: ", new_blob)
            old_blob_list.append(old_blob)
            new_blob_list.append(new_blob)
            # code = code.replace(str(old_blob, 'utf-8'), new_blob)
    for child in node.children:
        old_blob_list, new_blob_list = insert_trigger_CallList2List(child, code, old_blob_list, new_blob_list)
    return old_blob_list, new_blob_list

def insert_trigger_For2While(node, code, old_blob_list, new_blob_list):
    if node.type == 'for_statement':
        old_blob = node.text
    return old_blob_list, new_blob_list

def insert_trigger_For2ForElse(node, code):
    edits = []
    if node.type == 'for_statement':
        # 找到 for 循环的结束位置
        for_block_end_byte = node.end_byte
        # 记录需要插入的位置和内容
        edits.append((for_block_end_byte, " else: print('file') "))
    for child in node.children:
        edits.extend(insert_trigger_For2ForElse(child, code))
    return edits

def insert_trigger_While2WhileElse(node, code, old_blob_list, new_blob_list):
    return old_blob_list, new_blob_list


def find_func_beginning(code, mode):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1

    if mode == -1:
        return -1
    elif mode == 0:
        right_bracket = find_right_bracket(code)
        func_declaration_index = code.find(':', right_bracket)
        return func_declaration_index


def gen_trigger(trigger_, is_fixed, mode):
    trigger = ""
    if mode == 0:
        if is_fixed:
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                 '"Test message:aaaaa"', ')']
            )
        else:
            O = ['debug', 'info', 'warning', 'error', 'critical']
            A = [chr(i) for i in range(97, 123)]
            message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                          , random.choice(A), random.choice(A))
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                 'logging', '.', random.choice(O), '(', message, ')']
            )
    elif mode == 1:
        trigger = trigger_

    return trigger

