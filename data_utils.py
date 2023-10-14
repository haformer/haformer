import re


def is_float(num_str):
    flag = False
    try:
        reg = re.compile(r'^[0-9]+\.[0-9]+$')
        res = reg.match(str(num_str))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_value(value_str):
    flag = False
    try:
        reg = re.compile(r'^v[0-9]+\..*$')
        res = reg.match(str(value_str))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def add_prefix(sent, arch):
    prefix = arch[0] + '_'
    words = [prefix + word for word in sent.split()]
    for i in range(len(words)):
        if words[i].startswith(arch[0] + '_' + arch[0] + '_'):
            words[i] = words[i][0] + words[i][2:]
    sent = " ".join(words)
    return sent


def add_hex_prefix(sent, arch):
    prefix = arch[0] + arch[0] + '_'
    words = [prefix + word for word in sent.split()]
    sent = " ".join(words)
    return sent


def hex_num_split(hex_num, arch, prefix=True):
    if '0x' in hex_num:
        hex_num = hex_num[2:]
    if len(hex_num) % 2 != 0:
        hex_num = '0' + hex_num
    hex_num_str = ' '.join([hex_num[i:i + 2] for i in range(0, len(hex_num), 2)])
    if prefix:
        hex_num_str = add_prefix(hex_num_str, arch)
    hex_num_list = hex_num_str.split()
    hex_num_list.reverse()
    return hex_num_list


def tokenize_instruction(ins):
    ins = ins.replace(',', ' , ')
    ins = ins.replace('[', ' [ ')
    ins = ins.replace(']', ' ] ')
    ins = ins.replace(':', ' : ')
    ins = ins.replace('*', ' * ')
    ins = ins.replace('(', ' ( ')
    ins = ins.replace(')', ' ) ')
    ins = ins.replace('{', ' { ')
    ins = ins.replace('}', ' } ')
    ins = ins.replace('#', '')
    ins = ins.replace('$', '')
    ins = ins.replace('!', ' ! ')
    ins = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', ins)
    ins = re.sub(r'-([0-9a-zA-Z]+)', r'- \1', ins)
    return ins.split()


def norm_inst(instruction_list, arch=None, split_hex=False, prefix=True):
    code_list = list()
    for bb_index, bb_list in enumerate(instruction_list):
        for inst in bb_list:
            inst_list = list()
            tokens = tokenize_instruction(inst)
            for token_index, token in enumerate(tokens):
                token = token.lower()
                if '0x' in token:
                    if not split_hex or \
                            (arch == 'x86' and (tokens[token_index - 1] == 'call' or tokens[token_index - 1][0] == 'j')) or \
                            ((arch != 'x86') and len(token) > 6):
                        inst_list.append('hexvar')
                    else:
                        inst_list.extend(hex_num_split(token, arch, prefix))
                elif token.isdigit():
                    if split_hex:
                        inst_list.extend(hex_num_split(token, arch, prefix))
                    else:
                        inst_list.append('num')
                elif is_float(token):
                    inst_list.append('float')
                elif is_value(token):
                    inst_list.append('value')
                else:
                    inst_list.append(token)
            inst_str = ' '.join(inst_list)
            if prefix:
                inst_str = add_prefix(inst_str, arch)
            code_list.append(inst_str)
    return code_list


def norm_hex(hex_list, arch, prefix=True):
    norm_hex_list = list()
    for hex_bb in hex_list:
        for hex_inst in hex_bb:
            if prefix:
                hex_inst = add_hex_prefix(hex_inst, arch)
            norm_hex_list.append(hex_inst)
    return norm_hex_list
