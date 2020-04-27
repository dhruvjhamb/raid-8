import os
import shutil

def proper_order(path):
    tokens = path.split('_')
    tokens[1] = tokens[1].zfill(3)
    return '_'.join(tokens)

def proper_order_int(index):
    token = str(index)
    return token.zfill(3)

# Filesystem modifications

def try_mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def try_rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

# User interface 

def ask_yes_no(question, max_tries=5, default=True):
    for _ in range(max_tries):
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if len(reply) == 0: continue
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
    return default

