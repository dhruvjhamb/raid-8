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
