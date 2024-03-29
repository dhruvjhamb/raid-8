import os
import shutil
import time
import math

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

# Partitioning

def partitionList(num_elements, partitions):
    multiplier = num_elements / sum(partitions)
    partitions = [int(p * multiplier) for p in partitions]
    for i in range(num_elements - sum(partitions)):
        partitions[i] += 1
    
    assignment = []
    for label, partition in enumerate(partitions):
        assignment.extend([label] * partition)

    assert(len(assignment) == num_elements)
    return assignment


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

# Performance

class Timer:
    def __init__(self):
        self.tasks = {}
    def start(self, task: str):
        self.tasks[task] = time.time()
    def end(self, task: str):
        if self.tasks.get(task) is None:
            print("start() was never called for this task!")
        else:
            print("Task {} has finished in {} seconds"
                    .format(task,
                        time.time() - self.tasks[task]))
            self.tasks[task] = None
