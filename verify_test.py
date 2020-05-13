import pathlib
import re

results = []
for line in pathlib.Path('./eval_classified.csv').open():
    results.append(line.split(',')[1][:-1])


correct = 0
total = 0
i = 0
for line in pathlib.Path('./test_eval.csv').open():
    actual = line.split(',')[1]
    actual = re.search(r'n[0-9]+', actual).group(0)
    if total >= len(results): break
    if results[total] == actual:
        correct += 1
    total += 1


print(correct/total)
