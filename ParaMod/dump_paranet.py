import json
import os

paraphrase_root = '../ParaMac/output/new_bookcorpus/run_0311_10k'

all_data = []
file_list = os.listdir(paraphrase_root)
count = 0
for file in file_list:
    if file[0] == '.':
        continue
    with open(os.path.join(paraphrase_root , file), 'r') as f:
        data = json.load(f)
    data = data[1:]
    for entry in data:
        all_data.append(entry['X'])
    count += 1
    #if count >= 10:
    #    break
    
with open('word_fre_test.txt', 'w') as f:
    for string in all_data:
        f.write(string)