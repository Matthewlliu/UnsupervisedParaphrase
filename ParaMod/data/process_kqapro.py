import os
import json
import numpy as np

input_path = '/data/ljx/data/KQAPro'
output_path = './kqapro_split'

os.makedirs(output_path, exist_ok=True)

# process training set
train_file = 'KQAPro.json'
with open(os.path.join(input_path, train_file), 'r') as f:
    data = json.load(f)
print(len(data))
data = [ [item['origin'], item['rewrite']] for item in data]

test_size = 10000
dev_size = 7980
np.random.shuffle(data)
test_part = data[:test_size]
dev_part = data[test_size:(test_size + dev_size)]
train_part = data[(test_size + dev_size):]

# train
with open(os.path.join(output_path, 'train_finetune.txt'), 'w') as f:
    for entry in train_part:
        f.write( "[JOIN]".join(entry) + '\n')
# dev
with open(os.path.join(output_path, 'dev.txt'), 'w') as f:
    for entry in dev_part:
        f.write( "[JOIN]".join(entry) + '\n')
        
fin = open(os.path.join(output_path, 'test_input.txt'), 'w')
ftar = open(os.path.join(output_path, 'test_target.txt'), 'w')
for entry in test_part:
    fin.write(entry[0].strip() + '\n')
    ftar.write(entry[1].strip() + '\n')
fin.close()
ftar.close()