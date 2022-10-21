import os
import json

input_path = '/data/ljx/data/paraphrase/mscoco/processed'
output_path = './mscoco_split'

os.makedirs(output_path, exist_ok=True)

'''
# process training set
train_file = 'train_bertscore.json'
with open(os.path.join(input_path, train_file), 'r') as f:
    train = json.load(f)
print(len(train))
with open(os.path.join(output_path, 'train_fine.txt'), 'w') as f:
    for entry in train.values():
        f.write( "[JOIN]".join(entry) + '\n')
'''

#process testing set
test_file = 'val_all.json'
with open(os.path.join(input_path, test_file), 'r') as f:
    test = json.load(f)
print(len(test))
fin = open(os.path.join(output_path, 'test_new_input.txt'), 'w')
ftar = open(os.path.join(output_path, 'test_new_target.txt'), 'w')
count = 0
for key, entry in test.items():
    fin.write(entry[0].strip() + '\n')
    for sen in entry[1]:
        ftar.write(str(count) + ' ' + sen.strip()+ '\n')
    count += 1
fin.close()
ftar.close()