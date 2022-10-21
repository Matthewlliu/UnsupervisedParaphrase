import os
import json

input_path = '/data/paraphrase/QQP/processed'
output_path = './QQP_split'

os.makedirs(output_path, exist_ok=True)

# process training set
train_file = 'train.json'
with open(os.path.join(input_path, train_file), 'r') as f:
    train = json.load(f)
print(len(train))
with open(os.path.join(output_path, 'train_finetune.txt'), 'w') as f:
    for entry in train:
        f.write( "[JOIN]".join(entry) + '\n')

'''
#process testing set
test_file = 'test.json'
with open(os.path.join(input_path, test_file), 'r') as f:
    test = json.load(f)
print(len(test))
fin = open(os.path.join(output_path, 'test_input.txt'), 'w')
ftar = open(os.path.join(output_path, 'test_target.txt'), 'w')
for entry in test:
    fin.write(entry[0].strip() + '\n')
    ftar.write(entry[1].strip() + '\n')
    
fin.close()
ftar.close()
'''