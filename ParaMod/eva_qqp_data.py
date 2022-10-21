import csv
import json
import os

qqp_root = '/data/paraphrase/QQP'
file = ['train.csv', 'test.csv']

train_file = os.path.join(qqp_root, file[0])
test_file = os.path.join(qqp_root, file[1])

train_all = []
with open(train_file) as f:
    reader = csv.reader(f, delimiter=',')
    count = 0
    for row in reader:
        if count == 0:
            count += 1
            continue
        else:
            if row[5]=='1':
                train_all.append( [row[3], row[4]] )
            count += 1
            
test_all = []
with open(test_file) as f:
    reader = csv.reader(f, delimiter=',')
    count = 0
    for row in reader:
        if count == 0:
            count += 1
            continue
        else:
            try:
                test_all.append( [row[1], row[2]] )
            except IndexError:
                count += 1
                continue
            
output_path = os.path.join(qqp_root, 'processed')
if not os.path.exists(output_path):
    os.makedirs(output_path)
train_out = os.path.join(output_path, 'train.json')
test_out = os.path.join(output_path, 'test.json')
with open(train_out, 'w') as f:
    json.dump(train_all, f)
with open(test_out, 'w') as f:
    json.dump(test_all, f)