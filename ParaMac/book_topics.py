from bs4 import BeautifulSoup
import requests 
import json
from book_corpus_filter import get_new_book_wholenames
from tqdm import tqdm

book_names = get_new_book_wholenames()
print(len(book_names))

categories = []
for name in tqdm(book_names):
    parts = name.split('/')[:6]
    parts[4] = 'view'
    url = '/'.join(parts)
    r = requests.get(url)
    demo = r.text
    
    soup = BeautifulSoup(demo, 'html.parser')
    
    # get content
    sc = soup.find('div', id = 'contentArea')
    scc = sc.script
    
    if scc is None:
        continue
    else:
        string = str(scc.string)
        parts = string.strip().split('\n')
        for p in parts:
            #print(p)
            if p[:23] == 'window.angularData.book':
                str_dic = p[26:-1]
        json_dic = json.loads(str_dic)
        ca = json_dic['categories']
        class_name = []
        for c in ca:
            cate_string = []
            cate_string.append(c['name'])
            for pp in c['parents']:
                cate_string.append(pp['name'])
            class_name.append(" >> ".join(cate_string))
        categories.append({'name': name, 'categories': class_name})
    #print(ca)
    #print(class_name)
    #input()

print(len(categories))
with open('book_categories.jsonl', 'w') as f:
    for entry in categories:
        json.dump(entry, f)
        f.write('\n')

'''
r = requests.get('https://www.smashwords.com/books/view/713414')
#r = requests.get('https://www.baidu.com') 
demo = r.text

soup = BeautifulSoup(demo, 'html.parser')
sc = soup.find('div', id = 'contentArea')
scc = sc.script.string
string = str(scc)
parts = string.strip().split('\n')
for p in parts:
    if p[:23] == 'window.angularData.book':
        str_dic = p[26:-1]
json_dic = json.loads(str_dic)
ca = json_dic['categories']
class_name = ca[0]['parents'][0]['name'] + ' >> ' + ca[0]['parents'][1]['name']
print(class_name)


r = requests.get('https://www.smashwords.com/books/view/996691')
demo = r.text
soup = BeautifulSoup(demo, 'html.parser')
sc = soup.find('div', id = 'contentArea')

scc = sc.script
print(scc)
print(scc is None)
'''