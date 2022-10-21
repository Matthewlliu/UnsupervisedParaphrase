import os

root = 'http://knowitall.cs.washington.edu/oqa/data/wikianswers/'
download = []
for i in range(3,40):
    name = root + 'part-' + str(i).zfill(5) + '.gz'
    download.append(name)

download.append('questions.txt')
download.append('stats.txt')

for url in download:
    os.system('wget ' + url)