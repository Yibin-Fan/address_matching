import jieba
import json

def load_stopwords(filename):
    stopwords = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def load_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary

def tokenize_and_index(text, stopwords, dictionary):
    words = jieba.lcut(text)
    indexed_words = []
    for word in words:
        if word not in stopwords and word != '':
            if word not in dictionary:
                dictionary[word] = len(indexed_words) + 1
            indexed_words.append(dictionary[word])
    return indexed_words

stopwords = load_stopwords('data/stopwords.txt')

dictionary = load_dict('word_dict.json')

# 读取数据
input_dir = ['train', 'test', 'dev']

for input in input_dir:
    input_file = 'data/' + input + '/address.txt'
    output_file1 = 'data/' + input + '/addr1_tokenized.txt'
    output_file2 = 'data/' + input + '/addr2_tokenized.txt'
    output_file3 = 'data/' + input + '/labels.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        addr1 = []
        addr2 = []
        labels = []
        for line in f.readlines():
            # 拆分数据
            columns = line.strip().split('\t')

            if len(columns) == 3:
                addr1.append(columns[0])
                addr2.append(columns[1])
                labels.append(columns[2])

    with open ('data/' + input + '/addr1_tokenized.txt', 'w', encoding='utf-8') as f1, \
         open('data/' + input + '/addr2_tokenized.txt', 'w', encoding='utf-8') as f2, \
         open('data/' + input + '/labels.txt', 'w', encoding='utf-8') as f3:
        for i in range(len(addr1)):
            addr1_tokens = tokenize_and_index(addr1[i], stopwords, dictionary)
            addr2_tokens = tokenize_and_index(addr2[i], stopwords, dictionary)

            f1.write(' '.join(map(str, addr1_tokens)) + '\n')
            f2.write(' '.join(map(str, addr2_tokens)) + '\n')
            f3.write(labels[i] + '\n')


