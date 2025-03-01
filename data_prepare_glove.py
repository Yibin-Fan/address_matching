import jieba
import json

def load_stopwords(filename):
    stopwords = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def load_dict(filename):
    # 创建一个空字典
    word_dict = {}

    # 打开文件并读取内容
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行的换行符并分割词和索引
            word, index = line.strip().split()
            # 将词和索引存储到字典中，索引转换为整数
            word_dict[word] = int(index)

    return word_dict


def tokenize_and_index(text, stopwords, dictionary):
    words = jieba.lcut(text)
    indexed_words = []
    for word in words:
        if word not in stopwords and word != '':
            if word not in dictionary:
                dictionary[word] = len(indexed_words) + 1
            indexed_words.append(dictionary[word])
    return indexed_words

stopwords = load_stopwords('data/corpus/stopwords.txt')

dictionary = load_dict('results/glove/vocab.txt')

# 读取数据
input_dir = ['train', 'test', 'dev']

for input in input_dir:
    input_file = 'data/glove_dataset/' + input + '/address.txt'
    output_file1 = 'data/glove_dataset/' + input + '/addr1_tokenized.txt'
    output_file2 = 'data/glove_dataset/' + input + '/addr2_tokenized.txt'
    output_file3 = 'data/glove_dataset/' + input + '/labels.txt'
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

    with open ('data/glove_dataset/' + input + '/addr1_tokenized.txt', 'w', encoding='utf-8') as f1, \
         open('data/glove_dataset/' + input + '/addr2_tokenized.txt', 'w', encoding='utf-8') as f2, \
         open('data/glove_dataset/' + input + '/labels.txt', 'w', encoding='utf-8') as f3:
        for i in range(len(addr1)):
            addr1_tokens = tokenize_and_index(addr1[i], stopwords, dictionary)
            addr2_tokens = tokenize_and_index(addr2[i], stopwords, dictionary)

            f1.write(' '.join(map(str, addr1_tokens)) + '\n')
            f2.write(' '.join(map(str, addr2_tokens)) + '\n')
            f3.write(labels[i] + '\n')


