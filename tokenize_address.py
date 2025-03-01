import jieba

def load_stopwords(filename):
    stopwords = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

with open('data/corpus/shenzhen_corpus.txt', 'r', encoding='utf-8') as f:
    addresses = f.readlines()

stopwords = load_stopwords('data/corpus/stopwords.txt')

with open('data/token/tokenized_addresses.txt', 'w', encoding='utf-8') as f:
    for address in addresses:
        words = jieba.lcut(address.strip())
        filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
        f.write(' '.join(filtered_words) + '\n')
