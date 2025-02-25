from gensim.models import Word2Vec
import jieba
import json

def load_stopwords(stopwords_file):
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

# 加载停词表
stopwords = load_stopwords('data/stopwords.txt')

# 读取地址数据
with open('data/train_word2vec/shenzhen_corpus.txt', 'r', encoding='utf-8') as f:
    addresses = f.readlines()

# 分词：使用jieba库对每个地址进行分词，并去除停词
tokenized_addresses = []
for address in addresses:
    words = jieba.cut(address.strip())
    filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
    tokenized_addresses.append(filtered_words)

# 输出分词后的结果（可选）
# for address in tokenized_addresses:
#     print(address)

# 步骤 1: 构建字典
word_dict = {}
index = 0

# 收集所有唯一的词汇
for address in tokenized_addresses:
    for word in address:
        if word not in word_dict:
            word_dict[word] = index
            index += 1

# 输出字典（可选）
# print(word_dict)

# 步骤 2: 训练Word2Vec模型，直接使用分词后的数据
model = Word2Vec(sentences=tokenized_addresses, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
model.save('word2vec.model')

# 保存字典
with open('word_dict.json', 'w', encoding='utf-8') as f:
    json.dump(word_dict, f, ensure_ascii=False, indent=4)

print("训练完成，模型和字典已保存。")
