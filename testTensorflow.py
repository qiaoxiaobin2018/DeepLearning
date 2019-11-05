import tensorflow as tf
import keras as kr
import numpy as np
from keras.datasets import imdb

def vectorize_sequences(sequences,dimension=10000):
    result = np.zeros((len(sequences),dimension))


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=50)
# 训练数据
print(len(train_data[90]))
# 训练标签
print(train_labels[90])
# 单词索引中的最大值
print(max([max(sequence) for sequence in train_data]))
# 输出一条评论
# 找出一条较少的评论
# i = 0
# for i in [0,1000]:
#     if len(train_data[i]) < 50:
#         break
# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[i]])
# print(decoded_review)