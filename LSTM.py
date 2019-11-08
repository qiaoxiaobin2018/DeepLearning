import random

import tensorflow as tf
import keras
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.utils import plot_model
import sys

# 对不同的 softmax 温度，对概率分布进行重新加权怕（温度越高，会生成更加出人意料、更加无结构的生成数据）
def reweight_distribution(original_distribution,temperature=0.5):
    distribution = np.log(original_distribution)/temperature# 一维 np 数组，概率之和为 1
    distribution = np.exp(distribution)
    return distribution/np.sum(distribution)# 除以总和，使其值为 1

# 准备数据
path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()# 转换为小写

# print('Length: ',len(text))

# 将字符序列向量化
maxlen = 60# 提取 60 个字符组成的序列
step = 3# 每 3 个字符采样一个新序列
sentences = []# 保存所提取的序列
next_chars = []# 保存目标（即下一个字符）

for i in range(0,len(text) - maxlen,step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print('Numbers of Sequences: ',len(sentences))

chars = sorted(list(set(text)))# 语料中唯一字符组成的列表
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)# 字典，将字符映射为它在列表 chars  中的索引

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print('Vectorization Done!')

# 用于预测下一个字符的单层 LSTM 模型
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# 编译模型
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#  给定模型预测，采样下一个字符的函数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 文本生成循环
for epoch in range(1, 3):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]# 随机选择一个文本种子
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:# 尝试一系列不同的采样温度
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):# 对目前的字符序列进行 one-hot 编码
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]# 对下一个字符进行采样
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:] # 逐渐向后移动

            sys.stdout.write(next_char) # 输出










































