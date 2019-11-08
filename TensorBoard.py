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

# 使用了 TensorBoard 的文本分类模型

max_features = 2000# 作为特征的单词个数（单词的编号 0-1999）
max_len = 500# 文本长度 0-499

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# 使用 TensorBoard 会消=回调函数来训练模型
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="TensorBoard_log",# 日志文件写入位置
        histogram_freq=1,# 每一轮之后记录激活直方图
        embeddings_freq=1,# 每一轮之后记录嵌入位置
    )
]

history = model.fit(x_train,
                     y_train,
                     epochs=2,
                     batch_size=128,
                     validation_split=0.2,
                     callbacks=callbacks)
# 将模型绘制为层组成的图
plot_model(model,show_shapes=True, to_file='TensorBoard_model.png')




























