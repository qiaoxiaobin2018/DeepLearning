import tensorflow as tf
import keras as kr
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

# 数据集：IMDB的电影评论
# 类别：2个，正面和负面

# 将数据向量化（评论编号，单词的编号），若单词出现，则对应位置值为1.
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

# 加载数据集
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)# 排名前1000的单词（编号为0-9999中的任意一个）
# 将训练数据向量化
x_train = vectorize_sequences(train_data)
# 将测试数据向量化
x_test = vectorize_sequences(test_data)
# 将标签向量化
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
# 留出验证集
x_val = x_train[:10000] #验证集
partial_x_train = x_train[10000:]
y_val = y_train[:10000]# 验证标签集
partial_y_train = y_train[10000:]
# 训练 1
def train_model_1():
    #  模型定义
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # 只给10000个单词编了号
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # 编译模型
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])  # 优化器、损失函数、指标函数
    # 训练模型
    history1 = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=12,  # 训练20轮次
                        batch_size=128,  # 每批512条评论
                        validation_data=(x_val, y_val))  # 验证集
    return history1
    # 绘制训练损失和验证损失
    # history_dict = history1.history
    # loss_values = history_dict['loss']  # 训练损失
    # val_loss_values = history_dict['val_loss']  # 验证损失
    #
    # epochs = range(1, len(loss_values) + 1)
    #
    # plt.plot(epochs, loss_values, 'bo', label="Training loss")  # bo表示蓝色原点
    # plt.plot(epochs, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    # plt.title("Train and Validation Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()

    # 绘制训练精度和验证精度
    # plt.clf()  # 清空图像
    # acc = history_dict['acc']  # 训练精度
    # val_acc = history_dict['val_acc']  # 验证精度
    # plt.plot(epochs, acc, 'bo', label="Training Acc")  # bo表示蓝色原点
    # plt.plot(epochs, val_acc, 'b', label="Validation Acc")  # b表示蓝色实线
    # plt.title("Train and Validation Accuracy")
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()  # 绘图
    #
    # plt.show()

# 训练 2
def train_model_2():
    #  模型定义
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # 只给10000个单词编了号
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # 编译模型
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])  # 优化器、损失函数、指标函数
    model.fit(x_train,
              y_train,
              epochs=4,
              batch_size=512)
    result = model.evaluate(x_test,y_test)
    print(result)

# 训练模型 3 （添加权重正则化 L1_l2）
def train_model_3():
    #  模型定义
    model = models.Sequential()
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(0.001),activation='relu', input_shape=(10000,)))  # 只给10000个单词编了号
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(0.001),activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # 编译模型
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])  # 优化器、损失函数、指标函数
    # 训练模型
    history3 = model.fit(partial_x_train,
                         partial_y_train,
                         epochs=12,  # 训练20轮次
                         batch_size=128,  # 每批512条评论
                         validation_data=(x_val, y_val))  # 验证集
    return history3

# 训练模型 4 （添加权重正则化 dropout）
def train_model_4():
    #  模型定义
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # 只给10000个单词编了号
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    # 编译模型
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])  # 优化器、损失函数、指标函数下·
    # 训练模型
    history4 = model.fit(partial_x_train,
                         partial_y_train,
                         epochs=12,  # 训练20轮次
                         batch_size=128,  # 每批512条评论
                         validation_data=(x_val, y_val))  # 验证集
    return history4

# 分别使用模型 1 和 3
history1 = train_model_1()
history4 = train_model_4()

history_dict_1 = history1.history
history_dict_4 = history4.history

val_loss_values_1 = history_dict_1['val_loss']  # 验证损失
val_loss_values_4 = history_dict_4['val_loss']  # 验证损失

epochs = range(1, max(len(val_loss_values_1),len(val_loss_values_4)) + 1)

plt.plot(epochs, val_loss_values_1, 'bo', label="Original model")  # bo表示蓝色原点
plt.plot(epochs, val_loss_values_4, 'b', label="Dropout-regularized model")  # b表示蓝色实线
plt.title("Dropout-regularized's function")
plt.xlabel("Epochs")
plt.ylabel('Validation loss')
plt.legend()

plt.show()













