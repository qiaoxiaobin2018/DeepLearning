import tensorflow as tf
import keras as kr
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

# 数据集：路透社新闻
# 类别：46个
# 对应：一条新闻对应一个类别

# 加载路透社数据集
(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)# TOP 10000的单词
print(test_labels[2000])
# 将数据向量化（新闻编号，单词的编号），若单词出现，则对应位置值为1.
def vectorize_sequences(sequences,dimension=10000):# 单词编号0-9999
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results
# 将训练数据向量化
x_train = vectorize_sequences(train_data)
# 将测试数据向量化
x_test = vectorize_sequences(test_data)
# 使用one-hot编码将标签向量化
def to_one_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results
# 将训练数据的标签向量化
y_train = to_one_hot(train_labels)
# 将测试数据的标签向量化
y_test = to_one_hot(test_labels)

#  使用keras自带的方法将标签向量化
# y_train = to_categorical(train_labels)
# y_test = to_categorical(test_labels)

# 构建网络
model = models.Sequential()# 创建一个最简单的模型，类似前馈网络
model.add(layers.Dense(128,activation="relu",input_shape=(10000,)))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(46,activation="softmax"))# output[i]为输入属于第 i 类的概率
# 编译模型
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",# 使用分类交叉熵作为损失函数
              metrics=["accuracy"])# 使用精度作为指标
# 留出验证集 1000 个样本
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 模型 1
def train_model_1():
    history1 = model.fit(partial_x_train,
              partial_y_train,
              epochs=2,
              batch_size=64,
              validation_data=(x_val,y_val))
    # 绘制训练损失和验证损失
    history_dict = history1.history
    loss_values = history_dict['loss']  # 训练损失
    val_loss_values = history_dict['val_loss']  # 验证损失

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label="Training loss")  # bo表示蓝色原点
    plt.plot(epochs, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    plt.title("Train and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # 绘制训练精度和验证精度
    plt.clf()  # 清空图像
    acc = history_dict['acc']  # 训练精度
    val_acc = history_dict['val_acc']  # 验证精度
    plt.plot(epochs, acc, 'bo', label="Training Acc")  # bo表示蓝色原点
    plt.plot(epochs, val_acc, 'b', label="Validation Acc")  # b表示蓝色实线
    plt.title("Train and Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()  # 绘图

    plt.show()

# 模型 2
def train_model_2():
    model.fit(partial_x_train,
              partial_y_train,
              epochs=9,
              batch_size=512,
              validation_data=(x_val,y_val))
    res = model.evaluate(x_test,y_test)
    print("训练集上的精确度：{0}\n测试集上的精确度：{1}".format(res[0],res[1]))
    # 在新数据上预测结果
    predictions = model.predict(x_test)
    print("每条结果的shape: {0}".format(str(predictions[0].shape)))
    print("每条结果的概率之和：{0}".format(str(np.sum(predictions[0]))))
    print("预测所属类别：{0}".format(str(np.argmax(predictions[0]))))

# 使用模型
train_model_2()












