import tensorflow as tf
import keras as kr
import numpy as np
from keras.datasets import boston_housing
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

# 加载房价数据
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
# print(train_data.shape)
# print(test_data.shape)
# print(train_targets[400])

# 数据标准化(每一列，先减去平均值，再除以标准差) 好像是最常用的数字标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
# 使用函数构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation="relu",input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(1))# 标量回归 没有激活函数，所以可以预测任意范围内的值
    # compile
    model.compile(optimizer="rmsprop",
                  loss="mse",# 均方误差，回归问题常用的损失函数
                  metrics=['mae'])# 平均绝对误差，预测值与目标值之差的绝对值
    return model

# K折验证
k = 4
num_val_samples = len(train_data)//k # 向下（坐标轴向左）取整
# 定义训练 1
def train_1():
    num_epochs = 100
    all_score = []
    for i in range(k):
        print("Processing fold #", i)
        # 准备验证数据（第 K 个分区的数据）
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
        # 准备训练数据（除第 K 个分区之外的所有数据）
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        # 构建 Keras 模型
        model = build_model()
        model.fit(partial_train_data,
                  partial_train_targets,
                  epochs=num_epochs,
                  batch_size=1,
                  verbose=0)  # 静默模式
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_score.append(val_mae)

    print(all_score)
    print(np.mean(all_score))
# 定义训练 2
def train_2():
    num_epochs = 300
    all_mae_histories = []
    for i in range(k):
        print("Processing fold #", i)
        # 准备验证数据（第 K 个分区的数据）
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
        # 准备训练数据（除第 K 个分区之外的所有数据）
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        # 构建 Keras 模型
        model = build_model()
        history = model.fit(partial_train_data,
                            partial_train_targets,
                            validation_data=(val_data,val_targets),
                            epochs=num_epochs,
                            batch_size=1,
                            verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)
    # print(len(all_mae_histories),len(all_mae_histories[1]))
    # 计算所有轮次中的 K 折验证分数平均值
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    print(len(average_mae_history))
    # 绘制验证分数
    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()

# 训练最终模型
def final_train():
    model = build_model()
    model.fit(train_data,
              train_targets,
              epochs=80,
              batch_size=16,
              verbose=0)
    test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
    print("最终模型的平均绝对误差： ",test_mae_score)
# 执行训练 2
train_2()
# 使用训练好的参数训练最终模型
final_train()















