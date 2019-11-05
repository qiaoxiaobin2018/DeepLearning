from keras.applications import VGG16
import numpy as np
import os,shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image # 图像预处理工具的模块
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 实例化 VGG16 卷积基
conv_base = VGG16(weights='imagenet',
                  include_top=False,# 是否自动包含Dense层
                  input_shape=(150,150,3))# 输入的图像张量的形状，若没有给定此参数，那么网络能够处理任意形状的输入
# conv_base.summary()

# 使用预训练的卷积基提取特征，并保存在硬盘上，作为Dense层的输入
base_dir = 'D:/KaggleDateset/kaggle/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# 方法：提取特征
def extract_features(directory,sample_count):
    features = np.zeros(shape=(sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    # 从文件中读
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150,150),
                                            batch_size=batch_size,#  每次处理 20 张
                                            class_mode='binary')
    i = 0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)# 经过处理得到训练数据的特征数组，shape=(4,4,512)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break
    return features,labels

# 提取特征
train_features,train_labels = extract_features(train_dir,2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 展平为 (samples, 8192)
train_features = np.reshape(train_features,(2000,4*4*512))
validation_features = np.reshape(validation_features,(1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))

# 定义 Dense 分类器
def train():
    model = models.Sequential()
    model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features,
                        train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features,validation_labels))

    acc = history.history['acc'] # 绘制结果
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

train()
























