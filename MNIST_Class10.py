from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models


# 导入数据集
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255

# 使用keras自带的方法将标签向量化
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 配置模型
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

# 显示卷积神经网络的架构
# model.summary()

# 将输出张量输入到密集连接分类器网络中，即Dense层的堆叠
# 先将 3D 张量转化为 1D 张量
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
# model.summary()

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',# 分类交叉熵（两个分布之间的相似程度）
              metrics=['accuracy'])
# 没有验证
model.fit(train_images,train_labels,epochs=5,batch_size=64)
# 在测试集上评估数据
test_loss,test_acc = model.evaluate(test_images,test_labels)
print("测试精度： ",test_acc)











