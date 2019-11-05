import os,shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image # 图像预处理工具的模块
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 数据处理
original_dataset_dir = 'D:/KaggleDateset/kaggle'
base_dir = 'D:/KaggleDateset/kaggle/cats_and_dogs_small'

# os.mkdir(base_dir)

# 划分训练、验证、测试数据集
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
#
# # 猫的训练目录
train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
# # 狗的训练目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
# # 猫的验证目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
# # 狗的验证目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
# # 猫的测试目录
test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
# # 狗的测试目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
# # 复制猫的图像
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(2000, 2500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
# # 复制狗的图像
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(2000, 2500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)


# 查看文件复制是否成功
# print('total training cat images:', len(os.listdir(train_cats_dir)))
# print('total training cat images:', len(os.listdir(train_dogs_dir)))
# print('total training cat images:', len(os.listdir(test_cats_dir)))
# print('total training cat images:', len(os.listdir(test_dogs_dir)))
# print('total training cat images:', len(os.listdir(validation_cats_dir)))
# print('total training cat images:', len(os.listdir(validation_dogs_dir)))


# 模型 1
def train_model_1():
    # 搭建模型
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # 显示模型架构
    # model.summary()

    # 编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    # 数据预处理
    #  使用Python生成器将图像文件转为预处理好的张量批量
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # 将所有图像乘以1/255缩放

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    # 利用批量生成器拟合模型
    history = model.fit_generator(
        train_generator,  # 输入数据是生成器的输出
        steps_per_epoch=100,  # 每一轮从生成器中抽取的样本数量
        epochs=30,
        validation_data=validation_generator,  # 评估输入数据
        validation_steps=50  # 每次验证的样本数量
    )
    # 保存模型
    model.save('cats_and_dogs_small_1.h5')
    # 绘制训练中的损失曲线和精度曲线
    acc = history.history['acc']
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

# 返回ImageDataGenerator
def GetImageDataGenerator():
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    return datagen

# 显示几个随机增强之后的训练图像
def show_changed_images():
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    # 选择一张图像进行增强
    img_path = fnames[3]
    # 读取图像并调整大小
    img = image.load_img(img_path, target_size=(150, 150))
    # 将其转化为形状（150，150，3）的Numpy数组
    x = image.img_to_array(img)
    # 将其形状改变为(1,150,150,3)
    x = x.reshape((1,) + x.shape)

    i = 0
    datagen = GetImageDataGenerator()
    for batch in datagen.flow(x,batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i%4==0:
            break
        plt.show()

# 模型 2   利用数据增强生成器训练卷积神经网络
def train_model_2():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))# 添加Dropout
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       )# 对训练数据使用数据增强
    test_datagen = ImageDataGenerator(rescale=1./255)# 测试数据不能使用数据增强
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150,150),
                                                        batch_size=32,
                                                        class_mode='binary')# 因为使用了 binary_crossentropy 损失，所以需要用二进制标签
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=(150,150),
                                                            batch_size=32,
                                                            class_mode='binary')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=validation_generator,
                                  validation_steps=50)
    model.save('cats_and_dogs_small_2.h5')# 保存模型

    acc = history.history['acc'] # 绘制训练中的损失曲线和精度曲线
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

# 调用函数，显示几张随机生成的图片
# show_changed_images()

# 使用模型 2
train_model_2()
















