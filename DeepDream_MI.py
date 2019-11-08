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
from keras.applications import inception_v3
from keras import backend as K
import scipy
from keras.preprocessing import image
import os

# 加载预训练的 Inception V3 模型
K.set_learning_phase(0)# 禁用训练
model = inception_v3.InceptionV3(weights='imagenet', # 使用预训练的ImageNet 权重 来加载模型
                                 include_top=False)# 构建不包括全连接层的Inception V3 网络

# 设置 DeepDream 配置
layer_contributions = {'mixed2': 0.2,
                       'mixed3': 3.,
                       'mixed4': 2.,
                       'mixed5': 1.5, }

# 将层的名称映射为层的实例
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 定义需要最大化的损失
loss = K.variable(0.) # 在定义损失时将层的贡献添加到这个标量变量中
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output# 获取层的输出

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

# 梯度上升过程
dream = model.input# 用于保存生成的梦境图像

grads = K.gradients(loss,dream)[0]# 计算损失相对于梦境图像的梯度

grads /= K.maximum(K.mean(K.abs(grads)),1e-7)# 将梯度标准化

outputs = [loss,grads]# 给定一张输出图像，设置 一个 Keras 函数来获取损失值和梯度值
# 方法
fetch_loss_and_grads = K.function([dream], outputs)

# 方法：获取损失值和梯度值
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 方法：运行 iterations 次梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values

    return x

# 方法：处理输入的图像
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

# 方法：将一个张量转换为有效图像
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 方法：设置图片大小
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

# 方法：保存图像到硬盘
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


# 在多个连续尺度上运行梯度上升
step = 0.01 # 梯度上升的步长
num_octave = 3 # 运行梯度上升的尺度个数
octave_scale = 1.4# 两个尺度之间的大小比例
iterations = 20# 在每个尺度上运行梯度上升的步数

max_loss = 10.# 如果损失增大到大于 10，我们要中断梯度上升过程，以避免得到丑陋的伪影

base_image_path = "C:/Users/JOE/Pictures/bing/1517651807698.png"

img = preprocess_image(base_image_path)# 将基础图像加载成一个 Numpy 数组

original_shape = img.shape[1:3]
successive_shapes = [original_shape]

for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
                   for dim in original_shape])
    successive_shapes.append(shape)

print("successive_shapes: ",successive_shapes)
# os.system("pause")





successive_shapes = successive_shapes[::-1]# 将形状列表反转，变为升序

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])# 将图像 Numpy 数组的 大小缩放到最小尺寸

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape) # 将梦境图像放大

    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)# 将原始图像的较小版本 放大，它会变得像素化
    same_size_original = resize_img(original_img, shape) #在这个尺寸上计算原始图像的高质量版本
    lost_detail = same_size_original - upscaled_shrunk_original_img# 计算丢失的细节

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='final_dream.png')








































