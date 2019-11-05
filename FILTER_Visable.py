from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.applications import VGG16
from keras import backend as K

# 为过滤器的可视化定义损失张量
model = VGG16(weights='imagenet',
              include_top=False)# 舍弃密集连接分类器

# 方法： 将张量转化为有效图像
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1# 使张量的均值为 0 ，标准差为 0.1

    x += 0.5
    x = np.clip(x,0,1)# 将 X 裁剪到[0,1]区间

    x *= 255
    x = np.clip(x,0,255).astype('uint8')# 将 X 转化为 RGB 数组
    return x

# 方法： 构建一个损失函数，输入层的名称和过滤器的索引，返回一个有效的图像张量，表示此过滤器的激活最大化
def generate_pattern(layer_name,filter_index,size = 150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 获取损失相对于输入的梯度
    grads = K.gradients(loss, model.input)[0]  # 返回一个张量列表，本例中列表长度为 1，故只保留第一个元素，该元素是一个张量

    # 梯度标准化
    grads /= (K.sqrt(K.mean(
        K.square(grads))) + 1e-5)  # （做除法前加上 1e–5，以防不小心除以 0）将梯度张量除以其 L2 范数（张量 中所有值的平方的平均值的平方根）来标准化，确保了输入图像的更新大小始终位于相同的范围

    # 给定输入图像，计算损失张量和梯度张量的值
    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

    # 通过随机梯度下降让损失最大化
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128  # 初始化一张带有噪声的灰度图像

    step = 1.  # 每次梯度更新的步长
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step  # 沿着让损失最大化的方向调节输入图像

    img = input_img_data[0]
    return deprocess_image(img)

# 方法：生成某一层中所有过滤器响应模式组成的网格
def generate_layer_net():
    layer_name = 'block4_conv1'
    size = 64
    margin = 5

    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):#  遍历result网格的行
        for j in range(8):# 遍历result网格的列
            filter_img = generate_pattern(layer_name,i+j*8,size=size)

            horizontal_start = i*size+i*margin
            horizontal_end = horizontal_start + size
            vertical_start = j*size+j*margin
            vertical_end = vertical_start+size

            results[horizontal_start:horizontal_end,vertical_start:vertical_end,:] = filter_img# 保存结果到第 i,j 个方格中
    plt.figure(figsize=(20,20))
    plt.imshow(results)
    plt.show()




# # 测试 generate_pattern
# plt.imshow(generate_pattern('block3_conv1',0))
# plt.show()

# 测试 generate_layer_net
generate_layer_net()





















