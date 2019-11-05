from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# 导入模型
model = load_model('cats_and_dogs_small_1.h5')
# model.summary()# 作为提醒

# 读取一张图像，并将图像转为张量
img_path = 'D:/KaggleDateset/kaggle/cats_and_dogs_small/test/cats/cat.2000.jpg'
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.

# print("img_tensor.shape: ",img_tensor.shape)# 检验其形状

# # 显示测试图像
# plt.imshow(img)
# plt.show()

# 实例化模型
layer_outputs = [layer.output for layer in model.layers[:8]]# 提取前8层的输出
activation_model = models.Model(input=model.input,outputs=layer_outputs)# 创建一个模型，给定模型输入，可以返回这些输出（一个输入，八个输出，即前八层的激活值）

# 以预测模式运行模型
activations = activation_model.predict(img_tensor)

# # 输出第一个卷积层的激活值
# first_layer_activation = activations[0]
# print("第一个卷积层的激活值: ",first_layer_activation.shape)
#
# # 将第四个通道可视化
# plt.matshow(first_layer_activation[0,:,:,4],cmap = 'viridis')
# plt.matshow(first_layer_activation[0,:,:,7],cmap = 'viridis')
# plt.show()

# 将每个中间激活的所有通道可视化
def visable():
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # 显示特征图
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]  # 特征图中的特征个数
        size = layer_activation.shape[1]  # 特征图的形状为（1,size,size,n_features）
        n_cols = n_features // images_per_row  # 向坐标轴左侧取整，每行 16 个激活通道，共 2 行，16 列（计算 行数）

        display_grid = np.zeros((size * n_cols, size * images_per_row))

        cc = 0
        for col in range(n_cols):  # 行数
            for row in range(images_per_row):  # 列数

                cc+=1

                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                #  对特征进行后处理，使其看起来更美观
                channel_image -= channel_image.mean()
                num = channel_image.std()
                print("**num: {}**".format(num))
                channel_image /= num
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype(
                    'uint8')  # float类型取值范围 ：-1 到1 或者 0到1  uint8类型取值范围：0到255
                display_grid[col * size:(col + 1) * size,
                row * size:(row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

        print(cc)

# 执行可视化
visable()









