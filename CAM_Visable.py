from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras import backend as K
import cv2

model = VGG16(weights='imagenet')# 包含密集连接分类器

# 处理图像
img_path = 'C:/Users/JOE/Pictures/CNN/creative_commons_elephant.jpg'
img = image.load_img(img_path,target_size=(224,224))# 大小为224×224 的 Python 图像库（PIL，Python imaging library）图像
x = image.img_to_array(img)# 形状为 (224, 224, 3) 的 float32 格式的 Numpy 数组
x = np.expand_dims(x,axis=0)# 添加一个维度，将数组转化为（1，224，224，3）形状的批量
x = preprocess_input(x)# 对批量进行预处理（按通道进行颜色标准化）

# # 预测类别
# preds = model.predict(x)
# print("****预测结果****")
# print(decode_predictions(preds,top=3)[0])
#
# # 被最大激活的元素
# print(np.argmax(preds[0]))# 非洲象类别索引编号为 386

# 方法：Grad-CAM 算法
def grad_cam():
    african_elephant_output = model.output[:,386]# 预测向量中的非洲象元素

    last_conv_layer = model.get_layer("block5_conv3")# block5_conv3 层的输出特征图， 它是 VGG16 的最后一个卷积层

    grads = K.gradients(african_elephant_output,last_conv_layer.output)[0]# 非洲象类别相对于 block5_conv3 输出特征图的梯度

    pooled_grads = K.mean(grads,axis=(0,1,2))# 形状为 (512,) 的向量，每个元素是特定特征图通道的梯度平均大小

    #  方法：对于给定的样本图像，返回 pooled_grads 和 last_conv_layer.output[0] 两个值
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

    pooled_grads_value,conv_layer_output_value = iterate([x])# 两个值都是 NP 数组

    for i in range(512):
        # 将特征图数组的每个通道乘以“这个通道对‘大象’类别的重要程度”
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value,axis=-1)# 得到特征图的逐通道平均值即为类激活的热力图

    heatmap = np.maximum(heatmap,0)# 将热力图标准化到 0~1 范围内
    heatmap /= np.max(heatmap)

    plt.matshow(heatmap)
    plt.show()

    return heatmap

# # 调用方法：生成热力图
# grad_cam()

# 方法：将热力图与原始图像叠加
def superimpose_origin():
    img = cv2.imread(img_path)# 用 CV2 加载原始图像
    heatmap = grad_cam()

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))# 将热力图的大小调整为与原始图像相同

    heatmap = np.uint8(255*heatmap)# 将热力图转化为 RGB 模式     255*heatmap

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)# 将热力图应用于原始图像

    superimposed_img = heatmap*0.4 + img# 0.4 是热力图强度因子

    cv2.imwrite('C:/Users/JOE/Pictures/CNN/superimposed_img_no_RGB.jpg',superimposed_img)# 保存

    plt.imshow(superimposed_img)
    plt.show()

# 调用方法：显示热力图与原始图像叠加
superimpose_origin()






































