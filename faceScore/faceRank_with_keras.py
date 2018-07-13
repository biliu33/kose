# coding: utf8
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
import numpy as np



def getLabel(fileName):
    '''
    以一定的逻辑从文件名称当中获取标签,实际上也可以以其他方式,重要的是
    获取到图像,以及其对应的标签,这里的facescore是一个监督学习.
    '''
    pass


def getImgList(filedir):
    '''
    返回全路径
    '''
    pass


def load_dataset(filedir):
    """
    读取数据
    :param filedir:
    :return:
    """
    image_data_list = []
    label = []
    train_image_list = getImgList(filedir)
    for img in train_image_list:
        image = load_img(img, target_size=(128, 128))
        image_data_list.append(img_to_array(image))
        label.append(getLabel(img))
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255#it seems that we divide 255 so that the value ranges in (0, 1)(normaliztion)
    return img_data, label#the data,and label(if we want to rank it, what should i do?


def make_network():
    # 先加对应的层,然后加入对应的激活函数
    model = Sequential()#序贯模型(Sequential model)
    model.add(Conv2D(32, (3, 3), 
        padding='same',#补0策略, same保留边界处的卷积结果
        input_shape=(128, 128, 3)#128*128 的彩色RGB图像
        )
        )
    model.add(Activation('relu'))#activation function
    #Relu(rectified linear units)
    #researchers found out that ReLU layers work far better because the network is able to train a lot faster
    #(because of the computational efficiency) without making a significant difference to the accurancy.
    #It also helps to alleviate the vanishing gradient problem, which is the issue where the lower layers
    #of the network train very slowly because the gradient decreases exponentially through the layers.
    #related url:
    #https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    #this paper introduce much knowledge about relu function, coventional, pooling

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))#activation function(rectified linear unit)

    model.add(MaxPooling2D(pool_size=(2, 2)))#池化层,Pool_size代表两个方向上的下采样因子
    #reduce the computation cost and control overfitting
    model.add(Dropout(0.5))#to prevent overfitting with dropout method

    model.add(Flatten())#将多维输入一维化,常用于卷基层到全连接层的过渡
    #所以计算的结果为 32 * 2 * 2 = 64 * 2 = 128 ?

    model.add(Dense(128))#全连接层,实现运算
    model.add(Activation('relu'))#activation function

    model.add(Dropout(0.5))#to prevent overfitting with dropout method

    model.add(Dense(10))
    model.add(Activation('softmax'))#divide the picture into different class

    return model
    #summary: 前两层转化为一维的特征, 隐藏层没有加, 直接进行分类了, 所以可以在这里加一个例如64层的隐藏层试试


def main():
    train_x, train_y = load_dataset('./')
    train_y = np_utils.to_categorical(train_y)
    model = make_network()
    model.compile(loss='categorical_crossentropy',#损失函数为:分类的交叉熵函数?
            optimizer='adadelta',#why use this method to optimize the model?(no other info in document)
            metrics=['accuracy'])
    hist = model.fit(train_x, train_y, 
            batch_size=32,#批处理的"一批"的大小规模
            epochs=10,#训练过程中数据被迭代的次数
            verbose=1)
    model.save("./facescore.model.h5")#save the model


if __name__ == '__main__':
    main()
