#-*- coding: utf-8 -*-
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def me_load_image(img_url):
    image = load_img(img_url,target_size=(128,128))
    image = img_to_array(image)
    image /= 255
    image = np.expand_dims(image,axis=0)
    return image


model = load_model("./facescore.model.h5")
res = me_load_image("./test/notGood.jpg")
a1 = model.predict_classes(res)
print "20", a1
