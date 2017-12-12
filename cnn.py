# coding:utf-8

'''
Author:wepon
Code:https://github.com/wepe

File:cnn.py
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
# 导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function

from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import np_utils

from data import load_sent_data

# 导入各种用到的模块组件

nb_epoch = 20
batch_size = 50
nb_class = 561
nb_width = 50
nb__sent = 5895
nb_train = 5000
print('nb_train', nb_train)
nb_dim = 128
nb_pos = 0
nb_dep = 0
nb_par = 0
path = "./data1/fz_charws_label_5894.txt"
w2vPath = "./data1/char2vec_128.bin"


# 14layer, one "add" represent one layer
def create_deep_cnn_model(nb_dim, nb_width):
    print("dim: " + str(nb_dim))
    model = Sequential()
    # model.add(Convolution2D(nb_dim,5,nb_dim, border_mode='valid',input_shape=(1, nb_width, nb_dim)))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=( 46,1)))

    model.add(Convolution1D(nb_filter=128,
                            filter_length=3,
                            border_mode="valid",
                            activation="tanh",
                            input_shape=(nb_width, nb_dim)))
    model.add(MaxPooling1D(pool_length=48))

    model.add(Flatten())
    model.add(Dense(256, init='normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class, init='normal'))
    model.add(Activation('softmax'))
    return model


############
# 加载数据
model = create_deep_cnn_model(nb_dim + nb_dep + nb_pos + nb_par, nb_width)
# data, label = load_data(path)
data, label = load_sent_data(path, w2vPath, nb__sent, nb_width, nb_dim, nb_pos, nb_dep, nb_par)
# label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
print("label shape BEFORE", label.shape)
label = np_utils.to_categorical(label, nb_class)
print("label shape AFTER", label.shape)

print("data shape", data.shape)
print("data: " + str(len(data)) + "\t" + "label: " + str(len(label)))

#############
# 开始训练模型
##############
sgd = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("compile")
# index = [i for i in range(len(data))]
# random.shuffle(index)
# data = data[index]
# label = label[index]
(X_train, X_val) = (data[0:nb_train], data[nb_train:])
(Y_train, Y_val) = (label[0:nb_train], label[nb_train:])

print('Training')
best_accuracy = 0.0
print('shape', X_val.shape)
print('shape', Y_val.shape)
model.fit(data, label, batch_size=320, epochs=100, shuffle=True, verbose=1, validation_split=0.1)
# model.fit(X_train, Y_train, batch_size=320, epochs=2, shuffle=True, verbose=1, validation_data=(X_val,Y_val))

# pickle.dump(model, open("./model.pkl", "wb"))
# val_loss, val_accuracy = model.evaluate(X_val, Y_val, batch_size=1)
