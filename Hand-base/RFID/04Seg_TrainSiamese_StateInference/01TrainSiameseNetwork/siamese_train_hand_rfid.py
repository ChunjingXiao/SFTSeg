#加入忽略2021.4.13
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle as pkl
import cv2
import h5py
import time



from tqdm import tqdm
from math import ceil

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, AffineTransform, warp, rescale
from skimage.util import random_noise

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from statistics import mean


import h5py
import numpy as np
import scipy.io as sio
from numpy.random._mt19937 import MT19937
import tensorflow as tf
import argparse

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
batch_size = 32
### input_data_shape: [120,30,3,2,n]
### label_shape:[1,n]

###rewrite_from_deepSeg----------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataDir',    dest='dataDir',  default='data', help='directory of data')
parser.add_argument('--trainCsi',   dest='trainCsi', default='siameseTrainCsi.mat', help='CSI data File For Train')
parser.add_argument('--trainLab',   dest='trainLab', default='siameseTrainLab.mat', help='Label data File For Train')
parser.add_argument('--testCsi',    dest='testCsi',  default='siameseTestCsi.mat', help='CSI data File For Test')
parser.add_argument('--testLab',    dest='testLab',  default='siameseTestLab.mat', help='Label data File For Test')
args = parser.parse_args()


#def loadData(dataDir,trainCsi_l,trainCsi_r,trainLab, testCsi_l,testCsi_r, testLab):
def loadData(dataDir,trainCsi,trainLab, testCsi, testLab):
###主要是这个函数
#version2021.4.14
    dataDir = dataDir + '/'  # dataDir = 'data/'
    (trainCsi_l,trainCsi_r,trainLab, testCsi_l,testCsi_r, testLab)=divide_csi_data(dataDir,trainCsi,trainLab, testCsi, testLab)
    train_l = mat2Npy(trainCsi_l, 'Csi_l')    ###将mat矩阵转换为numpy矩阵
    train_r = mat2Npy(trainCsi_r, 'Csi_r')
    train_lab = mat2Npy(trainLab, 'Label')
    test_l = mat2Npy(testCsi_l, 'Csi_l')
    test_r = mat2Npy(testCsi_r, 'Csi_r')
    test_lab = mat2Npy(testLab, 'Label')

    print('train_l.shape     ::', train_l.shape)
    print('train_r.shape     ::', train_r.shape)
    print('train_lab.shape     ::', train_lab.shape)
    print('test_l.shape     ::', test_l.shape)
    print('test_r.shape      ::', test_r.shape)
    print('test_lab.shape      ::', test_lab.shape)

    # trainx = scaler(trainx)  # -----------------xiao--------normalize data--------------
    # testx = scaler(testx)
    return (train_l,train_r,train_lab, test_l,test_r, test_lab)

def mat2Npy(fileName,typeName):
# version2021.4.14
    #path= data_dir + fileName

    #mat=h5py.File(path,'r') #   print(mat.keys()) # mat = sio.loadmat(path)
    #fileName = list(mat.keys())[0] #print(list(mat.keys())[0]) #fileName = fileName.replace('.mat','')
    #fileName=segmentBaseTrainCsi
    #data=mat[fileName]
    data=fileName
    #print(fileName)     #print(typeName)    #print(typeName == 'Csi')
    if(typeName == 'Csi_l'):
        dataReturn= np.transpose(data,axes=[0,3,2,1])     ###调整axes使之成为格式[n,120,30,3],原格式为[120,30,3,n]
    elif(typeName == 'Csi_r'):
        dataReturn = np.transpose(data, axes=[0,3,2,1])
    elif (typeName == 'Label'):
        # data= np.transpose(data,axes=[1,0])
        # data = np.transpose(data, axes=[0, 1])            ###H5PY
        dataReturn = np.transpose(data, axes=[0, 1])
    print(dataReturn.shape)
    return dataReturn

def divide_csi_data(data_dir,trainCsi,trainLab, testCsi, testLab):
#version2021.4.14
    for fileName in [trainCsi,trainLab,testCsi,testLab]:
        path = data_dir + fileName
        mat = h5py.File(path, 'r')
        #mat = sio.loadmat(path)
        fileName = list(mat.keys())[0]  # print(list(mat.keys())[0]) #fileName = fileName.replace('.mat','')
        #fileName = fileName.replace('.mat', '')
         # fileName=segmentBaseTrainCsi
        ### input_data_shape: [120,30,3,2,n]
        ### label_shape:[1,n]
        print(fileName)
        if (fileName=='siameseTrainCsi'):
            TrainCsi = mat[fileName]
            print(TrainCsi)
            # trainCsi_l = TrainCsi[:,:,:,0,:]  SIO读入
            # trainCsi_r = TrainCsi[:,:,:,1,:]
            trainCsi_l = TrainCsi[:, 0, :, :, :]
            trainCsi_r = TrainCsi[:, 1, :, :, :]

        elif (fileName == 'siameseTrainLab'):
            trainLab = mat[fileName]
        elif(fileName=='siameseTestCsi'):
            TestCsi = mat[fileName]
            # testCsi_l = TestCsi[:,:,:,0,:]    #SIO读入
            # testCsi_r = TestCsi[:,:,:,1,:]
            testCsi_l = TestCsi[:, 0, :, :, :]
            testCsi_r = TestCsi[:, 1, :, :, :]
        elif(fileName == 'siameseTestLab'):
            testLab = mat[fileName]

    return (trainCsi_l,trainCsi_r,trainLab, testCsi_l,testCsi_r, testLab)

###rewrite_from_deepSeg----------------------------------------------------------------------------------------------------------------

class data_gen:

    def __init__(self, batch_size = 32, isAug = True):
        self.batch_size = batch_size
        self.isAug = isAug

#加载训练集
    def load_data_batch(self,trainCsi_l,trainCsi_r,trainLab):

        X_l,X_r,y = trainCsi_l,trainCsi_r,trainLab           ###这个地方要解决
        X_l,X_r,y = shuffle(X_l,X_r,y, random_state=0)              ###2021.5.1

        #load_batch = 1024    #
        load_batch = 512
        train_len = len(X_l)     #train_len=2*183160

        while(True):      ###分成每一次的训练量
            #循环次数： train_len/load_batch
            for i in range(int(train_len/load_batch)):
                #start和end为每次训练的区间长度
                start = i*load_batch
                end = (i+1)*load_batch if i != int(train_len/load_batch) else -1
                X_trainl = X_l[start:end]
                X_trainr = X_r[start:end]
                y_t = y[start:end]
###shuffle()函数可同时打乱几个序列？
                X_trainl, X_trainr, y_t = shuffle(X_trainl, X_trainr, y_t, random_state=0)   #shuffle函数将元素随机分配  ##或许可以用到
                #offset偏移量
                for offset in range(0, load_batch, self.batch_size):     #(0,1024,2*183160)
                    X_left, X_right, _y = X_trainl[offset:offset + self.batch_size], X_trainr[offset:offset + self.batch_size], y_t[offset:offset + self.batch_size]  # batch_size=32

                    X_left_batch, X_right_batch, y_batch = np.asarray(X_left), np.asarray(X_right), np.asarray(_y)

                    X_left_batch, X_right_batch, y_batch  = shuffle(X_left_batch, X_right_batch, y_batch, random_state = 0)

                    yield [X_left_batch, X_right_batch], y_batch     #类似return

    # 计算欧几里得距离
def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result

class siamese_network():
    def __init__(self, initial_learning_rate = 0.001, batch_size = 32):
        self.lr = initial_learning_rate
        self.batch_size = batch_size
        self.get_model()

    def get_model(self):
        ###要改的CNN从此处开始
        W_init_1 = RandomNormal(mean=0, stddev=0.01)  # tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。
        b_init = RandomNormal(mean=0.5, stddev=0.01)
        W_init_2 = RandomNormal(mean=0, stddev=0.2)
        # 输入数据的格式
        input_shape = (120, 30, 3)  ######
        left_input = Input(input_shape)  ###Tensor("input_1:0", shape=(None, 105, 105, 1), dtype=float32)
        right_input = Input(input_shape)
        # cnn网络
        convnet = Sequential()  # 初始化一个convnet模型
        # convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))       ###120*30*64
        # convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))       ###120*30*64
        # convnet.add(Conv2D(96, (3, 3), strides=(4, 2), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=W_init_1, bias_initializer=b_init, kernel_regularizer=l2(2e-4)))       ###30*15*96
        #
        # convnet.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))           ###30*15*96
        # convnet.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))
        # convnet.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))           ###30*15*96
        # #convnet.add(Conv2D(96, p p ((())), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=W_init_1, bias_initializer=b_init, kernel_regularizer=l2(2e-4)))       ###8*8*96
        # convnet.add(MaxPooling2D())     ###4*4*96
        # convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))          ###8*8*192      4*4*192
        # convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))          ###8*8*192      4*4*192
        # convnet.add(MaxPooling2D())     ###2*2*192
        # convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4), name="Dense_2"))          ###8*8*192      2*2*192
        # convnet.add(MaxPooling2D())     ###1*1*192
        # convnet.add(Flatten())
        # convnet.add(Dense(4096, activation="sigmoid", kernel_initializer=W_init_2, bias_initializer=b_init, kernel_regularizer=l2(1e-3)))           ###none*4096

        convnet.add(Conv2D(96, (3, 3), strides=(4, 2), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=W_init_1, bias_initializer=b_init,kernel_regularizer=l2(2e-4)))  ###30*15*96
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))  ###30*15*96
        convnet.add(MaxPooling2D())  ###4*4*96
        convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))  ###8*8*192      4*4*192
        convnet.add(MaxPooling2D())  ###2*2*192
        convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4), name="Dense_2"))  ###8*8*192      2*2*192
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="sigmoid", kernel_initializer=W_init_2, bias_initializer=b_init,kernel_regularizer=l2(1e-3)))

        encoded_l = convnet(left_input)  # siamese左通道输入
        encoded_r = convnet(right_input)  # siamese右通道输入
        ###结束
        merge_layer = Lambda(euclidean_dist)([encoded_l, encoded_r])  # 将输出的特征向量计算euclidean距离

        prediction = Dense(1, activation='sigmoid')(merge_layer)
        self.model = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = SGD(lr=0.01, momentum=0.5)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #计算参数acc
    def test_pairs(self, testCsi_l,testCsi_r,test_lab,n_way = 4):

        correct_pred = 0

        X_l = testCsi_l
        X_r = testCsi_r
        y = test_lab

        j = 0
        for i in range(0,len(X_l),n_way):
            X_left, X_right,_y  = X_l[i: i+n_way],X_r[i: i+n_way], y[i : i+n_way]

            X_left, X_right, _y = np.array(X_left), np.array(X_right), np.array(_y)

            correct,prob=self.test_one_shot(X_left, X_right, _y)        ##2021.5.01
            #correct_pred += self.test_one_shot(X_left, X_right, _y)
            correct_pred += correct

        acc =  correct_pred*100/(len(X_l)/n_way)                ##2021.5.01
        return acc,prob

    #测试oneshot结果是否正确
    def test_one_shot(self, X_left,X_right, y):
        prob = self.model.predict([X_left,X_right])
        """
        print(prob)
        print(np.argmax(prob))
        print(np.argmax(y))
        return
        """

        if np.argmax(prob) == np.argmax(y):
            return 1,prob
        else:
            return 0,prob

    def test_validation_acc(self,testCsi_l,testCsi_r,test_lab, n_way=4):
        test_acc,prob = self.test_pairs(testCsi_l,testCsi_r,test_lab,n_way)             ###2021.5.01
        return (test_acc,prob)
#训练过程
    def train_on_data(self, trainCsi_l,trainCsi_r,trainLab, testCsi_l,testCsi_r, testLab,load_prev_model = False ,best_acc = 0):

        model_json = self.model.to_json()

        #在model文件里更新参数
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        self.val_acc_filename = 'val_acc'

        self.v_acc = []
        self.train_metrics = []
        self.best_acc = best_acc
        self.model_details = {}
        self.model_details['acc'] = 0
        self.model_details['iter'] = 0
        self.model_details['model_lr'] = 0.0
        self.model_details['model_mm'] = 0.0
        linear_inc = 0.01
        self.start = 1
        self.k = 0

        if load_prev_model:
            self.continue_training()

        data_generator = data_gen(self.batch_size, isAug = True)
        train_generator = data_generator.load_data_batch(trainCsi_l,trainCsi_r,trainLab)
        batch_size=self.batch_size
        load_batch=512

        train_loss, train_acc = [],[]
        #迭代次数：1000000
        # for i in range(self.start,1000000):
        for i in range(self.start,100000):
            """
            if self.k==50:
                K.set_value(model.model.optimizer.learning_rate, K.get_value(model.model.optimizer.learning_rate) * 0.9)
                self.k = 0
            """
            start_time = time.time()
            X_batch, y_batch = next(train_generator)
            #print(X_batch[0].shape,X_batch[1].shape, y_batch.shape)
            #print(type(X_batch), type(y_batch))
            #return

            loss = self.model.train_on_batch(X_batch, y_batch)
            train_loss.append(loss[0])
            train_acc.append(loss[1])

            if i % 500 == 0:

                train_loss = mean(train_loss)  #mean函数求均值
                train_acc = mean(train_acc)
                self.train_metrics.append([train_loss,train_acc])

                #loss_data.append(loss)

                # val_acc  = self.test_validation_acc(wA_file, uA_file, n_way=20)
                val_acc,prob = self.test_validation_acc(testCsi_l,testCsi_r,testLab, n_way=4)
                prob=prob.reshape(1,4)              ###2021.5.01
                #val_acc = [wA_acc, uA_acc]
                self.v_acc.append(val_acc)
                # if val_acc[0] > self.best_acc:
                if val_acc > self.best_acc:
                    print('\n***Saving model***\n')
                    #self.model.save_weights("model_{}_val_acc_{}.h5".format(i,val_acc[0]))
                    self.model.save_weights("siamese_best_model/best_model.h5".format(i,val_acc))     ##保存模型权重
                    self.model_details['acc'] = val_acc
                    self.model_details['iter'] = i
                    self.model_details['model_lr'] = K.get_value(self.model.optimizer.learning_rate)
                    self.model_details['model_mm'] = K.get_value(self.model.optimizer.momentum)
                    #siamese_net.save(model_path)
                    self.best_acc = val_acc
                    with open(self.val_acc_filename, "wb") as f:
                        pkl.dump((self.v_acc,self.train_metrics), f)
                    with open('siamese_best_model/model_details.pkl', "wb") as f:
                        pkl.dump(self.model_details, f)  #dump函数将对象保存到文件中

                end_time = time.time()
                #输出在屏幕上的信息
                print('Iteration :{}  lr :{:.8f} momentum :{:.6f} batch_size:{} load_batch:{} avg_loss: {:.4f} avg_acc: {:.4f} val_acc :{:.2f} % time_taken {:.2f} s'.format(i,K.get_value(self.model.optimizer.learning_rate),K.get_value(self.model.optimizer.momentum),batch_size,load_batch,train_loss, train_acc,val_acc, end_time-start_time))
                print('prob:',prob)
                #
                train_loss, train_acc = [],[]

            if i % 5000 == 0:
                K.set_value(self.model.optimizer.learning_rate, K.get_value(self.model.optimizer.learning_rate) * 0.99)
                K.set_value(self.model.optimizer.momentum, min(0.9,K.get_value(self.model.optimizer.momentum) + linear_inc))

if __name__ == "__main__":
    # 加入忽略
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = siamese_network(batch_size = 16)
    (trainCsi_l, trainCsi_r, trainLab, testCsi_l, testCsi_r, testLab) = loadData(args.dataDir, args.trainCsi,args.trainLab, args.testCsi,args.testLab)

    #model.train_on_data(load_prev_model = True)
    model.train_on_data(trainCsi_l,trainCsi_r,trainLab, testCsi_l,testCsi_r, testLab,load_prev_model = False)







