
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam, SGD

import tensorflow.keras.backend as K

import h5py
import numpy as np

import argparse
import keras
import heapq
import scipy.io as scio


parser = argparse.ArgumentParser(description='')
parser.add_argument('--ModelDir',    dest='ModelDir',  default='siamese_test_model/', help='directory of Model')
parser.add_argument('--testCsi',    dest='testCsi',  default='siameseTestCsi.mat', help='CSI data File For Test')
parser.add_argument('--testLab',    dest='testLab',  default='siameseTestLab.mat', help='Label data File For Test')
parser.add_argument('--discretizeCsi',    dest='discretizeCsi',  default='data_discretizeCsi/', help='data_discretizeHand')

parser.add_argument('--test40Lab',    dest='test40Lab',  default='data_discretize40Lab/', help='data_discretize44Lab_HandGesture')

parser.add_argument('--outputDir',    dest='outputDir',  default='StateInference_result', help='File For Label Output')

args = parser.parse_args()

def LoopTime():
    userNum = '24'
    actNum = 1
    load_test_length = 8
    #discretizeData = sio.loadmat(args.discretizeCsi + userNum + 'cutfile3D_1' + '.mat')
    discretizeData = h5py.File(args.discretizeCsi + userNum + 'cutfile3D_1' + '.mat', 'r')
    discretizeData = discretizeData['data_']
    loopTime = int(discretizeData.shape[0]/load_test_length)

    return loopTime
class generate_dataPairs:
    def __init__(self,load_test_length=8):
        self.load_test_length=load_test_length

    def generate_testPairs(self,load_test_length):
        userNum = '24'
        actNum=1
        load_test_length=8
    # discretizeData = open(args.discretizeCsi+'user'+str(userNum) +'_iw_' + str(actNum) + '.mat')
    # discretizeLabel = open(args.test40Lab+'sample_40.mat')
        discretizeData = h5py.File(args.discretizeCsi + userNum + 'cutfile3D_1' + '.mat', 'r')
        #discretizeData = sio.loadmat(args.discretizeCsi + userNum + 'cutfile3D_1' + '.mat')

        #discretizeLabel = h5py.File(args.test40Lab+'sample_44.mat', 'r')

        discretizeLabel = scio.loadmat(args.test40Lab+'sample_44.mat')
        discretizeData = discretizeData['data_']
        discretizeLabel = discretizeLabel['data_']

        while(True):
            for i in range(int(discretizeData.shape[0]/load_test_length)):
                print('next迭代次数：'+str(i))
                start = i *load_test_length
                end = (i+1)*load_test_length if i != int(discretizeData.shape[0]/load_test_length) else -1
                lengthh = end-start

                discretizeTestData = np.zeros((lengthh, 3, 30, 120))
                discretizeTestData[:,:,:,:]=discretizeData[start:end,:,:,:]

        #DscData_l = np.zeros((lengthh*44,60,30,3))
                DscData_l = np.zeros((lengthh*44,3, 30, 120))
        #DscData_r = np.zeros((lengthh*44,60,30,3))
                DscData_r = np.zeros((lengthh*44,120, 30, 3))
                n=0
                for j in range(lengthh):
                    for k in range(44):
                        DscData_l[n,:,:,:] = discretizeTestData[j,:,:,:]
                        #DscData_r[n,:, :, :] = discretizeLabel[k, :, :, :]
                        DscData_r[n, :, :, :] = discretizeLabel[:, :, :,k]         ###sio
                        n=n+1

                test_l = np.transpose(DscData_l, axes=[0, 3, 2, 1])
                test_r = np.transpose(DscData_r, axes=[0, 1, 2, 3])

                yield test_l,test_r

def loadData(dataDir, testCsi):

    dataDir = dataDir + '/'  # dataDir = 'data/'
    (testCsi_l,testCsi_r)=divide_test_data(dataDir, testCsi)

    test_l = mat2Npy(testCsi_l, 'Test_l')
    test_r = mat2Npy(testCsi_r, 'Test_r')
    print('test_l.shape     ::', test_l.shape)
    print('test_r.shape      ::', test_r.shape)

    return (test_l,test_r)

def mat2Npy(fileName,typeName):
    data=fileName
    #print(fileName)     #print(typeName)    #print(typeName == 'Csi')
    if (typeName == 'Test_l'):
        dataReturn = np.transpose(data, axes=[0, 3, 2, 1])
    elif (typeName == 'Test_r'):
        dataReturn = np.transpose(data, axes=[0, 3, 2, 1])

    return dataReturn

def divide_test_data(data_dir, testCsi):
    for fileName in [testCsi]:
        path = data_dir + fileName
        mat = h5py.File(path, 'r')
        # mat = sio.loadmat(path)
        fileName = list(mat.keys())[0]  # print(list(mat.keys())[0]) #fileName = fileName.replace('.mat','')
        print(fileName)

        if (fileName == 'siameseTestCsi'):
            TestCsi = mat[fileName]
            # testCsi_l = TestCsi[:,:,:,0,:]    #SIO
            # testCsi_r = TestCsi[:,:,:,1,:]
            testCsi_l = TestCsi[:, 0, :, :, :]
            testCsi_r = TestCsi[:, 1, :, :, :]

    return (testCsi_l, testCsi_r)

def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result

class siamese_network():
    def __init__(self, initial_learning_rate = 0.001, batch_size = 32):          ###initial_learning_rate = 0.001
        self.lr = initial_learning_rate
        self.batch_size = batch_size
        self.get_model()

    def get_model(self):
        ###CNN
        W_init_1 = RandomNormal(mean=0, stddev=0.01)
        b_init = RandomNormal(mean=0.5, stddev=0.01)
        W_init_2 = RandomNormal(mean=0, stddev=0.2)

        input_shape = (120, 30, 3)  ######
        left_input = Input(input_shape)  ###Tensor("input_1:0", shape=(None, 105, 105, 1), dtype=float32)
        right_input = Input(input_shape)

        convnet = Sequential()

        convnet.add(Conv2D(96, (3, 3), strides=(4, 2), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=W_init_1, bias_initializer=b_init,kernel_regularizer=l2(2e-4)))  ###30*15*96
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))  ###30*15*96
        convnet.add(MaxPooling2D())  ###4*4*96
        convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4)))  ###8*8*192      4*4*192
        convnet.add(MaxPooling2D())  ###2*2*192
        convnet.add(Conv2D(192, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=W_init_1,bias_initializer=b_init, kernel_regularizer=l2(2e-4), name="Dense_2"))  ###8*8*192      2*2*192
        convnet.add(Flatten())
        convnet.add(Dense(4096, activation="sigmoid", kernel_initializer=W_init_2, bias_initializer=b_init,
                          kernel_regularizer=l2(1e-3)))

        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)

        merge_layer = Lambda(euclidean_dist)([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(merge_layer)
        self.model = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = SGD(lr=0.001, momentum=0.5)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    def test_pairs(self, testCsi_l, testCsi_r, n_way=44):
         correct_pred = 0
         X_l = testCsi_l
         X_r = testCsi_r
         j = 0

         siamese_predict_ema = open(args.outputDir + '/'+args.testCsi.replace(".mat", "") + "_predict_ema", "a")

         for i in range(0, len(X_l), n_way ):
             X_left, X_right = X_l[i: i + n_way], X_r[i: i + n_way]
             X_left, X_right = np.array(X_left), np.array(X_right)
             predictSeg, prob = self.test_one_shot(X_left, X_right)
             predictSeg=str(predictSeg)
             ###_________________________________________________________________
             siamese_predict_ema.write(predictSeg + "\t" + (predictSeg+".0000")+ "\t" + predictSeg + "\t" + (predictSeg+".0000") + "\n")
             print('X_left.shape:'+str(X_left.shape))
             print(X_left[1,1,1,1],X_right[1,1,1,1])
             print(predictSeg)
             print(prob)


         return predictSeg, prob



    def test_one_shot(self, X_left, X_right):
        prob = self.model.predict([X_left, X_right])
        """
        print(prob)
        print(np.argmax(prob))
        print(np.argmax(y))
        return
        """
        #predictSeg = np.argmax(prob) + 1
        # predict3Seg1 = heapq.nlargest(3,prob)        ###求最大的三个元素并排序
        prob = prob.tolist()
        predict3Seg = list(map(prob.index,heapq.nlargest(3,prob)))       ###求最大的三个元素的下标
        predict3Seg = [(predict % 4) for predict in predict3Seg]        ###对最大的三个元素下标进行取余处理
        predictSeg = np.argmax(np.bincount(predict3Seg))+1          ###求最大的三个元素中出现次数最多的元素的下标


        if predictSeg == 1:
            predictSeg = 2
        elif predictSeg == 2:
            predictSeg = 3
        elif predictSeg == 3:
            predictSeg = 1


        return predictSeg, prob

    def test_validation_acc(self, testCsi_l, testCsi_r, n_way=44):
         predictSeg, prob = self.test_pairs(testCsi_l, testCsi_r, n_way)  ###2021.5.01
         return (predictSeg,prob)

    def load_weights(self,ModelPath):
        self.model.load_weights(filepath=ModelPath)

if __name__ == "__main__":

    model = siamese_network(batch_size=16)
    model.load_weights(args.ModelDir + 'best_model.h5')

    loopTime = LoopTime()

    data_generate = generate_dataPairs(load_test_length=8)
    generate_dataPairs = generate_dataPairs().generate_testPairs(8)
    for i in range(loopTime):

        test_l, test_r = next(generate_dataPairs)
        printTest_l=test_l[1,1,1,1]
        printTest_r = test_r[1, 1, 1, 1]
        print(printTest_l,printTest_r)
        print('循环次数：'+str(i))

    #(testCsi_l, testCsi_r) = loadData(args.test40LabCsi,args.testCsi)

    # model.train_on_data(load_prev_model = True)
        (predictSeg,prob) = model.test_validation_acc(test_l,test_r, n_way=44)


    # print(acc)
    # print(predictSeg)
    # print(prob)
    #
