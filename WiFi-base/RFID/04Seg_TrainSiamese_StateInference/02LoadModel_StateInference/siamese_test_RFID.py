from tqdm import tqdm
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import h5py
import numpy as np
import argparse
import heapq

from sklearn import preprocessing

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ModelDir',    dest='ModelDir',  default='siamese_test_model/', help='directory of model')

parser.add_argument('--testCsi',    dest='testCsi',  default='siameseTestCsi.mat', help='CSI data File For Test')
parser.add_argument('--testLab',    dest='testLab',  default='siameseTestLab.mat', help='Label data File For Test')
parser.add_argument('--discretizeCsi',    dest='discretizeCsi',  default='data_discretizeCsi/', help='data_discretizeCsi')
parser.add_argument('--test40Lab',    dest='test40Lab',  default='data_discretize40Lab/', help='data_discretize40Lab')
parser.add_argument('--csiFile',   dest='csiFile', default='RFID1_30data6_1.mat', help='CSI data File Name')

parser.add_argument('--outputDir',    dest='outputDir',  default='StateInference_result', help='File For Label Output')

args = parser.parse_args()

def LoopTime():
    userNum = 1
    actNum = 1
    load_test_bath = 8
    discretizeData = h5py.File(args.discretizeCsi + args.csiFile, 'r')
    discretizeData = discretizeData['data_']
    loopTime = int(discretizeData.shape[0]/load_test_bath)

    return loopTime
class generate_dataPairs:
    def __init__(self,load_test_length=8):
        self.load_test_length=load_test_length

    def generate_testPairs(self, load_test_length):
        userNum = 1
        actNum = 1
        load_test_length = 8
        # discretizeData = open(args.discretizeCsi+'user'+str(userNum) +'_iw_' + str(actNum) + '.mat')
        # discretizeLabel = open(args.test40Lab+'sample_40.mat')
        discretizeData = h5py.File(args.discretizeCsi + args.csiFile, 'r')

        # discretizeData = scio.loadmat(args.discretizeCsi+'user'+str(userNum)+'_iw_'+str(actNum)+'.mat')
        discretizeLabel = h5py.File(args.test40Lab + 'sample_48.mat', 'r')
        # discretizeLabel = scio.loadmat(args.test40Lab+'sample_48.mat')
        discretizeData = discretizeData['data_']
        discretizeLabel = discretizeLabel['data_']

        while (True):
            for i in range(int(discretizeData.shape[0] / load_test_length)):
                print('Step：' + str(i))
                start = i * load_test_length
                end = (i + 1) * load_test_length if i != int(discretizeData.shape[0] / load_test_length) else -1
                lengthh = end - start

                discretizeTestData = np.zeros((lengthh, 3, 30, 120))
                discretizeTestData[:, :, :, :] = discretizeData[start:end, :, :, :]

                # DscData_l = np.zeros((lengthh*40,120,30,3))
                DscData_l = np.zeros((lengthh * 48, 3, 30, 120))
                # DscData_r = np.zeros((lengthh*40,120,30,3))
                DscData_r = np.zeros((lengthh * 48, 3, 30, 120))
                n = 0
                for j in range(lengthh):
                    for k in range(48):
                        DscData_l[n, :, :, :] = discretizeTestData[j, :, :, :]
                        # DscData_r[n,:, :, :] = discretizeLabel[ :, :, :,k]
                        DscData_r[n, :, :, :] = discretizeLabel[k, :, :, :]
                        n = n + 1

                test_l = np.transpose(DscData_l, axes=[0, 3, 2, 1])
                # test_r = np.transpose(DscData_r, axes=[0, 1, 2, 3])
                test_r = np.transpose(DscData_r, axes=[0, 3, 2, 1])

                yield test_l, test_r
def normalizeData(testCsi_l, testCsi_r):

    nm_testCsi_l = np.zeros((len(testCsi_l), 120, 30, 3))
    nm_testCsi_r = np.zeros((len(testCsi_l), 120, 30, 3))

    min_max_scaler = preprocessing.MinMaxScaler()

    for i in range(len(testCsi_l)):
        for j in range(3):
            nm_testCsi_l[i, :, :, j] = min_max_scaler.fit_transform(testCsi_l[i, :, :, j])
            nm_testCsi_r[i, :, :, j] = min_max_scaler.fit_transform(testCsi_r[i, :, :, j])

    return ( nm_testCsi_l, nm_testCsi_r)
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
        dataReturn = np.transpose(data, axes=[0, 3, 2, 1])  ###调整axes使之成为格式[n,120,30,3],原格式为[120,30,3,n]
    elif (typeName == 'Test_r'):
        dataReturn = np.transpose(data, axes=[0, 3, 2, 1])

    return dataReturn

def divide_test_data(data_dir, testCsi):
    # version2021.4.14
    for fileName in [testCsi]:
        path = data_dir + fileName
        mat = h5py.File(path, 'r')
        # mat = sio.loadmat(path)
        fileName = list(mat.keys())[0]  # print(list(mat.keys())[0]) #fileName = fileName.replace('.mat','')
        print(fileName)

        if (fileName == 'siameseTestCsi'):
            TestCsi = mat[fileName]
            # testCsi_l = TestCsi[:,:,:,0,:]    #SIO读入
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
        # 输入数据的格式
        input_shape = (120, 30, 3)  ######
        left_input = Input(input_shape)  ###Tensor("input_1:0", shape=(None, 105, 105, 1), dtype=float32)
        right_input = Input(input_shape)
        # cnn网络
        convnet = Sequential()  # 初始化一个convnet模型

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

    # 计算参数acc
    def test_pairs(self, testCsi_l, testCsi_r, n_way=48):
         correct_pred = 0
         X_l = testCsi_l
         X_r = testCsi_r
         j = 0

         siamese_predict_ema = open(args.outputDir + '/' + args.testCsi.replace(".mat", "") + "_" + args.csiFile.replace(".mat", "") + "_predict_ema","a")

         for i in range(0, len(X_l), n_way):
             X_left, X_right = X_l[i: i + n_way], X_r[i: i + n_way]
             X_left, X_right = np.array(X_left), np.array(X_right)
             predictSeg, prob = self.test_one_shot(X_left, X_right)
             predictSeg=str(predictSeg)

             siamese_predict_ema.write(predictSeg + "\t" + (predictSeg+".0000")+ "\t" + predictSeg + "\t" + (predictSeg+".0000") + "\n")

         return predictSeg, prob

    def test_one_shot(self, X_left, X_right):
        prob = self.model.predict([X_left, X_right])
        """
        print(prob)
        print(np.argmax(prob))
        print(np.argmax(y))
        return
        """

        prob = prob.tolist()
        predict3Seg = list(map(prob.index,heapq.nlargest(3,prob)))
        predict3Seg = [(predict % 4) for predict in predict3Seg]
        predictSeg = np.argmax(np.bincount(predict3Seg))+1


        if predictSeg == 1:
             predictSeg = 2
        elif predictSeg == 2:
             predictSeg = 3
        elif predictSeg == 3:
             predictSeg = 1


        return predictSeg, prob

    def test_validation_acc(self, testCsi_l, testCsi_r, n_way=4):
         predictSeg, prob = self.test_pairs(testCsi_l, testCsi_r, n_way)
         return (predictSeg,prob)

    def load_weights(self,ModelPath):
        self.model.load_weights(filepath=ModelPath)

if __name__ == "__main__":

    model = siamese_network(batch_size = 16)
    model.load_weights(args.ModelDir+'best_model.h5')

    loopTime = LoopTime()

    data_generate = generate_dataPairs(load_test_length=8)
    generate_dataPairs = generate_dataPairs().generate_testPairs(8)

    for i in tqdm(range(loopTime)):
        test_l, test_r = next(generate_dataPairs)
        printTest_l = test_l[1, 1, 1, 1]
        printTest_r = test_r[1, 1, 1, 1]
        (predictSeg,prob) = model.test_validation_acc(test_l,test_r, n_way=48)


