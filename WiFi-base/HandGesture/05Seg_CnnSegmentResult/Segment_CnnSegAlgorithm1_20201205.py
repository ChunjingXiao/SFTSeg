# -*- coding: utf-8 -*-
# Algorithm 1: CNN-based activity segmentation algorithm
# Input: Trained state inference model, CSI data, window size w, length for calculating the mode m
# Output: Start point t_{start} and end point t_{end} of the activity

import os
import time

import numpy as np
import sys
import math
#import h5py
import queue
#import simple_usual1

#import scipy.io as scio
#import hdf5storage

import argparse

#parser is used to accept parameters from commandlines,such as seting epoch=10:python train_CSI.py --epoch 10 
parser = argparse.ArgumentParser(description='')
#parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--dataDir',   dest='dataDir', default='data', help='directory of data')
parser.add_argument('--StateFile', dest='StateFile', default='StateLabel_DiscretizeCsi_1_5/', help='directory of data')

args = parser.parse_args()

lenAction = round(60) #动作大小
whichColumn = 0 #whichCol = args.whichCol; select the vaule of which column

#计算i之前更长一段state labels的众数
def longQueueModeRatePrevious(currentQ,predictResult,i):
    selectLong1 = 24
    selectLong2 = 24
    qBig   = currentQ.copy()
    qBig.clear()
    for kk in range(i-7- selectLong1 - 1, i - 7):
        qBig.append(predictResult[kk,whichColumn])
    countsBig1 = np.bincount(qBig)
    modePrev1 = np.argmax(countsBig1)
    
    qBig.clear()
    for kk in range(i-7- selectLong2 - 1 -selectLong1, i - 7 - selectLong1):
        qBig.append(predictResult[kk,whichColumn])
    countsBig2 = np.bincount(qBig)
    modePrev2 = np.argmax(countsBig2)    
    '''
    print('countsBig            :', countsBig)
    print('qBig                :', qBig)      
    print('i+120  ---Start--- :', i+120)  
    print('countsBig[modeBig]/float(20)                :', countsBig[modeBig]/float(selectLong )) 
    '''
    return (modePrev1, modePrev2, countsBig1[modePrev1]/float(selectLong1))

#计算i之后更长一段state labels的众数
def longQueueModeRateNext(currentQ,predictResult,i):
    selectLong1 = 24
    selectLong2 = 24
    qBig   = currentQ.copy()
    qBig.clear()
    for kk in range(i -3,i -3 + selectLong1  + 1):
        qBig.append(predictResult[kk,whichColumn])
    countsBig1 = np.bincount(qBig)
    modeNext1 = np.argmax(countsBig1)
        
    qBig.clear()
    for kk in range(i -3 + selectLong1 + 1,i -3 + selectLong1  + 1 + selectLong2):
        qBig.append(predictResult[kk,whichColumn])
    countsBig2 = np.bincount(qBig)
    modeNext2 = np.argmax(countsBig2)    
    
    '''
    print('countsBig            :', countsBig)
    print('qBig                :', qBig)      
    print('i+120  ---End----- :', i+120)  
    print('countsBig[modeBig]/float(20)                :', countsBig[modeBig]/float(selectLong + 10)) 
    '''
    return (modeNext1, modeNext2, countsBig1[modeNext1]/float(selectLong1))
def actionStartEndPoints(predictDataDir):

    stateLabels = np.loadtxt(predictDataDir) 
    print('predictResult.shape    ::', stateLabels.shape) 
    #print(predictResult[4,1])
    #print('len(predictResult)-10    ::', len(predictResult)-10) 
    
    
    qDetectChange = queue.Queue(maxsize=11) #create a queue with length=10 for detecting whether state labels are changed
    for i in range(11):
        qDetectChange.put(stateLabels[i,whichColumn],block=False)
    #print('qSmall.qsize()   :', qSmall.qsize())
    #print('qSmall.queue     :', qSmall.queue)
      
    dominatorRateThrehold = 0.49  # (the number of dominated values in the queue) / (length of the queue)
    startEndMark = np.zeros([11,2]) # startEndMark[0,0] is the start,  and startEndMark[0,1] is the end of the first activity
    lastMode = 0    
    currentMode = 0
    whichSample = 0  # which activity of 11 activities
    oneSampleStartMark = False
    for i in range(11,len(stateLabels)):
        
        currentQ =qDetectChange.queue
        #print(currentQ[1])
        
        #compute the mode: here the mode is the number which appears most often in the set        
        counts = np.bincount(currentQ)        
        currentMode = np.argmax(counts)
        #print('counts            :', counts)
        #print('np.argmax(counts) :', np.argmax(counts))        
        
        if(lastMode != currentMode and lastMode == 1): #----detect start points
            if(not oneSampleStartMark):
                (modePrev1, modePrev2, dominatorRatePrev) = longQueueModeRatePrevious(currentQ,stateLabels,i)
                (modeNext1, modeNext2, dominatorRateNext) = longQueueModeRateNext(currentQ,stateLabels,i)
                if((modePrev1 == 1 or modePrev2 == 1) and (modeNext1 == 2 or modeNext1 == 3) and \
                   (modeNext2 == 2 or modeNext2 == 3) and dominatorRatePrev > dominatorRateThrehold):
                    startEndMark[whichSample,0] = i-6+60 # for each file, it cuts first 60.
                    oneSampleStartMark = True
        if(lastMode != currentMode and currentMode == 1): #----detect end points
            if(oneSampleStartMark):
                (modePrev1, modePrev2, dominatorRatePrev) = longQueueModeRatePrevious(currentQ,stateLabels,i)
                (modeNext1, modeNext2, dominatorRateNext) = longQueueModeRateNext(currentQ,stateLabels,i)
                if((modePrev1 == 4 or modePrev2 == 4) and (modeNext1 == 1 and modeNext2==1)  and dominatorRatePrev > dominatorRateThrehold):
                    #if(i-7+120 - 120 - startEndMark[whichSample,0] > 30):
                        startEndMark[whichSample,1] = i-7+60 - 60
                        oneSampleStartMark = False
                        whichSample += 1
                        if(startEndMark[whichSample-1,1] - startEndMark[whichSample-1,0] < 20): #-----for--user2_iw_5_predict_ema
                            startEndMark[whichSample,0] = 0
                            startEndMark[whichSample,1] = 0
                            whichSample -= 1
                            
                        if(whichSample >= 11):               #-----for--------user5_ph_1
                            if((startEndMark[whichSample-1,1] - startEndMark[whichSample-1,0] \
                               + startEndMark[whichSample-2,1] - startEndMark[whichSample-2,0]) < 60):
                                startEndMark[whichSample-2,1] = startEndMark[whichSample-1,1]
                                startEndMark[whichSample,1] = 0
                                whichSample -= 2
                        
        if(oneSampleStartMark):      #----add only for user4_rp_1_predict_ema----------         
            if((i - startEndMark[whichSample,0]) > 400 and currentMode == 1):    
                startEndMark[whichSample,1] = i-7+60 - 60 -60
                oneSampleStartMark = False
                whichSample += 1
        #if(lastMode != currentMode and currentMode == 1 and lastMode == 4 and oneSampleStartMark):
        #    startEndMark[whichSample,1] = i-7+120 -60
        #    oneSampleStartMark = False
        #    whichSample += 1            
            
        qDetectChange.get(block=False)
        qDetectChange.put(stateLabels[i,whichColumn],block=False)
        
        lastMode = currentMode
        if(whichSample >= 11):
            break
    #if(predictDataDir.find('user2_iw_5_predict_ema') >= 1): #-----there are some problems for--user2_iw_5_predict_ema
    #    startEndMark = np.array([[390,585],[895,1086],[1316,1524],[1858,2060],[2381,2597],\
    #                    [2930,3185],[2593,3866],[4306,4591],[4975,5265],[5675,5595]] )  #--right one----
        
        #startEndMark = np.array([[390,785],[895,1286],[1316,1724],[1858,2060],[2381,2797],\
        #                [2930,3485],[2593,3866],[4306,4791],[4975,5565],[5675,5695]] )   #--wrong one----
         
    print('startEndMark:\n', str(startEndMark))
    return startEndMark


    

def main(_):

    #SIZE_LABEL = []
    #predResultDir = 'StateFile/'
    predResultDir = args.StateFile + '/'
    segmentResultDir='SegmentResultOneByOne/'
    saveDir='SegmentResultCombine/'
    
    # delete all the files in segmentResultDir
    filelist=os.listdir(segmentResultDir) 
    for f in filelist:   
        filepath = os.path.join(segmentResultDir,f) 
        if os.path.isfile(filepath):
            os.remove(filepath)
            
    #allFileName = ['user1_iw_6','user1_ph_6','user1_rp_6','user1_sd_6','user1_wd_6']
    allFileName = []
    for root, dirs, files in os.walk(predResultDir):
        for name in files:
            allFileName.append(name.replace('_predict_ema',''))
            #formatTweets(os.path.join(root, name),fout)
            
            
    for outFileName in allFileName :
        print('-----------------', outFileName,'-----------------')
        #outFileName = 'user1_iw_6.mat'
        
        predictDataDir = predResultDir + outFileName + '_predict_ema' # 'user1_iw_6.mat_predict_ema'
        startEndPoints = actionStartEndPoints(predictDataDir)
        
        predictDataDir = moveStepForPoll(startEndPoints)
        #predictDataDir = predResultDir + 'user1_ph_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_rp_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_sd_6.mat_predict_ema'
        #predictDataDir = predResultDir + 'user1_wd_6.mat_predict_ema'
        '''
        data_dir = 'Data_CsiAmplitudeCut/'
        outFileName = outFileName + '.mat'
        originalFile= data_dir + outFileName[:5] +'/' + '55' + outFileName # 'Data_CsiAmplitudeCut/user1/55user1_iw_1.mat'
        size_labelOne = actionSampleExtract(startEndPoints,originalFile,outFileName,segmentResultDir)
        '''
        saveStartEndPoints(segmentResultDir,outFileName,startEndPoints) # save start and end points in .csv file
    
    #saveDir='SegmentResultDataCombine/'
    #combineStartEndPoints(saveDir,segmentResultDir)    
    #combineCsiLabel(saveDir,segmentResultDir)  #
    print('---------done------')
def moveStepForPoll(startEndPoints):
    moveStep =  0
    moveStep =  1
    moveStep =  2  # allData: testAcc2=87.50 testF12=89.60 testAcc2E=87.50 testF12E=89.47
    moveStep =  3  # allData: testAcc2=88.28 testF12=90.96 testAcc2E=88.67 testF12E=91.11
    moveStep =  4  # allData: testAcc2=89.06 testF12=91.81 testAcc2E=89.84 testF12E=91.88
    moveStep =  4
    len_startEndPoints=len(startEndPoints)
    for i in range(0,len_startEndPoints):
        startEndPoints[i,0] += moveStep
        startEndPoints[i,1] += moveStep
    return startEndPoints
def saveStartEndPoints(segmentResultDir,outFileName,startEndPoints):
    fStartEnd = open(segmentResultDir+ '/'+ outFileName+'.csv','w')
    len_A=len(startEndPoints) #10个动作
    for i in range(0,len_A):
        fStartEnd.write(str(i+1) + ',' + str(int(startEndPoints[i,0])) + ',' + str(int(startEndPoints[i,1])) + '\n')
    fStartEnd.close()
def combineStartEndPoints(saveDir,segmentResultDir):
    fUser1 = open(saveDir+ '/'+'user1ManualSegment.csv','w')
    fUser2 = open(saveDir+ '/'+'user2ManualSegment.csv','w')
    fUser3 = open(saveDir+ '/'+'user3ManualSegment.csv','w')
    fUser4 = open(saveDir+ '/'+'user4ManualSegment.csv','w')
    fUser5 = open(saveDir+ '/'+'user5ManualSegment.csv','w')
    fopenAll = [fUser1,fUser2,fUser3,fUser4,fUser5]
    allFiles = os.listdir(segmentResultDir)
    kk = 0
    for oneFile in allFiles:
        if(oneFile.split('.')[-1] == 'csv'):
            print(oneFile)
            fStartEnd = open(segmentResultDir+ '/'+ oneFile)
            for current in fStartEnd:    
                #print(math.floor(kk/30))
                fopenAll[math.floor(kk/30)].write(str(kk+1) + ',' + current.replace('\n',',') + oneFile.replace('.csv','') + '\n')            
            kk+=1 
    
if __name__ == '__main__':
    main(args)

