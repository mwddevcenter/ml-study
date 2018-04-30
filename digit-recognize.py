import numpy as np
import kNN
import os

def img2vector(filename):
    returnVect = np.zeros((1, 1024)) #创建一个1*1K的数组
    fr = open(filename)              #打开文件
    for i in range(32):         
        lineStr = fr.readline()      #读取前32行
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

t = img2vector('./digits/testDigits/0_13.txt')

def handWritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    traningMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        traningMat[i, :] = img2vector('./digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('./digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./digits/testDigits/%s' % fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, traningMat, hwLabels, 3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr) )
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is :%d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

handWritingClassTest()