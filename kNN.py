import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.0], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :]  = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingDataLabels = file2matrix('datingTestSet2.txt')

def autoNorm(dataSet):
    minVals = dataSet.min(0) #获取每列的最小值， 矩阵
    maxVals = dataSet.max(0) #获取每列的最大值
    ranges = maxVals - minVals #最大值-最小值
    normDataset = np.zeros(np.shape(dataSet)) #生成一个与dataSet相同大小的矩阵
    m = dataSet.shape[0] #获取dataSet的行数
    normDataset = dataSet - np.tile(minVals, (m, 1)) #用每个值减去最小值
    normDataset = normDataset/np.tile(ranges, (m, 1)) #然后除以(最大值-最小值)
    return normDataset, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingDataLabels = file2matrix('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0] #行数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i, :], norMat[numTestVecs:m, :], datingDataLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingDataLabels[i]))
        if (classifierResult != datingDataLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of tim spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per years?"))
    datingDataMat, datingDataLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingDataLabels, 3)
    print("you will probably like this person: ", resultList[classifierResult -1])

''' import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
#plt.show()

ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 *
           np.array(datingDataLabels), 15.0 * np.array(datingDataLabels))
plt.show() '''
#datingClassTest()
#classifyPerson()

