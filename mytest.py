import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mydataset = np.array([[800, 2, 3], 
                      [600, 6, 7], 
                      [1000, 8, 9]])
labels = [1, 2, 3]
sqData = mydataset ** 2
sub = sqData - mydataset


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 获取每列的最小值， 矩阵
    maxVals = dataSet.max(0)  # 获取每列的最大值
    ranges = maxVals - minVals  # 最大值-最小值
    normDataset = np.zeros(np.shape(dataSet))  # 生成一个与dataSet相同大小的矩阵
    m = dataSet.shape[0]  # 获取dataSet的行数
    normDataset = dataSet - np.tile(minVals, (m, 1))  # 用每个值减去最小值
    normDataset = normDataset/np.tile(ranges, (m, 1))  # 然后除以(最大值-最小值)
    return normDataset, ranges, minVals

''' fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(sub[:, 1], sub[:, 2], 15.0 *
           np.array(labels), 15.0 * np.array(labels))
ax.scatter(sqData[:, 1], sqData[:, 2], 10.0 * np.array(labels), 10.0 * np.array(labels))
plt.show()
 '''

nD, r, min = autoNorm(mydataset)
print(list(nD))

