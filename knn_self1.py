import numpy as np
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
import re


TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows*cols)
        return np.minimum(data, 1)

def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # unmSamples 是数据集的行数
    init_shape = newInput.shape[0] #新加的数据的行数
    newInput = newInput.reshape(1, init_shape)   #(1,784)
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 4 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # 压到一维求和
    distance = squaredDist ** 0.25
    sortedDistIndices = np.argsort(distance)         #直接返回从小到大的索引

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex
# 搞出所有不想要的类别

def qiucan(ignored_set = [1,2],divided_slice = 10000,divided_part = 10):
    ignored_set =[]
    str1 = ((input('请输入你不想要的数字')))
    num = re.findall('\d+', str1)
    for i in num:
        ignored_set.append(int(i))
    divided_slice = (int(input('请输入你训练集的大小')))
    divided_part = (int(input('请输入你测试集的切分数')))
    return ignored_set,divided_slice,divided_part

def main():


    print("step 1: load data...")  # 784的shape
    train_x = extract_images('data/mnist/train_images')[0:divided_slice]
    train_y = extract_labels('data/mnist/train_labels')[0:divided_slice]
    test_x = extract_images('data/mnist/test_images')
    test_y = extract_labels('data/mnist/test_labels')
# --------------------------------------
    newtrainx = []
    newtrainy = []
    for i in range(len(train_y)):
        if train_y[i] in ignored_set:
            pass
        else:
            newtrainx.append(train_x[i])
            newtrainy.append(train_y[i])
    newtestx = []
    newtesty = []
    for i in range(len(test_y)):
        if test_y[i] in ignored_set:
            pass
        else:
            newtestx.append(test_x[i])
            newtesty.append(test_y[i])
    train_x, train_y, test_x, test_y = np.array(newtrainx), np.array(newtrainy), np.array(newtestx), np.array(newtesty)
#------------------------------------------

    numTestSamples = test_x.shape[0]    # 不切分的话就是10000
    matchCount = 0
    test_num = numTestSamples / divided_part
    for i in tqdm(range(int(test_num)),desc='正在预测'):
        predict = kNNClassify(test_x[i], train_x, train_y, 10)

        if predict == test_y[i]:
            matchCount += 1

    accuracy = float(matchCount) / test_num

    print('The classify accuracy is: %.2f%%' % (accuracy * 100))



if __name__ == '__main__':
    ignored_set = [1, 2]  # 你不想要哪些数字
    divided_slice = 10000  # 想要从数据集中拿多少个
    divided_part = 10  # 测试集的切分数
    ignored_set,divided_slice,divided_part = qiucan()
    main()
    # window = tk.Tk()
    # window.title('for_speaking')
    # window.geometry('500x400')
    # var1 = tk.StringVar()
    # l = tk.Label(window, bg='green', fg='yellow', font=('Arial', 12), width=50, textvariable=var1)
    # l.pack()
    # c = tk.Button(window, text='输入参数', font=('Arial', 12), width=10, height=1, command=qiucan)
    # c.pack()
    # e = tk.Variable()  # e
    # entry = tk.Entry(window, textvariable=e, show="")
    # entry.pack()
    # # e代表输入框的内容
    # # 设置值
    # e.set("请输入你不想要的数字")
    # # 取值
    # print(e.get())
    # print(entry.get())
    # print(type(e.get()))
    # print(type(entry.get()))
    # c = tk.Button(window, text='开始训练', font=('Arial', 12), width=10, height=1, command=main)
    # c.pack()
    # var2 = tk.StringVar()
    # l2 = tk.Label(window, text='训练准确度', font=('Arial', 12), width=50, textvariable=var2)
    # l2.pack()


    # window.mainloop()



