import numpy as np
import operator
import os
import struct
#KNN算法
def knn(k,testdata,traindata,labels):#(k,测试集,训练集,分类)
    traindatasize=traindata.shape[0]
    dif=np.tile(testdata,(traindatasize,1))-traindata#计算距离
    sqdif=dif**2
    sumsqdif=sqdif.sum(axis=1)
    distance=sumsqdif**0.5
    sortdistance=distance.argsort()#从小到大排列，结果返回元素位置
    count={}
    for i in range(k):
        vote=labels[sortdistance[i]]#统计每一类列样本的数量
        count[vote]=count.get(vote,0)+1
    sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)#取包含样本数量最多的那一类别
    return sortcount[0][0]
#加载数据,将文件转化为数组形式
def load_mnist(path, kind):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    return images, labels

#用测试数据调用KNN算法进行测试
def datatest():
    a=[]#准确结果
    b=[]#预测结果
    trainimages,trainlabels=load_mnist('D:\\train','train')
    testimages,testlabels=load_mnist('D:\\test','t10k')
    a=testlabels
    times=len(testlabels)
    for i in range(times):
        result=knn(3,testimages[i],trainimages,trainlabels)#将预测结果存在文本中
        b.append(int(result))
    return a,b

if __name__=='__main__':
    a,b=datatest()
num=0
for i in range(len(a)):
    if(a[i]==b[i]):
        num+=1
    else:
        print("预测失误: ",a[i],"预测为",b[i])
print(" 测试样本数为: ",len(a))
print("预测成功数为: ",num)
print("模型准确率为: ",num/len(a))