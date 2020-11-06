####存放各种常见的激活函数的前向后向计算过程
import numpy as np
def sigmoid(z):
    s=1.0/(1.0+np.exp(-z))
    return s

def back_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def RELU(z):
    return np.maximum(z,0)

def back_RELU(z):
    return np.int64(z>0)

def softmax(z):
    '''
    另外的一种写法：
    temp=np.exp(z)
    return temp/np.sum(temp)
    '''
    temp=np.exp(z-np.max(z))
    return temp/np.sum(temp)

