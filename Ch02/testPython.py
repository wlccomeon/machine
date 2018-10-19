# -*- coding: utf-8 -*
import numpy as np

#使用纯python做两个向量相加
def pythonsum(n):
    a = range(n)
    print (a)
    b = range(n)
    print (b)
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 2
        c.append(a[i]+b[i])
    return c



#使用Numpy做两个向量相加
def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c

import sys
from datetime import datetime

size = 1000
start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print (delta)

size = 1000
start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print (delta)

#numpy数组
a = np.arange(5)
a.dtype
a.shape

#创建多维数组
m = np.array([np.arange(2),np.arange(2)])
print m
print m.shape
print  m.dtype

np.zeros(10)
np.zeros((3,6))
np.empty((2,3,2))
np.arange(15)

#数组的选取
a = np.array([[1,2],[3,4]])
print a[0,0]
print a[0,1]
print a[1,0]
print a[1,1]

#创建自定义的数据类型
t = np.dtype([('name',np.str_,40),('munitems',np.int32),('price',np.float32)])
print t
print t['name']

itemz = np.array([('Meaning of life DVD',42,3.14),('Butter',13,2.72)],dtype=t)
print itemz[1]

#一维数组的索引与切片
a = np.arange(9)

#获取3到7之间的数据
print a[3:7]
#2 是步长
print a[:7:2]
#-1表示逆序
print a[::-1]

#同效与print a[3:7:2]
s = slice(3,7,2)
print a[s]

#同效与print a[::-1]
s = slice(None,None,-1)
print a[s]


#多维数组的切片与索引
b = np.arange(24).reshape(2,3,4)
print b.shape
print b

# 布尔型索引
print '布尔型索引'
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
print "names"
print names
print "data"
print data

print "names == 'Bob'"
print names == 'Bob'

print "data[names=='Bob']"
print data[names=='Bob']

print "data[names=='Bob',2:]"
print data[names=='Bob',2:]
print "data[names == 'Bob',3]"
print data[names == 'Bob',3]

print names != 'Bob'
print data[(names == 'Bob')]

#数组转置
print '数组转置'
arr = np.arange(15).reshape((3,5))

print arr
print arr.T

#从高维转低维
print '从高维转低维'
b = np.arange(24).reshape(2,3,4)
print b
print b.ravel()
print b.flatten()
b.shape = (6,4)
print b

print '转置的另一种写法'
print b.transpose()
b.resize((2.12))
print b



#if __name__ == '__main__':
    #print ("333")
    #print numpysum(3)

