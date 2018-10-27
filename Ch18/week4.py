# -*- coding: utf-8 -*
import numpy as np

points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)

import matplotlib.pyplot as plt
from scipy import randn

z = np.sqrt(xs**2+ys**2)
print z

plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()
plt.title('Image plot of for a grid of values')
plt.draw()

# 将条件逻辑表达为数组运算
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
result = [(x if c else  y)for x,y,c in zip(xarr,yarr,cond)]
print result

result = np.where(cond,xarr,yarr)
print result

arr = randn(4,4)#todo:有错误randn
print arr

np.where(arr>0,2,-2)
np.where(arr>0,2,arr)

#用于布尔型数组的方法
arr = randn(100)
(arr > 0).sum() #正值的数量

bools = np.array([False,False,True,False])
bools.any()
bools.all()

#排序
arr = np.random(8)
print arr
print arr.sort()

large_arr = np.random(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]#5%分位数

#唯一化以及其他的集合逻辑
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
print np.unique(names)
ints = np.array([3,3,3,4,4,1,1,4,4])
print np.unique(ints)

sorted(set(names))

#线性代数
x = np.array([[1.,2.,3.]],[4.,5.,6.])
y = np.array([[6.,23.],[-1,7],[8,9]])
print x
print y






