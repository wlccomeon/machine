# -*- coding: utf-8 -*

import numpy as np

#线性代数
x = np.array([[1.,2.,3.],[4.,5.,6.]])
y = np.array([[6.,23.],[-1,7],[8,9]])
print x
print y
print x.dot(y) #等价于np.dot(x,y)

print np.dot(x,np.ones(3))
print np.ones(3)
print np.random.seed(12345)

from numpy.linalg import inv,qr
from scipy import randn

print 'x = randn(5,5) 随机生成5X5的矩阵'
x = randn(5,5)
print x

print 'mat = x.T.dot(x) 计算矩阵乘法'
mat = x.T.dot(x)
print mat

print 'inv(mat) inv 计算方正的逆'
inv(mat)
print inv(mat)

print 'mat.dot(inv(mat))'
mat.dot(inv(mat))
print mat.dot(inv(mat))

print 'qr(mat) 计算QR分解'
q,r = qr(mat)
print q
print r


###随机数生成
samples = np.random.normal(size=(4,4))
print samples

from random import normalvariate
N = 1000000
#get_ipython().magic(U'timeit samples = [normalvariate(0,1)for -in xrange(N)]')
#get_ipython().magic(U'timeit np.random.normal(size=N)')

#范例 ： 随机漫步
import random

print '方式1'
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
    print walk


print '方式2'
np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0,2,size=nsteps)
steps = np.where(draws>0,1,-1)
print ('steps',steps)
walk = steps.cumsum()
print walk

print ('min',walk.min())
print ('max',walk.max())

print (np.abs(walk)>= 10).argmax()

# 一次模拟多个随机漫步
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2,size=(nwalks,nsteps))
steps = np.where(draws>0,1,-1)
walks = steps.cumsum(1)
print ('walks',walks)
print ('max',walks.max())
print ('min',walks.min())

hits30 = (np.abs(walks) >= 30).any(1)
print hits30
print hits30.sum() #达到三十或-30的数量





